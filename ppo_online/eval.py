#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from behavioural_cloning.eval_tokenizer_latent_bc_exact import (
    _get_wandb_mode,
    create_run_directory,
    write_episode_video,
)
from behavioural_cloning.train_base import TokenizerBackbone, load_tokenizer_encoder
from ppo_online.networks import TokenizerLatentBCPPOActorCritic
from ppo_online.render import (
    PushTDenseRewardWrapper,
    StridedObservationStackWrapper,
    TokenizerLatentObsWrapper,
    extract_checkpoint_args,
    extract_checkpoint_state_dict,
    extract_state_array,
    load_state_dict_safe,
)
from ppo_online.train import TrainConfig, resolve_device
from ppo_online.train import OpenLoopChunkExecutionWrapper
from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, PUSHT_FIXED_TARGET_POSE, make_pusht_env
from ppo_online.model_paths import resolve_ppo_checkpoint_path, resolve_tokenizer_path
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


@dataclass(frozen=True)
class PPOEvalConfig:
    checkpoint: str
    device: str = "auto"
    stochastic: bool = False
    execution_mode: str = "open-loop"
    replan_interval: int = 1
    temporal_ensemble_decay: float = 0.01
    env_id: str = DEFAULT_PUSHT_ENV_ID
    episodes: int = 20
    seed: int = 42
    max_episode_steps: int = 300
    fixed_target_pose: tuple[float, float, float] = tuple(PUSHT_FIXED_TARGET_POSE.tolist())
    fixed_target_block_success: bool = True
    fixed_target_max_reset_attempts: int = 100
    agent_block_coef: float = 0.0
    block_start_radius: float | None = None
    video: bool = False
    video_fps: int = 10
    video_resolution: int = 512
    output_root: str = "runs/evaluations"
    run_name: str | None = None
    tokenizer_path: str | None = None
    max_step_pixels: float = 15.0
    action_mode: str = "absolute"
    frame_stride: int = 5
    reward_mode: str = "dense"
    wandb: bool = False
    wandb_project: str = "pusht-ppo"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return slug or "eval"


def _write_metrics(result: dict, run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.json"
    payload = {
        "config": result["config"],
        "episodes": result["episodes"],
        "summary": result["summary"],
        "artifacts": {
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "video_dir": str(run_dir / "videos") if result["config"]["record_video"] else None,
        },
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Saved evaluation metrics to: {metrics_path}")
    return metrics_path


def _init_wandb(args: argparse.Namespace, config: dict):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb evaluation logging was requested, but wandb is not installed.")
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=_get_wandb_mode(args),
        config=config,
    )


def _success_from_info(info: dict, terminated: bool) -> float:
    for key in ("success", "is_success", "task_success", "block_success"):
        if key in info:
            return float(np.asarray(info[key]).squeeze())
    return float(bool(terminated))


def _find_base_state(env: gym.Env) -> np.ndarray:
    cursor = env
    visited: set[int] = set()
    while cursor is not None and id(cursor) not in visited:
        visited.add(id(cursor))
        state = getattr(cursor, "state", None)
        if state is not None:
            return extract_state_array(state)
        cursor = getattr(cursor, "env", None)
    raise RuntimeError("Could not find base PushT state in wrapped evaluation environment.")


def infer_ppo_latent_architecture(state_dict: dict[str, torch.Tensor], payload: dict | None = None) -> dict[str, int]:
    if isinstance(payload, dict):
        contract = payload.get("contract")
        if isinstance(contract, dict):
            required = ("latent_dim", "frame_stack", "action_dim", "action_chunk_size", "hidden_dim")
            if all(key in contract for key in required):
                return {
                    "feature_dim": int(contract["latent_dim"]),
                    "frame_stack": int(contract["frame_stack"]),
                    "frame_stride": int(contract.get("frame_stride", 1)),
                    "action_dim": int(contract["action_dim"]) * int(contract["action_chunk_size"]),
                    "action_chunk_size": int(contract["action_chunk_size"]),
                    "hidden_dim": int(contract["hidden_dim"]),
                }
    log_std = state_dict.get("log_std")
    if log_std is None:
        raise ValueError("PPO checkpoint is missing log_std; unsupported checkpoint format.")
    action_dim = int(log_std.shape[-1])
    if action_dim % 2 != 0:
        raise ValueError(f"Expected even flat action_dim for PushT chunk actions, got {action_dim}")
    action_chunk_size = int(action_dim // 2)

    first_weight = state_dict.get("bc_policy.net.0.weight")
    if first_weight is None:
        raise ValueError("PPO checkpoint is missing bc_policy.net.0.weight; expected bc_latent PPO checkpoint.")
    hidden_dim = int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])
    if input_dim % 512 != 0:
        raise ValueError(f"Could not infer frame_stack from bc_policy input_dim={input_dim}")
    feature_dim = 512
    frame_stack = int(input_dim // feature_dim)
    return {
        "feature_dim": feature_dim,
        "frame_stack": frame_stack,
        "frame_stride": 1,
        "action_dim": action_dim,
        "action_chunk_size": action_chunk_size,
        "hidden_dim": hidden_dim,
    }


def make_eval_env(
    config: PPOEvalConfig,
    *,
    tokenizer_path: str,
    frame_stack: int,
    frame_stride: int,
) -> gym.Env:
    _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
    env = make_pusht_env(
        env_id=config.env_id,
        render_mode="rgb_array",
        image_height=int(tokenizer_info["H"]),
        image_width=int(tokenizer_info["W"]),
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=True,
        max_episode_steps=int(config.max_episode_steps),
        fixed_target_pose=np.asarray(config.fixed_target_pose, dtype=float),
        fixed_target_block_success=bool(config.fixed_target_block_success),
        fixed_target_max_reset_attempts=int(config.fixed_target_max_reset_attempts),
        fixed_target_agent_block_coef=float(config.agent_block_coef),
        block_start_near_goal=config.block_start_radius is not None,
        block_start_radius=float(config.block_start_radius or 0.0),
        relative=bool(config.action_mode in {"relative", "swm_relative"}),
        reward_mode=str(config.reward_mode),
        render_obs=False,
    )
    env = PushTDenseRewardWrapper(env, env_id=config.env_id)
    return env


class ChunkExecutionState:
    def __init__(self, chunk_size: int, *, replan_interval: int, temporal_ensemble_decay: float):
        self.chunk_size = int(chunk_size)
        self.replan_interval = int(replan_interval)
        self.temporal_ensemble_decay = float(temporal_ensemble_decay)
        self.pending_actions: deque[np.ndarray] = deque()
        self.pending_chunks: deque[dict[str, object]] = deque()

    def reset(self) -> None:
        self.pending_actions.clear()
        self.pending_chunks.clear()

    def next_open_loop(self, chunk: np.ndarray) -> np.ndarray:
        if not self.pending_actions:
            self.pending_actions.extend(np.asarray(chunk, dtype=np.float32))
        return np.asarray(self.pending_actions.popleft(), dtype=np.float32)

    def next_receding_horizon(self, chunk: np.ndarray) -> np.ndarray:
        if not self.pending_actions:
            execute_count = min(max(1, self.replan_interval), len(chunk))
            self.pending_actions.extend(np.asarray(chunk[:execute_count], dtype=np.float32))
        return np.asarray(self.pending_actions.popleft(), dtype=np.float32)

    def next_temporal_ensemble(self, chunk: np.ndarray) -> np.ndarray:
        self.pending_chunks.append({"chunk": np.asarray(chunk, dtype=np.float32), "offset": 0})
        candidates = []
        weights = []
        for age, entry in enumerate(reversed(self.pending_chunks)):
            offset = int(entry["offset"])
            if offset >= self.chunk_size:
                continue
            candidates.append(np.asarray(entry["chunk"], dtype=np.float32)[offset])
            weights.append(math.exp(-self.temporal_ensemble_decay * age))
        if not candidates:
            action = np.zeros(2, dtype=np.float32)
        else:
            weights_np = np.asarray(weights, dtype=np.float32)
            weights_np /= max(weights_np.sum(), 1e-8)
            action = np.sum(np.asarray(candidates, dtype=np.float32) * weights_np[:, None], axis=0)
        for entry in self.pending_chunks:
            entry["offset"] = int(entry["offset"]) + 1
        while self.pending_chunks and int(self.pending_chunks[0]["offset"]) >= self.chunk_size:
            self.pending_chunks.popleft()
        return np.asarray(action, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate latent PPO on canonical PushT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--execution-mode", choices=["open-loop", "receding-horizon", "temporal-ensemble"], default="open-loop")
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument("--fixed-target-pose", type=float, nargs=3, default=PUSHT_FIXED_TARGET_POSE.tolist())
    parser.add_argument(
        "--fixed-target-block-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fixed-target-max-reset-attempts", type=int, default=100)
    parser.add_argument("--agent-block-coef", type=float, default=0.0)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--block-start-radius",
        type=float,
        default=None,
        help="sample block starts within this goal radius; omit for unrestricted starts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="runs/evaluations")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-step-pixels", type=float, default=15.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pusht-ppo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = resolve_ppo_checkpoint_path(args.checkpoint)
    device = resolve_device(args.device)
    payload = load_state_dict_safe(checkpoint, device)
    state_dict = extract_checkpoint_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unsupported PPO checkpoint format in {checkpoint}")
    checkpoint_args = extract_checkpoint_args(payload)
    architecture = infer_ppo_latent_architecture(state_dict, payload if isinstance(payload, dict) else None)
    payload_config = payload.get("config", {}) if isinstance(payload, dict) else {}
    tokenizer_default = args.tokenizer_path or payload_config.get("tokenizer_path") or checkpoint_args.get("tokenizer_ckpt_name") or TrainConfig().tokenizer_path
    tokenizer_path = resolve_tokenizer_path(tokenizer_default)
    action_mode = str(payload_config.get("action_mode", "absolute"))
    frame_stride = int(payload_config.get("frame_stride", architecture.get("frame_stride", TrainConfig().frame_stride)))
    reward_mode = str(payload_config.get("reward_mode", TrainConfig().reward_mode))

    network = TokenizerLatentBCPPOActorCritic(
        feature_dim=architecture["feature_dim"],
        frame_stack=architecture["frame_stack"],
        action_dim=architecture["action_dim"],
        action_chunk_size=architecture["action_chunk_size"],
        hidden_dim=architecture["hidden_dim"],
        init_log_std=TrainConfig().init_log_std,
    ).to(device)
    network.load_state_dict(state_dict, strict=True)
    network.eval()

    execution_mode = "temporal-ensemble" if args.temporal_ensemble else args.execution_mode
    if execution_mode == "temporal-ensemble" and args.stochastic:
        raise ValueError("temporal ensembling requires deterministic chunk predictions")

    cfg = PPOEvalConfig(
        checkpoint=checkpoint,
        device=str(device),
        stochastic=bool(args.stochastic),
        execution_mode=execution_mode,
        replan_interval=int(args.replan_interval),
        temporal_ensemble_decay=float(args.temporal_ensemble_decay),
        episodes=int(args.episodes),
        seed=int(args.seed),
        max_episode_steps=int(args.max_episode_steps),
        fixed_target_pose=tuple(float(x) for x in args.fixed_target_pose),
        fixed_target_block_success=bool(args.fixed_target_block_success),
        fixed_target_max_reset_attempts=int(args.fixed_target_max_reset_attempts),
        agent_block_coef=float(args.agent_block_coef),
        block_start_radius=None if args.block_start_radius is None else float(args.block_start_radius),
        video=bool(args.video),
        video_fps=int(args.video_fps),
        video_resolution=int(args.video_resolution),
        output_root=str(args.output_root),
        run_name=args.run_name,
        tokenizer_path=str(tokenizer_path),
        max_step_pixels=float(args.max_step_pixels),
        action_mode=action_mode,
        frame_stride=frame_stride,
        reward_mode=reward_mode,
        wandb=bool(args.wandb),
        wandb_project=str(args.wandb_project),
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )

    run_dir = create_run_directory(cfg.output_root, cfg.checkpoint, run_name=cfg.run_name)
    print(f"Evaluation run directory: {run_dir}")
    env = make_eval_env(
        cfg,
        tokenizer_path=str(tokenizer_path),
        frame_stack=architecture["frame_stack"],
        frame_stride=frame_stride,
    )
    executor = ChunkExecutionState(
        architecture["action_chunk_size"],
        replan_interval=cfg.replan_interval,
        temporal_ensemble_decay=cfg.temporal_ensemble_decay,
    )
    wandb_run = _init_wandb(
        args,
        {
            **asdict(cfg),
            "tokenizer_path": str(tokenizer_path),
            "architecture": architecture,
            "record_video": bool(cfg.video),
            "run_dir": str(run_dir),
        },
    )

    episodes = []
    returns = []
    lengths = []
    coverages = []
    successes = []
    print(
        f"Agent: ppo | fixed-target episodes={cfg.episodes} | seeds={cfg.seed}..{cfg.seed + cfg.episodes - 1} "
        f"| execution_mode={cfg.execution_mode} | block_start_radius={cfg.block_start_radius}"
    )
    with torch.no_grad():
        encoder = load_tokenizer_encoder(tokenizer_path)
        latent_backbone = TokenizerBackbone(
            encoder,
            patch=int(encoder.patch),
            output_dim=int(encoder.n_latents) * int(encoder.bottleneck_proj.out_features),
        ).to(device)
        latent_backbone.eval()
        for episode_index in range(cfg.episodes):
            _, _ = env.reset(seed=cfg.seed + episode_index)
            executor.reset()
            frames = [] if cfg.video else None
            done = False
            episode_return = 0.0
            step_count = 0
            final_info: dict = {}
            terminated = False
            truncated = False
            max_history_len = (architecture["frame_stack"] - 1) * frame_stride + 1
            frame_history: deque[np.ndarray] = deque(maxlen=max_history_len)
            while not done:
                frame = np.asarray(env.render(), dtype=np.uint8)
                frame_history.append(frame)
                if frames is not None:
                    frames.append(frame.copy())
                history = list(frame_history)
                newest = len(history) - 1
                indices = [
                    max(0, newest - i * frame_stride)
                    for i in range(architecture["frame_stack"] - 1, -1, -1)
                ]
                stacked_frames = np.stack([history[idx] for idx in indices], axis=0)
                input_tensor = (
                    torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=device)
                    .permute(0, 1, 4, 2, 3)
                    .to(torch.float32)
                    / 255.0
                )
                obs = (
                    latent_backbone.extract_features(input_tensor)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if cfg.stochastic:
                    action_flat, _, _, _ = network.get_action_and_value(state_tensor)
                    action_flat = action_flat.squeeze(0).cpu().numpy()
                else:
                    action_flat = network.actor_mean(state_tensor).squeeze(0).cpu().numpy()
                chunk = np.asarray(action_flat, dtype=np.float32).reshape(architecture["action_chunk_size"], 2)
                if cfg.execution_mode == "temporal-ensemble":
                    primitive = executor.next_temporal_ensemble(chunk)
                elif cfg.execution_mode == "receding-horizon":
                    primitive = executor.next_receding_horizon(chunk)
                else:
                    primitive = executor.next_open_loop(chunk)

                env_action = np.asarray(primitive, dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(env_action)
                final_info = dict(info)
                episode_return += float(reward)
                step_count += 1
                done = bool(terminated or truncated)

            success = _success_from_info(final_info, bool(terminated))
            coverage = float(final_info.get("coverage", 0.0))
            returns.append(float(episode_return))
            lengths.append(int(step_count))
            coverages.append(coverage)
            successes.append(float(success))
            if frames is not None:
                video_path = write_episode_video(
                    frames,
                    str(run_dir / "videos"),
                    episode_index,
                    bool(success),
                    fps=cfg.video_fps,
                    resolution=cfg.video_resolution,
                )
            else:
                video_path = None
            episode_payload = {
                "episode": int(episode_index),
                "seed": int(cfg.seed + episode_index),
                "episode_return": float(episode_return),
                "length": int(step_count),
                "success": float(success),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "video_path": str(video_path) if video_path is not None else None,
            }
            episodes.append(episode_payload)
            print(
                f"  episode {episode_index:03d}: success={float(success):.0f} "
                f"return={float(episode_return):8.1f} len={int(step_count):5d}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/episode_return": float(episode_return),
                        "eval/episode_length": float(step_count),
                        "eval/episode_success": float(success),
                    },
                    step=episode_index + 1,
                )
    env.close()

    summary = {
        "episodes": int(len(episodes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "terminated_rate": float(np.mean([episode["terminated"] for episode in episodes])),
        "truncated_rate": float(np.mean([episode["truncated"] for episode in episodes])),
        "mean_coverage": float(np.mean(coverages)) if coverages else float("nan"),
    }
    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    result = {
        "config": {
            "checkpoint": cfg.checkpoint,
            "device": cfg.device,
            "stochastic": cfg.stochastic,
            "execution_mode": cfg.execution_mode,
            "replan_interval": cfg.replan_interval,
            "temporal_ensemble_decay": cfg.temporal_ensemble_decay,
            "env_id": cfg.env_id,
            "episodes": cfg.episodes,
            "seed": cfg.seed,
            "max_episode_steps": cfg.max_episode_steps,
            "fixed_target_pose": list(cfg.fixed_target_pose),
            "fixed_target_block_success": cfg.fixed_target_block_success,
            "fixed_target_max_reset_attempts": cfg.fixed_target_max_reset_attempts,
            "agent_block_coef": cfg.agent_block_coef,
            "block_start_radius": cfg.block_start_radius,
            "record_video": cfg.video,
            "video_dir": str(run_dir / "videos") if cfg.video else None,
            "video_fps": cfg.video_fps,
            "video_resolution": cfg.video_resolution,
            "max_step_pixels": cfg.max_step_pixels,
            "action_mode": cfg.action_mode,
            "frame_stride": cfg.frame_stride,
            "tokenizer_path": str(tokenizer_path),
            "architecture": architecture,
        },
        "episodes": episodes,
        "summary": summary,
    }
    _write_metrics(result, run_dir)
    if wandb_run is not None:
        for key, value in summary.items():
            wandb_run.summary[f"eval/{key}"] = value
        wandb_run.finish()
    return result


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
