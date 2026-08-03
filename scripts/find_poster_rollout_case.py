from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from behavioural_cloning.train_base import TokenizerBackbone, load_tokenizer_encoder
from behavioural_cloning.train_tokenizer_latent_bc import TokenizerLatentBCPolicy
from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, PUSHT_FIXED_TARGET_POSE, make_pusht_env
from ppo_online.eval import ChunkExecutionState, _success_from_info, infer_ppo_latent_architecture
from ppo_online.render import extract_checkpoint_args, extract_checkpoint_state_dict, load_state_dict_safe
from ppo_online.model_paths import resolve_tokenizer_path
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt
from ppo_online.train import PushTDenseRewardWrapper, TrainConfig, resolve_device
from ppo_online.networks import TokenizerLatentBCPPOActorCritic


@dataclass
class AgentRun:
    seed: int
    success: float
    episode_return: float
    length: int
    coverage: float
    frames: list[np.ndarray]


class _LatentBCAgent:
    def __init__(self, checkpoint: str, device: torch.device):
        ckpt = torch.load(checkpoint, map_location="cpu")
        ckpt_args = dict(ckpt["args"])
        tokenizer_path = resolve_tokenizer_path(ckpt_args["tokenizer_ckpt_name"])
        _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
        encoder = load_tokenizer_encoder(tokenizer_path)
        self.backbone = TokenizerBackbone(
            encoder,
            patch=int(encoder.patch),
            output_dim=int(encoder.n_latents) * int(encoder.bottleneck_proj.out_features),
        ).to(device)
        self.backbone.eval()
        self.model = TokenizerLatentBCPolicy(
            latent_dim=int(self.backbone.raw_feature_dim),
            frame_stack=int(ckpt_args["seq_len"]),
            action_dim=2,
            hidden_dim=int(ckpt_args["hidden_dim"]),
            action_chunk_size=int(ckpt_args["action_chunk_size"]),
        ).to(device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self.device = device
        self.tokenizer_path = str(tokenizer_path)
        self.image_height = int(tokenizer_info["H"])
        self.image_width = int(tokenizer_info["W"])
        self.frame_stack = int(ckpt_args["seq_len"])
        self.frame_stride = int(ckpt_args["frame_stride"])
        self.chunk_size = int(ckpt_args["action_chunk_size"])
        self.action_mode = str(ckpt_args.get("action_mode", "relative"))
        self.normalize_actions = bool(ckpt_args.get("normalize_actions", False))
        self.action_scale = float(ckpt_args.get("action_scale", 1.0))
        self.swm_action_scale = float(ckpt_args.get("swm_action_scale", 100.0))
        self.pending_actions: deque[np.ndarray] = deque()

    def reset(self) -> None:
        self.pending_actions.clear()

    def act(self, frame_history: deque[np.ndarray], step_index: int) -> np.ndarray:
        del step_index
        if not self.pending_actions:
            with torch.no_grad():
                stacked_frames = _pad_history(frame_history, self.frame_stack, self.frame_stride)
                input_tensor = (
                    torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=self.device)
                    .permute(0, 1, 4, 2, 3)
                    .to(torch.float32)
                    / 255.0
                )
                latent_stack = self.backbone.extract_features(input_tensor)
                pred = (
                    self.model(latent_stack)
                    .view(1, self.chunk_size, 2)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
            if self.action_mode == "swm_relative" and self.swm_action_scale != 100.0:
                pred = pred * (self.swm_action_scale / 100.0)
            if self.normalize_actions:
                pred = pred * self.action_scale
            self.pending_actions.extend(np.asarray(action, dtype=np.float32) for action in pred)
        return np.asarray(self.pending_actions.popleft(), dtype=np.float32)


class _LatentPPOAgent:
    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        *,
        stochastic: bool,
        execution_mode: str,
        replan_interval: int,
        temporal_ensemble_decay: float,
        tokenizer_path: str | None,
    ):
        payload = load_state_dict_safe(checkpoint, device)
        state_dict = extract_checkpoint_state_dict(payload)
        if state_dict is None:
            raise ValueError(f"Unsupported PPO checkpoint format in {checkpoint}")
        checkpoint_args = extract_checkpoint_args(payload)
        architecture = infer_ppo_latent_architecture(state_dict, payload if isinstance(payload, dict) else None)
        payload_config = payload.get("config", {}) if isinstance(payload, dict) else {}
        tokenizer_default = (
            tokenizer_path
            or payload_config.get("tokenizer_path")
            or checkpoint_args.get("tokenizer_ckpt_name")
            or TrainConfig().tokenizer_path
        )
        tokenizer_path = resolve_tokenizer_path(tokenizer_default)
        _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
        encoder = load_tokenizer_encoder(tokenizer_path)
        self.backbone = TokenizerBackbone(
            encoder,
            patch=int(encoder.patch),
            output_dim=int(encoder.n_latents) * int(encoder.bottleneck_proj.out_features),
        ).to(device)
        self.backbone.eval()
        self.network = TokenizerLatentBCPPOActorCritic(
            feature_dim=architecture["feature_dim"],
            frame_stack=architecture["frame_stack"],
            action_dim=architecture["action_dim"],
            action_chunk_size=architecture["action_chunk_size"],
            hidden_dim=architecture["hidden_dim"],
            init_log_std=TrainConfig().init_log_std,
        ).to(device)
        self.network.load_state_dict(state_dict, strict=True)
        self.network.eval()
        self.executor = ChunkExecutionState(
            architecture["action_chunk_size"],
            replan_interval=replan_interval,
            temporal_ensemble_decay=temporal_ensemble_decay,
        )
        self.device = device
        self.tokenizer_path = str(tokenizer_path)
        self.image_height = int(tokenizer_info["H"])
        self.image_width = int(tokenizer_info["W"])
        self.frame_stack = int(architecture["frame_stack"])
        self.frame_stride = int(payload_config.get("frame_stride", architecture.get("frame_stride", 1)))
        self.chunk_size = int(architecture["action_chunk_size"])
        self.action_mode = str(payload_config.get("action_mode", "absolute"))
        self.stochastic = bool(stochastic)
        self.execution_mode = str(execution_mode)

    def reset(self) -> None:
        self.executor.reset()

    def act(self, frame_history: deque[np.ndarray], step_index: int) -> np.ndarray:
        del step_index
        with torch.no_grad():
            history = list(frame_history)
            newest = len(history) - 1
            indices = [
                max(0, newest - i * self.frame_stride)
                for i in range(self.frame_stack - 1, -1, -1)
            ]
            stacked_frames = np.stack([history[idx] for idx in indices], axis=0)
            input_tensor = (
                torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=self.device)
                .permute(0, 1, 4, 2, 3)
                .to(torch.float32)
                / 255.0
            )
            obs = (
                self.backbone.extract_features(input_tensor)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.stochastic:
                action_flat, _, _, _ = self.network.get_action_and_value(state_tensor)
                action_flat = action_flat.squeeze(0).detach().cpu().numpy()
            else:
                action_flat = self.network.actor_mean(state_tensor).squeeze(0).detach().cpu().numpy()
        chunk = np.asarray(action_flat, dtype=np.float32).reshape(self.chunk_size, 2)
        if self.execution_mode == "temporal-ensemble":
            return self.executor.next_temporal_ensemble(chunk)
        if self.execution_mode == "receding-horizon":
            return self.executor.next_receding_horizon(chunk)
        return self.executor.next_open_loop(chunk)


def _pad_history(frame_history: deque[np.ndarray], seq_len: int, frame_stride: int) -> np.ndarray:
    history = list(frame_history)
    newest = len(history) - 1
    indices = [max(0, newest - i * frame_stride) for i in range(seq_len - 1, -1, -1)]
    return np.stack([history[idx] for idx in indices], axis=0)


def _make_env(*, image_height: int, image_width: int, action_mode: str, max_steps: int, block_start_radius: float | None, seed: int):
    env = make_pusht_env(
        env_id=DEFAULT_PUSHT_ENV_ID,
        render_mode="rgb_array",
        image_height=int(image_height),
        image_width=int(image_width),
        relative=(str(action_mode) in {"relative", "swm_relative"}),
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=True,
        fixed_target_pose=tuple(PUSHT_FIXED_TARGET_POSE.tolist()),
        fixed_target_block_success=True,
        fixed_target_max_reset_attempts=100,
        fixed_target_agent_block_coef=0.0,
        block_start_near_goal=block_start_radius is not None,
        block_start_radius=float(block_start_radius or 0.0),
        render_obs=False,
        max_episode_steps=int(max_steps),
    )
    env = PushTDenseRewardWrapper(env, env_id=DEFAULT_PUSHT_ENV_ID)
    env.reset(seed=seed)
    return env


def _rollout(agent: Any, *, seed: int, max_steps: int, block_start_radius: float | None) -> AgentRun:
    env = _make_env(
        image_height=agent.image_height,
        image_width=agent.image_width,
        action_mode=agent.action_mode,
        max_steps=max_steps,
        block_start_radius=block_start_radius,
        seed=seed,
    )
    agent.reset()
    max_history_len = (agent.frame_stack - 1) * agent.frame_stride + 1
    frame_history: deque[np.ndarray] = deque(maxlen=max_history_len)
    frames: list[np.ndarray] = []
    done = False
    total_reward = 0.0
    step_count = 0
    final_info: dict[str, Any] = {}
    terminated = False
    try:
        while not done and step_count < int(max_steps):
            frame = np.asarray(env.render(), dtype=np.uint8)
            frame_history.append(frame)
            frames.append(frame.copy())
            env_action = np.asarray(agent.act(frame_history, step_count), dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(env_action)
            total_reward += float(reward)
            step_count += 1
            final_info = dict(info)
            done = bool(terminated or truncated)
        if final_info:
            frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        return AgentRun(
            seed=int(seed),
            success=float(_success_from_info(final_info, bool(terminated))),
            episode_return=float(total_reward),
            length=int(step_count),
            coverage=float(final_info.get("coverage", 0.0)),
            frames=frames,
        )
    finally:
        env.close()


def _make_side_by_side(left_frames: list[np.ndarray], right_frames: list[np.ndarray], *, pad_value: int = 255) -> list[np.ndarray]:
    total = max(len(left_frames), len(right_frames))
    if total == 0:
        return []
    left_last = left_frames[-1]
    right_last = right_frames[-1]
    left_h, left_w = left_last.shape[:2]
    right_h, right_w = right_last.shape[:2]
    height = max(left_h, right_h)
    gap = 12
    canvas_w = left_w + gap + right_w
    out: list[np.ndarray] = []
    for idx in range(total):
        left = left_frames[idx] if idx < len(left_frames) else left_last
        right = right_frames[idx] if idx < len(right_frames) else right_last
        canvas = np.full((height, canvas_w, 3), pad_value, dtype=np.uint8)
        canvas[: left.shape[0], : left.shape[1]] = left
        canvas[: right.shape[0], left_w + gap : left_w + gap + right.shape[1]] = right
        canvas[:8, :left_w] = np.array([54, 116, 181], dtype=np.uint8)
        canvas[:8, left_w + gap : left_w + gap + right_w] = np.array([214, 95, 0], dtype=np.uint8)
        out.append(canvas)
    return out


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(path, frames, fps=int(fps))
        return
    imageio.mimsave(
        path,
        frames,
        fps=int(fps),
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a shared PushT seed where BC fails and PPO succeeds, then save comparison rollouts."
    )
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--ppo-checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/poster_rollout_compare")
    parser.add_argument("--tokenizer-path", default=None, help="Optional PPO tokenizer override")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--block-start-radius", type=float, default=200.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic-ppo", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=["open-loop", "receding-horizon", "temporal-ensemble"],
        default="open-loop",
    )
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    bc_agent = _LatentBCAgent(args.bc_checkpoint, device)
    ppo_agent = _LatentPPOAgent(
        args.ppo_checkpoint,
        device,
        stochastic=bool(args.stochastic_ppo),
        execution_mode=str(args.execution_mode),
        replan_interval=int(args.replan_interval),
        temporal_ensemble_decay=float(args.temporal_ensemble_decay),
        tokenizer_path=args.tokenizer_path,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_case: dict[str, Any] | None = None
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)):
        bc_run = _rollout(
            bc_agent,
            seed=seed,
            max_steps=int(args.max_steps),
            block_start_radius=float(args.block_start_radius) if args.block_start_radius is not None else None,
        )
        ppo_run = _rollout(
            ppo_agent,
            seed=seed,
            max_steps=int(args.max_steps),
            block_start_radius=float(args.block_start_radius) if args.block_start_radius is not None else None,
        )
        print(
            f"seed={seed} | bc success={bc_run.success:.0f} return={bc_run.episode_return:7.1f} len={bc_run.length:3d} "
            f"| ppo success={ppo_run.success:.0f} return={ppo_run.episode_return:7.1f} len={ppo_run.length:3d}"
        )
        if bc_run.success < 0.5 and ppo_run.success >= 0.5:
            margin = ppo_run.episode_return - bc_run.episode_return
            if best_case is None or margin > best_case["margin"]:
                best_case = {"seed": seed, "bc": bc_run, "ppo": ppo_run, "margin": margin}

    if best_case is None:
        raise RuntimeError(
            "Could not find a shared seed where BC failed and PPO succeeded. "
            "Try increasing --num-seeds or changing --block-start-radius."
        )

    side_by_side = _make_side_by_side(best_case["bc"].frames, best_case["ppo"].frames)
    side_path = out_dir / f"seed_{best_case['seed']:04d}_bc_fail_ppo_success_side_by_side.mp4"
    bc_path = out_dir / f"seed_{best_case['seed']:04d}_bc.mp4"
    ppo_path = out_dir / f"seed_{best_case['seed']:04d}_ppo.mp4"
    _write_video(side_by_side, side_path, int(args.fps))
    _write_video(best_case["bc"].frames, bc_path, int(args.fps))
    _write_video(best_case["ppo"].frames, ppo_path, int(args.fps))

    summary = {
        "seed": int(best_case["seed"]),
        "block_start_radius": None if args.block_start_radius is None else float(args.block_start_radius),
        "max_steps": int(args.max_steps),
        "bc_checkpoint": str(args.bc_checkpoint),
        "ppo_checkpoint": str(args.ppo_checkpoint),
        "bc": asdict(best_case["bc"]) | {"frames": None},
        "ppo": asdict(best_case["ppo"]) | {"frames": None},
        "side_by_side_video": str(side_path),
        "bc_video": str(bc_path),
        "ppo_video": str(ppo_path),
    }
    summary_path = out_dir / f"seed_{best_case['seed']:04d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved poster comparison seed={best_case['seed']} to {out_dir}")
    print(f"  side_by_side={side_path}")
    print(f"  bc_video={bc_path}")
    print(f"  ppo_video={ppo_path}")
    print(f"  summary={summary_path}")


if __name__ == "__main__":
    main()
