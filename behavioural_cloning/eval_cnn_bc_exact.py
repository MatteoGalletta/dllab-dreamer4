#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import importlib
import os
import math
import re
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavioural_cloning.train_cnn_bc import CNNBCPolicy, CNNBackbone, DirectChunkPolicyHead
from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, PUSHT_FIXED_TARGET_POSE, make_pusht_env

if importlib.util.find_spec("wandb") is not None:
    wandb = importlib.import_module("wandb")
else:
    wandb = None


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            major, minor = torch.cuda.get_device_capability(0)
            supported_arches = {
                arch.replace("sm_", "")
                for arch in torch.cuda.get_arch_list()
                if arch.startswith("sm_")
            }
            requested_arch = f"{major}{minor}"
            if supported_arches and requested_arch not in supported_arches:
                print(
                    "CUDA device detected but unsupported by this PyTorch build: "
                    f"sm_{requested_arch} not in {sorted(supported_arches)}. Falling back to CPU."
                )
                return torch.device("cpu")
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as error:
            print(f"CUDA auto-detection failed ({error}). Falling back to CPU.")
            return torch.device("cpu")
    return torch.device(device_name)


def extract_state_array(observation) -> np.ndarray:
    if isinstance(observation, dict):
        if "state" in observation:
            return np.asarray(observation["state"], dtype=np.float32)
        raise KeyError("Expected observation dict to contain a 'state' entry.")
    return np.asarray(observation, dtype=np.float32)


class PushTDenseRewardWrapper:
    def __init__(self, env, env_id: str | None = None):
        self.env = env
        self.target_pos = np.array([256.0, 256.0], dtype=np.float32)
        self.env_id = env_id or getattr(getattr(env, "spec", None), "id", "") or ""

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _resolve_goal_block_pose(self, info: dict, state: np.ndarray) -> np.ndarray | None:
        goal_pose = info.get("goal_pose")
        if goal_pose is not None:
            goal_pose = np.asarray(goal_pose, dtype=np.float32).reshape(-1)
            if goal_pose.shape[0] >= 3:
                return goal_pose[:3]

        goal_state = info.get("goal_state")
        if goal_state is not None:
            goal_state = np.asarray(goal_state, dtype=np.float32).reshape(-1)
            if goal_state.shape[0] >= 5:
                return np.array([goal_state[2], goal_state[3], goal_state[4]], dtype=np.float32)

        if state.shape[0] >= 5:
            return np.array([self.target_pos[0], self.target_pos[1], state[4]], dtype=np.float32)
        return None

    def _angle_distance(self, angle_a: float, angle_b: float) -> float:
        diff = abs(angle_a - angle_b) % (2.0 * math.pi)
        return min(diff, (2.0 * math.pi) - diff)

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        state = extract_state_array(observation)
        eef_pos = state[0:2].astype(np.float32)
        block_pos = state[2:4].astype(np.float32)
        block_angle = float(state[4]) if state.shape[0] >= 5 else 0.0
        goal_block_pose = self._resolve_goal_block_pose(info, state)
        goal_block_pos = goal_block_pose[:2] if goal_block_pose is not None else self.target_pos
        goal_block_angle = float(goal_block_pose[2]) if goal_block_pose is not None else block_angle

        dist_reach = float(np.linalg.norm(eef_pos - block_pos))
        dist_push = float(np.linalg.norm(block_pos - goal_block_pos))
        angle_error = self._angle_distance(block_angle, goal_block_angle)

        r_push = math.exp(-dist_push / 100.0)
        dist_reach_eff = max(0.0, dist_reach - 60.0)
        r_reach = math.exp(-dist_reach_eff / 100.0)
        r_angle = math.exp(-angle_error / (math.pi / 6.0))

        info = dict(info)
        info["original_reward"] = float(original_reward)
        info["reach_reward"] = float(r_reach)
        info["push_reward"] = float(r_push)
        info["angle_reward"] = float(r_angle)
        info["distance_to_block"] = dist_reach
        info["distance_to_target"] = dist_push
        info["angle_error"] = angle_error
        info["coverage_proxy"] = float(r_push)
        info["coverage"] = float(info.get("coverage", 0.0))

        dense_reward = (1.0 * r_reach) + (3.0 * r_push) + (1.0 * r_angle)
        info["dense_reward"] = float(dense_reward)
        return observation, float(dense_reward), terminated, truncated, info


def pad_history(frames: deque[np.ndarray], seq_len: int, frame_stride: int) -> np.ndarray:
    if not frames:
        raise ValueError("frame history is empty")
    newest = len(frames) - 1
    indices = [max(0, newest - i * int(frame_stride)) for i in range(seq_len - 1, -1, -1)]
    return np.stack([frames[idx] for idx in indices], axis=0)


def save_video(frames: list[np.ndarray], video_path: str, fps: int = 15):
    if not video_path or not frames:
        return
    output = Path(video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "Video saving for CNN BC eval requires ffmpeg in PATH. "
            "Run with --video-path \"\" to disable video output."
        )

    temp_dir = output.parent / f".{output.stem}_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        from imageio.v2 import imwrite
    except ImportError as exc:
        raise RuntimeError(
            "Video saving for CNN BC eval requires imageio when OpenCV is unavailable."
        ) from exc

    for index, frame in enumerate(frames):
        frame_path = temp_dir / f"frame_{index:06d}.png"
        imwrite(frame_path, np.asarray(frame, dtype=np.uint8))

    cmd = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(int(fps)),
        "-i",
        str(temp_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for frame_path in temp_dir.glob("*.png"):
        frame_path.unlink(missing_ok=True)
    temp_dir.rmdir()
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not encode video with ffmpeg for {output}: {result.stderr.strip() or result.stdout.strip()}"
        )
    print(f"Saved evaluation video to {output} using codec=libx264")


def _upscale(frame: np.ndarray, resolution: int) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    if resolution <= 0 or frame.shape[:2] == (resolution, resolution):
        return frame
    try:
        import cv2  # type: ignore
    except ImportError:
        return frame
    return cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_NEAREST)


def write_episode_video(
    frames: list[np.ndarray],
    video_dir: str,
    episode_index: int,
    success: bool,
    *,
    fps: int = 10,
    resolution: int = 512,
) -> str | None:
    if not frames:
        return None
    if fps <= 0:
        raise ValueError("video fps must be positive")
    output_dir = Path(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "success" if success else "fail"
    output_path = output_dir / f"episode_{episode_index:03d}_{tag}.mp4"
    save_video([_upscale(frame, resolution) for frame in frames], str(output_path), fps=fps)
    print(f"  saved {output_path}")
    return str(output_path)


def _success_from_info(info: dict, terminated: bool) -> bool:
    for key in ("success", "is_success", "task_success", "block_success"):
        if key in info:
            return float(np.asarray(info[key]).squeeze()) > 0.5
    return bool(terminated)


def _get_wandb_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "wandb_mode", "disabled") or "disabled"
    if mode == "online":
        has_api_key = bool(Path.home().joinpath(".netrc").exists()) or bool(os.environ.get("WANDB_API_KEY"))
        if not has_api_key:
            print("WANDB_API_KEY/.netrc not found, falling back from wandb online mode to offline mode.")
            return "offline"
    return mode


def _init_wandb(args: argparse.Namespace, config: dict):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb evaluation logging was requested, but wandb is not installed.")
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=_get_wandb_mode(args),
        config=config,
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return slug or "eval"


def create_run_directory(output_root: str, checkpoint: str, run_name: str | None = None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, "cnn_bc", _slug(Path(checkpoint).stem)]
    if run_name:
        parts.append(_slug(run_name))
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    base_name = "_".join(parts)
    for suffix in range(1000):
        name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        run_dir = output_root_path / name
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise RuntimeError(f"could not allocate an evaluation run directory under {output_root}")


def write_metrics_json(summary: dict[str, float], run_dir: Path, episodes: list[dict], config: dict) -> Path:
    metrics_path = run_dir / "metrics.json"
    payload = {
        "config": config,
        "episodes": episodes,
        "summary": summary,
        "artifacts": {
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "video_dir": str(run_dir / "videos") if config.get("record_video", False) else None,
        },
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Saved evaluation metrics to: {metrics_path}")
    return metrics_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate minimal CNN BC checkpoint on canonical fixed-target PushT."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", "--max_steps", "--max-episode-steps", dest="max_steps", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video-path", "--video_path", dest="video_path", type=str, default=None)
    parser.add_argument("--video", action="store_true", help="Save per-episode videos under the evaluation run directory.")
    parser.add_argument("--video-fps", "--video_fps", dest="video_fps", type=int, default=30)
    parser.add_argument("--video-resolution", "--video_resolution", dest="video_resolution", type=int, default=512)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--fixed-target-pose",
        "--fixed_target_pose",
        dest="fixed_target_pose",
        type=float,
        nargs=3,
        default=PUSHT_FIXED_TARGET_POSE.tolist(),
        metavar=("X", "Y", "ANGLE"),
    )
    parser.add_argument("--fixed-target-full-state-success", "--fixed_target_full_state_success", dest="fixed_target_full_state_success", action="store_true")
    parser.add_argument(
        "--fixed-target-max-reset-attempts",
        "--fixed_target_max_reset_attempts",
        dest="fixed_target_max_reset_attempts",
        type=int,
        default=100,
    )
    parser.add_argument("--agent-block-coef", "--agent_block_coef", dest="agent_block_coef", type=float, default=0.0)
    parser.add_argument(
        "--block-start-radius",
        "--block_start_radius",
        dest="block_start_radius",
        type=float,
        default=None,
        help="Sample block starts within this goal radius; omit for unrestricted starts.",
    )
    parser.add_argument(
        "--fixed-target-eval",
        action="store_true",
        help="Compatibility flag; canonical CNN BC eval now always uses fixed-target PushT.",
    )
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=["open-loop", "temporal-ensemble"],
        default="open-loop",
        help="Compatibility alias; maps to the CNN BC execution style.",
    )
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument("--seed", "--eval_seed", dest="eval_seed", type=int, default=42)
    parser.add_argument("--output-root", "--output_root", dest="output_root", default="runs/evaluations")
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="pusht-cnn-bc")
    parser.add_argument("--wandb-entity", "--wandb_entity", dest="wandb_entity", default=None)
    parser.add_argument("--wandb-run-name", "--wandb_run_name", dest="wandb_run_name", default=None)
    parser.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="pusht-cnn-bc-eval")
    parser.add_argument("--wandb-tags", "--wandb_tags", dest="wandb_tags", nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        "--wandb_mode",
        dest="wandb_mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
    )
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = resolve_device("auto")
    if args.video and not args.video_path:
        args.video_path = "videos"
    if args.execution_mode == "temporal-ensemble":
        args.temporal_ensemble = True
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    seq_len = int(ckpt_args.get("seq_len", 3))
    frame_stride = int(ckpt_args.get("frame_stride", 5))
    chunk_size = int(ckpt_args.get("action_chunk_size", 1))
    cnn_feature_dim = int(ckpt_args.get("cnn_feature_dim", 256))
    hidden_dim = int(ckpt_args.get("hidden_dim", 256))
    dropout = float(ckpt_args.get("dropout", 0.0))
    action_output_tanh = bool(ckpt_args.get("action_output_tanh", False))
    normalize_actions = bool(ckpt_args.get("normalize_actions", False))
    action_scale = float(ckpt_args.get("action_scale", 1.0))
    action_mode = str(ckpt_args.get("action_mode", "relative"))
    swm_action_scale = float(ckpt_args.get("swm_action_scale", 100.0))

    image_hw = ckpt_args.get("image_hw")
    img_h, img_w = (int(image_hw[0]), int(image_hw[1])) if image_hw else (96, 96)

    action_dim = chunk_size * 2
    backbone = CNNBackbone(in_channels=3, feature_dim=cnn_feature_dim)
    classifier = DirectChunkPolicyHead(
        in_dim=cnn_feature_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        dropout=dropout,
        output_tanh=action_output_tanh,
    )
    model = CNNBCPolicy(backbone=backbone, classifier=classifier).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    env = make_pusht_env(
        env_id=DEFAULT_PUSHT_ENV_ID,
        render_mode="rgb_array",
        image_height=img_h,
        image_width=img_w,
        relative=(action_mode in {"relative", "swm_relative"}),
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=True,
        fixed_target_pose=tuple(args.fixed_target_pose),
        fixed_target_block_success=not bool(args.fixed_target_full_state_success),
        fixed_target_max_reset_attempts=int(args.fixed_target_max_reset_attempts),
        fixed_target_agent_block_coef=float(args.agent_block_coef),
        block_start_near_goal=args.block_start_radius is not None,
        block_start_radius=float(args.block_start_radius or 0.0),
        render_obs=False,
        max_episode_steps=int(args.max_steps),
    )
    env = PushTDenseRewardWrapper(env, env_id=DEFAULT_PUSHT_ENV_ID)

    returns = []
    lengths = []
    coverages = []
    successes = []
    run_dir = create_run_directory(args.output_root, args.checkpoint, run_name=args.run_name)
    print(f"Evaluation run directory: {run_dir}")
    video_dir = str(run_dir / "videos") if args.video_path else None
    episode_payloads: list[dict] = []
    wandb_run = _init_wandb(
        args,
        {
            **vars(args),
            "checkpoint_args": ckpt_args,
            "device": str(device),
            "record_video": bool(video_dir),
            "run_dir": str(run_dir),
        },
    )

    print(
        f"Loaded CNN BC from {args.checkpoint} "
        f"| seq_len={seq_len} frame_stride={frame_stride} chunk={chunk_size} "
        f"| image_hw=({img_h}, {img_w}) cnn_feature_dim={cnn_feature_dim} "
        f"| action_mode={action_mode} normalize_actions={normalize_actions} "
        f"action_scale={action_scale} temporal_ensemble={args.temporal_ensemble} "
        f"| fixed_target_pose={tuple(float(x) for x in args.fixed_target_pose)} "
        f"| fixed_target_block_success={not bool(args.fixed_target_full_state_success)} "
        f"| block_start_radius={args.block_start_radius} | device={device}"
    )
    if args.render:
        print("WARNING: live --render is not supported here; use --video-path for saved video output.")

    with torch.no_grad():
        for episode_idx in range(int(args.episodes)):
            _, _ = env.reset(seed=int(args.eval_seed) + episode_idx)
            max_history_len = (seq_len - 1) * frame_stride + 1
            frame_history: deque[np.ndarray] = deque(maxlen=max_history_len)
            action_buffer: deque[np.ndarray] = deque()
            done = False
            total_reward = 0.0
            step_count = 0
            final_info: dict = {}
            episode_frames: list[np.ndarray] | None = [] if video_dir else None

            while not done and step_count < int(args.max_steps):
                frame = np.asarray(env.render(), dtype=np.uint8)
                frame_history.append(frame)
                if episode_frames is not None:
                    episode_frames.append(frame.copy())

                if not action_buffer:
                    stacked_frames = pad_history(frame_history, seq_len, frame_stride)
                    input_tensor = (
                        torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=device)
                        .permute(0, 1, 4, 2, 3)
                        .to(torch.float32)
                        / 255.0
                    )
                    pred = model(input_tensor).view(1, chunk_size, 2).squeeze(0).cpu().numpy()
                    if action_mode == "swm_relative" and swm_action_scale != 100.0:
                        pred = pred * (swm_action_scale / 100.0)
                    if normalize_actions:
                        pred = pred * action_scale
                    if args.temporal_ensemble:
                        action_buffer.clear()
                        action_buffer.extend([np.asarray(action, dtype=np.float32) for action in pred])
                    else:
                        action_buffer.extend([np.asarray(action, dtype=np.float32) for action in pred])

                env_action = np.asarray(action_buffer.popleft(), dtype=np.float32)
                _, reward, terminated, truncated, info = env.step(env_action)
                total_reward += float(reward)
                step_count += 1
                final_info = dict(info)
                done = bool(terminated or truncated)

                if step_count % int(args.print_every) == 0 or done:
                    coverage = float(final_info.get("coverage", 0.0))
                    print(
                        f"episode={episode_idx + 1:02d}/{int(args.episodes):02d} "
                        f"step={step_count:04d} reward={float(reward):8.3f} "
                        f"total={total_reward:8.3f} coverage={coverage:.3f}"
                    )

            success = _success_from_info(final_info, done)
            returns.append(total_reward)
            lengths.append(step_count)
            coverages.append(float(final_info.get("coverage", 0.0)))
            successes.append(float(success))
            print(
                f"episode={episode_idx + 1:02d} return={total_reward:.3f} "
                f"steps={step_count} success={int(success)} coverage={float(final_info.get('coverage', 0.0))}"
            )
            if video_dir and episode_frames is not None:
                video_file = write_episode_video(
                    episode_frames,
                    video_dir,
                    episode_idx,
                    bool(success),
                    fps=int(args.video_fps),
                    resolution=int(args.video_resolution),
                )
            else:
                video_file = None
            episode_payloads.append(
                {
                    "episode": int(episode_idx),
                    "seed": int(args.eval_seed) + int(episode_idx),
                    "episode_return": float(total_reward),
                    "length": int(step_count),
                    "success": float(success),
                    "terminated": bool(done and step_count < int(args.max_steps)),
                    "truncated": bool(step_count >= int(args.max_steps)),
                    "video_path": video_file,
                }
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/episode_return": float(total_reward),
                        "eval/episode_length": float(step_count),
                        "eval/episode_success": float(success),
                        "eval/episode_coverage": float(final_info.get("coverage", 0.0)),
                    },
                    step=episode_idx + 1,
                )

    env.close()
    summary = {
        "episodes": float(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "std_return": float(np.std(returns)) if returns else float("nan"),
        "min_return": float(np.min(returns)) if returns else float("nan"),
        "max_return": float(np.max(returns)) if returns else float("nan"),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "mean_coverage": float(np.mean(coverages)) if coverages else float("nan"),
        "success_rate": float(np.mean(successes)) if successes else float("nan"),
        "terminated_rate": float(np.mean([episode["terminated"] for episode in episode_payloads])) if episode_payloads else float("nan"),
        "truncated_rate": float(np.mean([episode["truncated"] for episode in episode_payloads])) if episode_payloads else float("nan"),
    }
    write_metrics_json(
        summary,
        run_dir,
        episode_payloads,
        {
            "checkpoint": args.checkpoint,
            "episodes": int(args.episodes),
            "seed": int(args.eval_seed),
            "max_episode_steps": int(args.max_steps),
            "fixed_target_pose": [float(x) for x in args.fixed_target_pose],
            "fixed_target_block_success": not bool(args.fixed_target_full_state_success),
            "fixed_target_max_reset_attempts": int(args.fixed_target_max_reset_attempts),
            "agent_block_coef": float(args.agent_block_coef),
            "block_start_radius": None if args.block_start_radius is None else float(args.block_start_radius),
            "record_video": bool(video_dir),
            "video_dir": str(run_dir / "videos") if video_dir else None,
            "video_fps": int(args.video_fps),
            "video_resolution": int(args.video_resolution),
            "execution_mode": "temporal-ensemble" if args.temporal_ensemble else "open-loop",
        },
    )
    print(
        f"Summary: episodes={int(summary['episodes'])} mean_return={summary['mean_return']:.3f} "
        f"mean_length={summary['mean_length']:.1f} mean_coverage={summary['mean_coverage']:.3f} "
        f"success_rate={summary['success_rate']:.3f}"
    )
    if wandb_run is not None:
        for key, value in summary.items():
            wandb_run.summary[f"eval/{key}"] = value
        wandb_run.finish()
    return summary


def main():
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
