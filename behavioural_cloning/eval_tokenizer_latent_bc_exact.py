#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
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

from behavioural_cloning.train_tokenizer_latent_bc import TokenizerLatentBCPolicy
from behavioural_cloning.train_base import TokenizerBackbone, load_tokenizer_encoder
from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, make_pusht_env
from ppo_online.model_paths import resolve_tokenizer_path
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt


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
            "Video saving for tokenizer latent eval requires ffmpeg in PATH. "
            "Run with --video-path \"\" to disable video output."
        )

    temp_dir = output.parent / f".{output.stem}_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        from imageio.v2 import imwrite
    except ImportError as exc:
        raise RuntimeError(
            "Video saving for tokenizer latent eval requires imageio when OpenCV is unavailable."
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate minimal tokenizer-latent BC checkpoint in exact PushT env.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-path", type=str, default="videos/tokenizer_latent_bc_exact.mp4")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--fixed-target-eval", action="store_true")
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device("auto")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    tokenizer_path = resolve_tokenizer_path(ckpt_args["tokenizer_ckpt_name"])
    _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
    encoder = load_tokenizer_encoder(tokenizer_path)
    latent_backbone = TokenizerBackbone(
        encoder,
        patch=int(encoder.patch),
        output_dim=int(encoder.n_latents) * int(encoder.bottleneck_proj.out_features),
    ).to(device)
    latent_backbone.eval()
    model = TokenizerLatentBCPolicy(
        latent_dim=int(latent_backbone.raw_feature_dim),
        frame_stack=int(ckpt_args["seq_len"]),
        action_dim=2,
        hidden_dim=int(ckpt_args["hidden_dim"]),
        action_chunk_size=int(ckpt_args["action_chunk_size"]),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    env = make_pusht_env(
        env_id=DEFAULT_PUSHT_ENV_ID,
        render_mode="rgb_array",
        image_height=int(tokenizer_info["H"]),
        image_width=int(tokenizer_info["W"]),
        relative=(str(ckpt_args.get("action_mode", "relative")) == "relative"),
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=args.fixed_target_eval,
        render_obs=False,
    )
    env = PushTDenseRewardWrapper(env, env_id=DEFAULT_PUSHT_ENV_ID)

    seq_len = int(ckpt_args["seq_len"])
    frame_stride = int(ckpt_args["frame_stride"])
    chunk_size = int(ckpt_args["action_chunk_size"])
    normalize_actions = bool(ckpt_args.get("normalize_actions", False))
    action_scale = float(ckpt_args.get("action_scale", 1.0))
    action_mode = str(ckpt_args.get("action_mode", "relative"))

    returns = []
    lengths = []
    coverages = []
    video_frames = []

    print(
        f"Loaded tokenizer-latent BC from {args.checkpoint} | tokenizer={tokenizer_path} "
        f"| seq_len={seq_len} frame_stride={frame_stride} chunk={chunk_size} "
        f"| action_mode={action_mode} normalize_actions={normalize_actions} "
        f"action_scale={action_scale} temporal_ensemble={args.temporal_ensemble} | device={device}"
    )

    with torch.no_grad():
        for episode_idx in range(int(args.episodes)):
            _, _ = env.reset(seed=int(args.seed) + episode_idx)
            max_history_len = (seq_len - 1) * frame_stride + 1
            frame_history: deque[np.ndarray] = deque(maxlen=max_history_len)
            action_buffer: deque[np.ndarray] = deque()
            done = False
            total_reward = 0.0
            step_count = 0
            final_info: dict = {}

            while not done and step_count < int(args.max_steps):
                frame = np.asarray(env.render(), dtype=np.uint8)
                frame_history.append(frame)
                if episode_idx == 0 and args.video_path:
                    video_frames.append(frame.copy())

                if not action_buffer:
                    stacked_frames = pad_history(frame_history, seq_len, frame_stride)
                    input_tensor = (
                        torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=device)
                        .permute(0, 1, 4, 2, 3)
                        .to(torch.float32)
                        / 255.0
                    )
                    latent_stack = latent_backbone.extract_features(input_tensor)
                    pred = model(latent_stack).view(1, chunk_size, 2).squeeze(0).cpu().numpy()
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

            returns.append(total_reward)
            lengths.append(step_count)
            coverages.append(float(final_info.get("coverage", 0.0)))
            print(
                f"episode={episode_idx + 1:02d} return={total_reward:.3f} "
                f"steps={step_count} coverage={float(final_info.get('coverage', 0.0))}"
            )

    env.close()
    if args.video_path:
        save_video(video_frames, args.video_path)
    print(
        f"Summary: episodes={len(returns)} mean_return={float(np.mean(returns)):.3f} "
        f"mean_length={float(np.mean(lengths)):.1f} mean_coverage={float(np.mean(coverages)):.3f}"
    )


if __name__ == "__main__":
    main()
