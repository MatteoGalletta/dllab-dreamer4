#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from behavioural_cloning.eval_bc_exact import save_video
from behavioural_cloning.train_tokenizer_latent_bc import TokenizerLatentBCPolicy
from behavioural_cloning.train_base import TokenizerBackbone, load_tokenizer_encoder
from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, make_pusht_env
from ppo_online.model_paths import resolve_tokenizer_path
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt
from ppo_online.train import PushTDenseRewardWrapper, resolve_device


def pad_history(frames: deque[np.ndarray], seq_len: int, frame_stride: int) -> np.ndarray:
    if not frames:
        raise ValueError("frame history is empty")
    newest = len(frames) - 1
    indices = [max(0, newest - i * int(frame_stride)) for i in range(seq_len - 1, -1, -1)]
    return np.stack([frames[idx] for idx in indices], axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate minimal tokenizer-latent BC checkpoint in exact PushT env.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-path", type=str, default="videos/tokenizer_latent_bc_exact.mp4")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--fixed-target-eval", action="store_true")
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
        relative=True,
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

    returns = []
    lengths = []
    coverages = []
    video_frames = []

    print(
        f"Loaded tokenizer-latent BC from {args.checkpoint} | tokenizer={tokenizer_path} "
        f"| seq_len={seq_len} frame_stride={frame_stride} chunk={chunk_size} "
        f"| normalize_actions={normalize_actions} action_scale={action_scale} | device={device}"
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
