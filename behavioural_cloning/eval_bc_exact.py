#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, make_pusht_env
from ppo_online.model_paths import resolve_bc_prior_path, resolve_tokenizer_path
from ppo_online.networks import BCActionClassifier, TokenizerBackbone
from ppo_online.render import extract_checkpoint_args, extract_checkpoint_state_dict, load_state_dict_safe
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt
from ppo_online.train import extract_state_array, resolve_device


class CNNBackbone(nn.Module):
    def __init__(self, *, in_channels: int, feature_dim: int = 256):
        super().__init__()
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)

        def conv_block(in_ch: int, out_ch: int, *, stride: int = 2) -> nn.Sequential:
            kernel = 5 if stride == 2 else 3
            padding = 2 if kernel == 5 else 1
            groups = min(8, out_ch)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.ReLU(),
            )

        self.backbone = nn.Sequential(
            conv_block(self.in_channels, 32, stride=2),
            conv_block(32, 64, stride=2),
            conv_block(64, 128, stride=2),
            conv_block(128, 256, stride=2),
            conv_block(256, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x_btchw.shape
        x = x_btchw.reshape(batch * steps, channels, height, width)
        features = self.proj(self.backbone(x))
        return features.view(batch, steps, -1)


class BCImagePolicy(nn.Module):
    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int],
        hidden_dim: int,
        dropout: float,
        action_chunk_size: int,
        seq_len: int,
        tokenizer_ckpt: str | None,
        temporal_layers: int,
        temporal_heads: int,
        backbone_device: torch.device | str,
    ):
        super().__init__()
        action_dim = int(action_chunk_size) * 2
        if tokenizer_ckpt is not None:
            self.backbone = TokenizerBackbone(tokenizer_ckpt=tokenizer_ckpt, device=backbone_device)
        else:
            self.backbone = CNNBackbone(in_channels=int(image_shape[2]))
        self.classifier = BCActionClassifier(
            in_dim=self.backbone.feature_dim,
            hidden_dim=int(hidden_dim),
            action_dim=action_dim,
            dropout=float(dropout),
            temporal_layers=int(temporal_layers),
            temporal_heads=int(temporal_heads),
            max_seq_len=int(seq_len),
        )
        self.action_chunk_size = int(action_chunk_size)

    def _ensure_sequence(self, image_obs: torch.Tensor) -> torch.Tensor:
        if image_obs.ndim == 4:
            image_obs = image_obs.unsqueeze(0)
        if image_obs.ndim != 5:
            raise ValueError(f"Expected image observations with shape (B, T, H, W, C), got {tuple(image_obs.shape)}")
        if image_obs.shape[-1] != 3:
            raise ValueError(f"Expected RGB images in the last dimension, got shape {tuple(image_obs.shape)}")
        return image_obs.permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, image_obs: torch.Tensor) -> torch.Tensor:
        sequence = self._ensure_sequence(image_obs)
        if isinstance(self.backbone, CNNBackbone):
            if sequence.dtype == torch.uint8:
                sequence = sequence.to(torch.float32) / 255.0
            else:
                sequence = sequence.to(torch.float32)
                if sequence.numel() > 0 and float(sequence.max().detach().cpu()) > 1.5:
                    sequence = sequence / 255.0
        features = self.backbone(sequence)
        return self.classifier(features)

    def predict_action_chunk(self, image_obs: torch.Tensor) -> torch.Tensor:
        actions = self.forward(image_obs)
        return actions[:, -1, :].view(actions.shape[0], self.action_chunk_size, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC checkpoint with a BC-native open-loop action queue.")
    parser.add_argument("--checkpoint", type=str, default="local_models/behavior_cloning/bc_best.pt")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("relative", "delta", "absolute"),
        default="relative",
        help="How to map predicted primitive actions into env actions.",
    )
    parser.add_argument(
        "--max-step-pixels",
        type=float,
        default=15.0,
        help="Only used with --action-mode delta.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="videos/bc_exact_eval.mp4",
        help="Output video path. Pass empty string to disable video saving.",
    )
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "_orig_mod.", "network.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned[clean_key] = value
    return cleaned


def resolve_model_config(ckpt_args: dict, cleaned_state: dict[str, torch.Tensor]):
    tokenizer_name = ckpt_args.get("tokenizer_ckpt_name")
    seq_len = int(ckpt_args.get("seq_len", 8))
    action_chunk_size = int(ckpt_args.get("action_chunk_size", 5))
    hidden_dim = int(ckpt_args.get("hidden_dim", 512))
    dropout = float(ckpt_args.get("dropout", 0.05))
    has_temporal = any(key.startswith("classifier.temporal_") for key in cleaned_state)
    temporal_layers = int(ckpt_args.get("temporal_layers", 2 if has_temporal else 0))
    temporal_heads = int(ckpt_args.get("temporal_heads", 4))
    return {
        "tokenizer_name": tokenizer_name,
        "seq_len": seq_len,
        "action_chunk_size": action_chunk_size,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "temporal_layers": temporal_layers,
        "temporal_heads": temporal_heads,
    }


def pad_history(frames: deque[np.ndarray], seq_len: int) -> np.ndarray:
    history = list(frames)
    if not history:
        raise ValueError("frame history is empty")
    if len(history) >= seq_len:
        history = history[-seq_len:]
    else:
        history = [history[0]] * (seq_len - len(history)) + history
    return np.stack(history, axis=0)


def map_primitive_to_env_action(
    primitive: np.ndarray,
    *,
    mode: str,
    current_eef: np.ndarray,
    max_step_pixels: float,
) -> np.ndarray:
    primitive = np.asarray(primitive, dtype=np.float32)
    if mode == "relative":
        return primitive
    if mode == "delta":
        return np.clip(current_eef + primitive * float(max_step_pixels), 0.0, 512.0)
    if mode == "absolute":
        return np.clip((primitive + 1.0) * 256.0, 0.0, 512.0)
    raise ValueError(f"Unsupported action mode: {mode}")


def save_video(frames: list[np.ndarray], video_path: str, fps: int = 15):
    if not video_path or not frames:
        return
    output = Path(video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for frame in frames:
            writer.write(np.asarray(frame, dtype=np.uint8)[..., ::-1])
    finally:
        writer.release()
    print(f"Saved evaluation video to {output}")


def main():
    args = parse_args()
    device = resolve_device("auto")
    checkpoint_path = resolve_bc_prior_path(args.checkpoint)
    payload = load_state_dict_safe(checkpoint_path, torch.device("cpu"))
    state_dict = extract_checkpoint_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unsupported BC checkpoint format in {checkpoint_path}")
    cleaned_state = clean_state_dict_keys(state_dict)
    ckpt_args = extract_checkpoint_args(payload)
    model_cfg = resolve_model_config(ckpt_args, cleaned_state)

    tokenizer_path = None
    image_hw = (224, 224)
    if model_cfg["tokenizer_name"] is not None:
        tokenizer_path = resolve_tokenizer_path(args.tokenizer_path or str(model_cfg["tokenizer_name"]))
        _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
        image_hw = (int(tokenizer_info["H"]), int(tokenizer_info["W"]))

    model = BCImagePolicy(
        image_shape=(image_hw[0], image_hw[1], 3),
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        action_chunk_size=model_cfg["action_chunk_size"],
        seq_len=model_cfg["seq_len"],
        tokenizer_ckpt=tokenizer_path,
        temporal_layers=model_cfg["temporal_layers"],
        temporal_heads=model_cfg["temporal_heads"],
        backbone_device=device,
    ).to(device)
    model.load_state_dict(cleaned_state, strict=True)
    model.eval()

    env = make_pusht_env(
        env_id=DEFAULT_PUSHT_ENV_ID,
        render_mode="rgb_array",
        image_height=image_hw[0],
        image_width=image_hw[1],
        relative=(args.action_mode == "relative"),
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=True,
        render_obs=False,
    )

    print(
        f"Loaded BC checkpoint from {checkpoint_path} | tokenizer={tokenizer_path} "
        f"| seq_len={model_cfg['seq_len']} chunk={model_cfg['action_chunk_size']} "
        f"| action_mode={args.action_mode} | device={device}"
    )

    all_returns: list[float] = []
    all_lengths: list[int] = []
    all_coverages: list[float] = []
    saved_frames: list[np.ndarray] = []

    for episode_idx in range(args.episodes):
        _, info = env.reset(seed=args.seed + episode_idx)
        del info
        frame_history: deque[np.ndarray] = deque(maxlen=model_cfg["seq_len"])
        action_buffer: deque[np.ndarray] = deque()
        done = False
        total_reward = 0.0
        step_count = 0
        final_info: dict = {}
        current_eef = extract_state_array(env.unwrapped._get_obs())[0:2].astype(np.float32)

        while not done and step_count < args.max_steps:
            frame = np.asarray(env.render(), dtype=np.uint8)
            frame_history.append(frame)
            if episode_idx == 0 and args.video_path:
                saved_frames.append(frame.copy())

            if not action_buffer:
                stacked_frames = pad_history(frame_history, model_cfg["seq_len"])
                input_tensor = torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=device)
                with torch.no_grad():
                    action_chunk = model.predict_action_chunk(input_tensor).squeeze(0).cpu().numpy()
                action_chunk = np.clip(action_chunk, -1.0, 1.0)
                action_buffer.extend(action_chunk)

            primitive = np.asarray(action_buffer.popleft(), dtype=np.float32)
            env_action = map_primitive_to_env_action(
                primitive,
                mode=args.action_mode,
                current_eef=current_eef,
                max_step_pixels=args.max_step_pixels,
            )
            obs, reward, terminated, truncated, info = env.step(env_action)
            state = extract_state_array(obs)
            current_eef = state[0:2].astype(np.float32)
            total_reward += float(reward)
            step_count += 1
            final_info = dict(info)
            done = bool(terminated or truncated)

            if args.print_every > 0 and (step_count % args.print_every == 0 or done):
                coverage = final_info.get("coverage")
                coverage_text = "n/a" if coverage is None else f"{float(coverage):.3f}"
                print(
                    f"episode={episode_idx + 1:02d}/{args.episodes:02d} step={step_count:04d} "
                    f"reward={float(reward):8.3f} total={total_reward:8.3f} coverage={coverage_text}"
                )

        all_returns.append(total_reward)
        all_lengths.append(step_count)
        if "coverage" in final_info:
            all_coverages.append(float(final_info["coverage"]))
        print(
            f"episode={episode_idx + 1:02d} return={total_reward:.3f} "
            f"steps={step_count} coverage={final_info.get('coverage', 'n/a')}"
        )

    env.close()
    save_video(saved_frames, args.video_path)

    mean_return = float(np.mean(all_returns)) if all_returns else 0.0
    mean_length = float(np.mean(all_lengths)) if all_lengths else 0.0
    mean_coverage = float(np.mean(all_coverages)) if all_coverages else float("nan")
    coverage_text = "n/a" if not all_coverages else f"{mean_coverage:.3f}"
    print(
        f"Summary: episodes={len(all_returns)} mean_return={mean_return:.3f} "
        f"mean_length={mean_length:.1f} mean_coverage={coverage_text}"
    )


if __name__ == "__main__":
    main()
