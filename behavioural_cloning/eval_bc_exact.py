#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, make_pusht_env
from ppo_online.model_paths import resolve_bc_prior_path, resolve_tokenizer_path
from ppo_online.networks import BCActionClassifier
from ppo_online.render import extract_checkpoint_args, extract_checkpoint_state_dict, load_state_dict_safe
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt
from ppo_online.train import PushTDenseRewardWrapper, extract_state_array, resolve_device

DREAMER4_MODEL_PATH = PROJECT_ROOT / "dreamer4-src" / "dreamer4" / "model.py"


def load_dreamer4_model_module():
    spec = importlib.util.spec_from_file_location("dreamer4_model_eval_bc", DREAMER4_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Dreamer4 model module from {DREAMER4_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dreamer4_model = load_dreamer4_model_module()
temporal_patchify = dreamer4_model.temporal_patchify


PUSHT_DATASET_DELTA_SCALE = 39.3


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    return cv2


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


class TokenizerBackbone(nn.Module):
    def __init__(self, encoder: nn.Module, patch: int, *, output_dim: int | None):
        super().__init__()
        self.encoder = encoder
        self.patch = int(patch)
        self.raw_feature_dim = int(self.encoder.n_latents) * int(self.encoder.bottleneck_proj.out_features)
        if output_dim is None or int(output_dim) == self.raw_feature_dim:
            self.feature_dim = self.raw_feature_dim
            self.projector = nn.Identity()
        else:
            self.feature_dim = int(output_dim)
            self.projector = nn.Sequential(
                nn.LayerNorm(self.raw_feature_dim),
                nn.Linear(self.raw_feature_dim, self.feature_dim),
                nn.ReLU(),
            )

    def extract_features(self, x_btchw: torch.Tensor) -> torch.Tensor:
        patches = temporal_patchify(x_btchw, self.patch)
        with torch.no_grad():
            z, _ = self.encoder(patches)
        return z.reshape(z.shape[0], z.shape[1], -1)

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        return self.projector(self.extract_features(x_btchw))


class DirectChunkPolicyHead(nn.Module):
    def __init__(self, *, in_dim: int, seq_len: int, hidden_dim: int, action_dim: int, dropout: float, output_tanh: bool):
        super().__init__()
        self.seq_len = int(seq_len)
        self.action_dim = int(action_dim)
        self.output_tanh = bool(output_tanh)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim) * self.seq_len, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.action_dim),
        )

    def forward(self, features_btD: torch.Tensor) -> torch.Tensor:
        if features_btD.ndim != 3:
            raise ValueError(f"Expected feature sequence with shape (B, T, D), got {tuple(features_btD.shape)}")
        batch, steps, feature_dim = features_btD.shape
        if steps != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {steps}")
        logits = self.net(features_btD.reshape(batch, steps * feature_dim))
        return torch.tanh(logits) if self.output_tanh else logits


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
        tokenizer_feature_dim: int,
        policy_style: str,
        temporal_layers: int,
        temporal_heads: int,
        temporal_context: int,
        backbone_device: torch.device | str,
        action_output_tanh: bool,
    ):
        super().__init__()
        action_dim = int(action_chunk_size) * 2
        if tokenizer_ckpt is not None:
            tokenizer, info = load_tokenizer_from_ckpt(tokenizer_ckpt, torch.device(backbone_device))
            encoder = tokenizer.encoder
            encoder.requires_grad_(False)
            encoder.eval()
            encoder.patch = int(info["patch"])
            self.backbone = TokenizerBackbone(
                encoder,
                patch=int(encoder.patch),
                output_dim=None if tokenizer_feature_dim is None else int(tokenizer_feature_dim),
            )
        else:
            self.backbone = CNNBackbone(in_channels=int(image_shape[2]))
        self.policy_style = str(policy_style)
        if self.policy_style == "direct_chunk_cnn":
            self.classifier = DirectChunkPolicyHead(
                in_dim=self.backbone.feature_dim,
                seq_len=int(seq_len),
                hidden_dim=int(hidden_dim),
                action_dim=action_dim,
                dropout=float(dropout),
                output_tanh=bool(action_output_tanh),
            )
        else:
            self.classifier = BCActionClassifier(
                in_dim=self.backbone.feature_dim,
                hidden_dim=int(hidden_dim),
                action_dim=action_dim,
                dropout=float(dropout),
                temporal_layers=int(temporal_layers),
                temporal_heads=int(temporal_heads),
                max_seq_len=int(seq_len),
                temporal_context=int(temporal_context),
                output_tanh=bool(action_output_tanh),
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
        elif isinstance(self.backbone, TokenizerBackbone):
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
        if actions.ndim == 3:
            actions = actions[:, -1, :]
        return actions.view(actions.shape[0], self.action_chunk_size, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC checkpoint with a BC-native open-loop action queue.")
    parser.add_argument("--checkpoint", type=str, default="local_models/behavior_cloning/bc_best.pt")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("relative", "delta", "absolute", "swm_relative"),
        default=None,
        help="How to map predicted primitive actions into env actions. Defaults to the checkpoint action_mode.",
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
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.35)
    parser.add_argument(
        "--fixed-target-eval",
        action="store_true",
        help="Rigidly align sampled PushT tasks to the fixed target pose, matching the other group's optional eval mode.",
    )
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
    frame_stride = int(ckpt_args.get("frame_stride", 1))
    action_chunk_size = int(ckpt_args.get("action_chunk_size", 5))
    hidden_dim = int(ckpt_args.get("hidden_dim", 512))
    has_tokenizer_projector = any(key.startswith("backbone.projector.") for key in cleaned_state)
    tokenizer_feature_dim_arg = ckpt_args.get("tokenizer_feature_dim")
    tokenizer_feature_dim = None if tokenizer_feature_dim_arg is None else int(tokenizer_feature_dim_arg)
    if tokenizer_name is not None and not has_tokenizer_projector:
        tokenizer_feature_dim = None
    elif tokenizer_name is not None and tokenizer_feature_dim is None:
        tokenizer_feature_dim = 256
    dropout = float(ckpt_args.get("dropout", 0.05))
    has_temporal = any(key.startswith("classifier.temporal_") for key in cleaned_state)
    policy_style = ckpt_args.get("policy_style")
    if policy_style is None:
        policy_style = "sequence_classifier" if has_temporal else "direct_chunk_cnn"
    temporal_layers = int(ckpt_args.get("temporal_layers", 2 if has_temporal else 0))
    temporal_heads = int(ckpt_args.get("temporal_heads", 4))
    temporal_context = int(ckpt_args.get("temporal_context", 3))
    action_output_tanh = bool(ckpt_args.get("action_output_tanh", True))
    action_mode = str(ckpt_args.get("action_mode", "relative"))
    swm_action_scale = float(ckpt_args.get("swm_action_scale", 100.0))
    image_hw_raw = ckpt_args.get("image_hw")
    image_hw = None
    if image_hw_raw is not None:
        try:
            if len(image_hw_raw) == 2 and image_hw_raw[0] is not None and image_hw_raw[1] is not None:
                image_hw = (int(image_hw_raw[0]), int(image_hw_raw[1]))
        except TypeError:
            image_hw = None
    return {
        "tokenizer_name": tokenizer_name,
        "seq_len": seq_len,
        "frame_stride": frame_stride,
        "action_chunk_size": action_chunk_size,
        "hidden_dim": hidden_dim,
        "tokenizer_feature_dim": tokenizer_feature_dim,
        "has_tokenizer_projector": has_tokenizer_projector,
        "dropout": dropout,
        "policy_style": policy_style,
        "temporal_layers": temporal_layers,
        "temporal_heads": temporal_heads,
        "temporal_context": temporal_context,
        "action_output_tanh": action_output_tanh,
        "action_mode": action_mode,
        "swm_action_scale": swm_action_scale,
        "image_hw": image_hw,
        "dataset": ckpt_args.get("dataset"),
    }


def infer_cnn_image_hw(model_cfg: dict[str, Any]) -> tuple[int, int]:
    if model_cfg["tokenizer_name"] is not None:
        raise ValueError("Tokenizer-backed checkpoints should resolve image size from the tokenizer.")
    image_hw = model_cfg.get("image_hw")
    if image_hw is not None:
        return int(image_hw[0]), int(image_hw[1])

    dataset_path = model_cfg.get("dataset")
    if dataset_path:
        dataset_path = Path(str(dataset_path))
        if dataset_path.suffix.lower() == ".npz" and dataset_path.exists():
            with np.load(str(dataset_path), allow_pickle=True) as data:
                for key in ("images", "pixels", "observations", "obs"):
                    if key in data:
                        sample_shape = tuple(np.asarray(data[key]).shape)
                        if len(sample_shape) >= 4:
                            return int(sample_shape[-3]), int(sample_shape[-2])
                        break
    return (224, 224)


def pad_history(frames: deque[np.ndarray], seq_len: int, frame_stride: int) -> np.ndarray:
    history = list(frames)
    if not history:
        raise ValueError("frame history is empty")
    newest = len(history) - 1
    indices = [max(0, newest - i * int(frame_stride)) for i in range(seq_len - 1, -1, -1)]
    return np.stack([history[idx] for idx in indices], axis=0)


def map_primitive_to_env_action(
    primitive: np.ndarray,
    *,
    mode: str,
    current_eef: np.ndarray,
    max_step_pixels: float,
    action_output_tanh: bool,
) -> np.ndarray:
    primitive = np.asarray(primitive, dtype=np.float32)
    if mode in {"relative", "swm_relative"}:
        return primitive
    if mode == "delta":
        return np.clip(current_eef + primitive * float(max_step_pixels), 0.0, 512.0)
    if mode == "absolute":
        if action_output_tanh:
            return np.clip((primitive + 1.0) * 256.0, 0.0, 512.0)
        return np.clip(primitive, 0.0, 512.0)
    raise ValueError(f"Unsupported action mode: {mode}")


def ensemble_primitive(
    pending_chunks: deque[dict[str, np.ndarray | int]],
    chunk_size: int,
    decay: float,
) -> np.ndarray:
    candidates = []
    weights = []
    for age, entry in enumerate(reversed(pending_chunks)):
        offset = int(entry["offset"])
        if offset >= chunk_size:
            continue
        candidates.append(np.asarray(entry["chunk"], dtype=np.float32)[offset])
        weights.append(math.exp(-decay * age))

    if not candidates:
        return np.zeros(2, dtype=np.float32)

    weights_np = np.asarray(weights, dtype=np.float32)
    weights_np /= max(weights_np.sum(), 1e-8)
    return np.sum(np.asarray(candidates, dtype=np.float32) * weights_np[:, None], axis=0)


def save_video(frames: list[np.ndarray], video_path: str, fps: int = 15):
    if not video_path or not frames:
        return
    output = Path(video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    ffmpeg_path = shutil.which("ffmpeg")
    cv2 = _import_cv2()
    if cv2 is not None:
        temp_output = output.with_name(f"{output.stem}.raw{output.suffix}")
        writer = None
        opened_codec = None
        for codec in ("mp4v", "avc1", "H264"):
            candidate = cv2.VideoWriter(str(temp_output), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
            if candidate.isOpened():
                writer = candidate
                opened_codec = codec
                break
            candidate.release()
        if writer is None:
            raise RuntimeError(f"Could not open video writer for {output}")
        try:
            for frame in frames:
                writer.write(np.asarray(frame, dtype=np.uint8)[..., ::-1])
        finally:
            writer.release()
        finalized_codec = opened_codec
        if ffmpeg_path is not None:
            cmd = [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_output),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                finalized_codec = "libx264"
                temp_output.unlink(missing_ok=True)
            else:
                temp_output.replace(output)
        else:
            temp_output.replace(output)
        print(f"Saved evaluation video to {output} using codec={finalized_codec}")
        return

    if ffmpeg_path is not None:
        temp_dir = output.parent / f".{output.stem}_frames"
        temp_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            frame_path = temp_dir / f"frame_{index:06d}.png"
            from imageio.v2 import imwrite

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
        if result.returncode == 0:
            for frame_path in temp_dir.glob("*.png"):
                frame_path.unlink(missing_ok=True)
            temp_dir.rmdir()
            print(f"Saved evaluation video to {output} using codec=libx264")
            return
        raise RuntimeError(
            f"Could not encode video with ffmpeg for {output}: {result.stderr.strip() or result.stdout.strip()}"
        )

    raise RuntimeError(
        "Video saving requires either OpenCV with video support or ffmpeg in PATH. "
        "Run eval without --video-path if neither is available."
    )


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
    effective_action_mode = str(args.action_mode or model_cfg["action_mode"])

    tokenizer_path = None
    image_hw = (224, 224)
    if model_cfg["tokenizer_name"] is not None:
        tokenizer_path = resolve_tokenizer_path(args.tokenizer_path or str(model_cfg["tokenizer_name"]))
        _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
        image_hw = (int(tokenizer_info["H"]), int(tokenizer_info["W"]))
    else:
        image_hw = infer_cnn_image_hw(model_cfg)

    model = BCImagePolicy(
        image_shape=(image_hw[0], image_hw[1], 3),
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        action_chunk_size=model_cfg["action_chunk_size"],
        seq_len=model_cfg["seq_len"],
        tokenizer_ckpt=tokenizer_path,
        tokenizer_feature_dim=model_cfg["tokenizer_feature_dim"],
        policy_style=model_cfg["policy_style"],
        temporal_layers=model_cfg["temporal_layers"],
        temporal_heads=model_cfg["temporal_heads"],
        temporal_context=model_cfg["temporal_context"],
        backbone_device=device,
        action_output_tanh=model_cfg["action_output_tanh"],
    ).to(device)
    model.load_state_dict(cleaned_state, strict=True)
    model.eval()

    env = make_pusht_env(
        env_id=DEFAULT_PUSHT_ENV_ID,
        render_mode="rgb_array",
        image_height=image_hw[0],
        image_width=image_hw[1],
        relative=(effective_action_mode in {"relative", "swm_relative"}),
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=args.fixed_target_eval,
        render_obs=False,
    )
    env = PushTDenseRewardWrapper(env, env_id=DEFAULT_PUSHT_ENV_ID)

    print(
        f"Loaded BC checkpoint from {checkpoint_path} | tokenizer={tokenizer_path} "
        f"| seq_len={model_cfg['seq_len']} frame_stride={model_cfg['frame_stride']} "
        f"chunk={model_cfg['action_chunk_size']} "
        f"| image_hw={image_hw[0]}x{image_hw[1]} "
        f"| action_mode={effective_action_mode} | swm_action_scale={model_cfg['swm_action_scale']:.3f} "
        f"| temporal_ensemble={args.temporal_ensemble} "
        f"| fixed_target_eval={args.fixed_target_eval} "
        f"| delta_scale_hint={PUSHT_DATASET_DELTA_SCALE:.1f} | device={device}"
    )

    all_returns: list[float] = []
    all_lengths: list[int] = []
    all_coverages: list[float] = []
    saved_frames: list[np.ndarray] = []

    for episode_idx in range(args.episodes):
        _, info = env.reset(seed=args.seed + episode_idx)
        del info
        max_history_len = (int(model_cfg["seq_len"]) - 1) * int(model_cfg["frame_stride"]) + 1
        frame_history: deque[np.ndarray] = deque(maxlen=max_history_len)
        action_buffer: deque[np.ndarray] = deque()
        pending_chunks: deque[dict[str, np.ndarray | int]] = deque()
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

            if args.temporal_ensemble or not action_buffer:
                stacked_frames = pad_history(
                    frame_history,
                    model_cfg["seq_len"],
                    model_cfg["frame_stride"],
                )
                input_tensor = torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=device)
                with torch.no_grad():
                    action_chunk = model.predict_action_chunk(input_tensor).squeeze(0).cpu().numpy()
                if args.temporal_ensemble:
                    pending_chunks.append({"chunk": action_chunk, "offset": 0})
                else:
                    action_buffer.extend(action_chunk)

            if args.temporal_ensemble:
                primitive = ensemble_primitive(
                    pending_chunks,
                    chunk_size=model_cfg["action_chunk_size"],
                    decay=args.temporal_ensemble_decay,
                )
            else:
                primitive = np.asarray(action_buffer.popleft(), dtype=np.float32)
            env_action = map_primitive_to_env_action(
                primitive,
                mode=effective_action_mode,
                current_eef=current_eef,
                max_step_pixels=args.max_step_pixels,
                action_output_tanh=bool(model_cfg["action_output_tanh"]),
            )
            obs, reward, terminated, truncated, info = env.step(env_action)
            state = extract_state_array(obs)
            current_eef = state[0:2].astype(np.float32)
            total_reward += float(reward)
            step_count += 1
            final_info = dict(info)
            done = bool(terminated or truncated)

            if args.temporal_ensemble:
                for entry in pending_chunks:
                    entry["offset"] = int(entry["offset"]) + 1
                while pending_chunks and int(pending_chunks[0]["offset"]) >= model_cfg["action_chunk_size"]:
                    pending_chunks.popleft()

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
