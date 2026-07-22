#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavioural_cloning.train_base import (
    PushTSequenceDataset,
    get_runtime_device,
    get_wandb_mode,
    is_rank0,
    load_ckpt,
    maybe_truncate_indices,
    normalize_image_batch,
    save_ckpt,
    seed_everything,
    split_indices,
    wandb,
)


class PushTNPZSequenceDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        *,
        seq_len: int,
        action_chunk_size: int,
        frame_stride: int,
        image_hw: tuple[int, int] | None = None,
        action_mode: str = "absolute",
        swm_action_scale: float = 100.0,
    ):
        self.npz_path = str(npz_path)
        self.seq_len = int(seq_len)
        self.action_chunk_size = int(action_chunk_size)
        self.frame_stride = int(frame_stride)
        self.raw_seq_len = (self.seq_len - 1) * self.frame_stride + self.action_chunk_size
        self.image_hw = None if image_hw is None else (int(image_hw[0]), int(image_hw[1]))
        self.action_mode = str(action_mode)
        self.swm_action_scale = float(swm_action_scale)

        with np.load(self.npz_path, allow_pickle=True) as data:
            image_key = self._resolve_first_key(data, ("images", "pixels", "observations", "obs"))
            action_key = self._resolve_first_key(data, ("actions", "action"))
            self.images = np.asarray(data[image_key])
            self.actions = np.asarray(data[action_key], dtype=np.float32)
            state_key = self._resolve_first_key(data, ("states", "state"))
            self.states = np.asarray(data[state_key], dtype=np.float32)
            self.episode_starts, self.episode_ends = self._resolve_episode_bounds(data, num_samples=len(self.actions))

        if len(self.images) != len(self.actions):
            raise ValueError(
                f"Image/action length mismatch in {self.npz_path}: images={len(self.images)} actions={len(self.actions)}"
            )
        if len(self.states) != len(self.actions):
            raise ValueError(
                f"State/action length mismatch in {self.npz_path}: states={len(self.states)} actions={len(self.actions)}"
            )
        if self.action_mode == "swm_relative":
            if self.states.shape[-1] < 2:
                raise ValueError(
                    f"SWM-relative conversion requires states with agent x/y, got shape {tuple(self.states.shape)}"
                )
            if self.swm_action_scale <= 0:
                raise ValueError("swm_action_scale must be positive")
            self.actions = np.clip(
                (self.actions - self.states[:, :2]) / self.swm_action_scale,
                -1.0,
                1.0,
            ).astype(np.float32, copy=False)

        self.valid_start_indices: list[int] = []
        for episode_start, episode_end in zip(self.episode_starts, self.episode_ends):
            max_start = int(episode_end) - self.raw_seq_len
            for index in range(int(episode_start), max_start + 1):
                self.valid_start_indices.append(index)

    @staticmethod
    def _resolve_first_key(data: np.lib.npyio.NpzFile, candidates: tuple[str, ...]) -> str:
        for key in candidates:
            if key in data:
                return key
        raise KeyError(f"Expected one of keys {candidates} in NPZ dataset, found {list(data.keys())}")

    @staticmethod
    def _resolve_episode_bounds(data: np.lib.npyio.NpzFile, *, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if "episode_starts" in data and "episode_ends" in data:
            starts = np.asarray(data["episode_starts"], dtype=np.int64)
            ends = np.asarray(data["episode_ends"], dtype=np.int64)
            return starts, ends
        if "episode_ends" in data:
            ends = np.asarray(data["episode_ends"], dtype=np.int64).reshape(-1)
            starts = np.concatenate([np.array([0], dtype=np.int64), ends[:-1]])
            return starts, ends
        if "ep_offset" in data and "ep_len" in data:
            starts = np.asarray(data["ep_offset"], dtype=np.int64)
            ends = starts + np.asarray(data["ep_len"], dtype=np.int64)
            return starts, ends
        if "episode_lengths" in data:
            lengths = np.asarray(data["episode_lengths"], dtype=np.int64)
            starts = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(lengths[:-1])])
            ends = np.cumsum(lengths)
            return starts, ends
        if "dones" in data:
            done = np.asarray(data["dones"]).astype(bool).reshape(-1)
            ends = np.nonzero(done)[0].astype(np.int64) + 1
            starts = np.concatenate([np.array([0], dtype=np.int64), ends[:-1]])
            if ends.size == 0 or int(ends[-1]) != int(num_samples):
                ends = np.concatenate([ends, np.array([num_samples], dtype=np.int64)])
                starts = np.concatenate([starts, np.array([starts[-1] if starts.size else 0], dtype=np.int64)])[: len(ends)]
            return starts, ends
        return np.array([0], dtype=np.int64), np.array([num_samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_start_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start_idx = int(self.valid_start_indices[idx])
        end_idx = start_idx + self.raw_seq_len
        images = self.images[start_idx:end_idx]
        actions = self.actions[start_idx:end_idx]

        obs_indices = np.arange(self.seq_len, dtype=np.int64) * self.frame_stride
        action_start = int(obs_indices[-1])
        action_end = action_start + self.action_chunk_size
        images = images[obs_indices]
        actions = actions[action_start:action_end].reshape(-1)

        if images.dtype != np.uint8:
            images = np.clip(images, 0.0, 255.0)
            if images.max() <= 1.5:
                images = images * 255.0
            images = images.astype(np.uint8)

        image_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        if self.image_hw is not None and tuple(image_tensor.shape[-2:]) != self.image_hw:
            image_tensor = F.interpolate(
                image_tensor,
                size=self.image_hw,
                mode="bilinear",
                align_corners=False,
            )
        action_tensor = torch.from_numpy(actions).float()
        return {
            "image": image_tensor,
            "action": action_tensor,
        }


class CNNBackbone(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        feature_dim: int = 256,
    ):
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


class DirectChunkPolicyHead(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        seq_len: int,
        hidden_dim: int,
        action_dim: int,
        dropout: float = 0.0,
        output_tanh: bool = False,
    ):
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


class CNNBCPolicy(nn.Module):
    def __init__(self, backbone: CNNBackbone, classifier: DirectChunkPolicyHead):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x_btchw)
        return self.classifier(features)


def _action_scale_for_training(actions: torch.Tensor, *, normalize_actions: bool, action_scale: float) -> torch.Tensor:
    if not normalize_actions:
        return actions
    return actions / float(action_scale)


def _resolve_action_mode(args: argparse.Namespace) -> str:
    action_mode = str(args.action_mode)
    if action_mode == "auto":
        return "absolute" if str(args.dataset).lower().endswith(".npz") else "relative"
    return action_mode


def _evaluate_validation_scaled(
    model: nn.Module,
    loader: DataLoader | None,
    *,
    device: torch.device,
    normalize_actions: bool,
    action_scale: float,
    max_batches: int = 0,
) -> dict[str, float]:
    if loader is None:
        return {}

    was_training = model.training
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = normalize_image_batch(batch["image"]).to(device, non_blocking=True)
            target_actions = batch["action"].to(device, non_blocking=True).to(torch.float32)
            target_actions = _action_scale_for_training(
                target_actions,
                normalize_actions=normalize_actions,
                action_scale=action_scale,
            )
            pred = model(images)
            loss_sum += float(F.mse_loss(pred, target_actions).item())
            mae_sum += float(torch.mean(torch.abs(pred - target_actions)).item())
            num_batches += 1
    if was_training:
        model.train()
    if num_batches == 0:
        return {}
    return {
        "validation/loss": loss_sum / num_batches,
        "validation/mae": mae_sum / num_batches,
        "validation/num_batches": float(num_batches),
    }


def _seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _stats_sidecar_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.suffix:
        return checkpoint_path.with_name(f"{checkpoint_path.stem}_stats.pth")
    return Path(f"{checkpoint_path}_stats.pth")


def _build_bc_stats(
    args: argparse.Namespace,
    *,
    dataset: Dataset,
    cnn_feature_dim: int,
    train_size: int,
    val_size: int,
) -> dict[str, object]:
    stats = {
        "frame_stack": int(args.seq_len),
        "frame_stride": int(args.frame_stride),
        "action_chunk_size": int(args.action_chunk_size),
        "cnn_feature_dim": int(cnn_feature_dim),
        "hidden_dim": int(args.hidden_dim),
        "action_dim": 2,
        "action_mode": str(args.action_mode),
        "normalize_actions": bool(args.normalize_actions),
        "action_scale": float(args.action_scale),
        "swm_action_scale": float(args.swm_action_scale),
        "dataset": str(args.dataset),
        "dataset_windows": int(len(dataset)),
        "train_windows": int(train_size),
        "val_windows": int(val_size),
        "tokenizer_ckpt_name": None,
        "policy_style": "direct_chunk_cnn",
    }
    underlying = getattr(dataset, "dataset", None)
    if underlying is not None:
        stats["source_dataset_type"] = type(underlying).__name__
    return stats


def _prepare_dataset(args: argparse.Namespace) -> Dataset:
    dataset_path = str(args.dataset)
    if dataset_path.lower().endswith(".npz"):
        image_hw = (args.image_hw[0], args.image_hw[1]) if args.image_hw else None
        full_dataset = PushTNPZSequenceDataset(
            dataset_path,
            seq_len=args.seq_len,
            action_chunk_size=args.action_chunk_size,
            frame_stride=args.frame_stride,
            image_hw=image_hw,
            action_mode=str(args.action_mode),
            swm_action_scale=float(args.swm_action_scale),
        )
    else:
        full_dataset = PushTSequenceDataset(
            h5_path=dataset_path,
            seq_len=args.seq_len,
            action_chunk_size=args.action_chunk_size,
            frame_stride=args.frame_stride,
        )
    return full_dataset


def train(args: argparse.Namespace):
    device, device_type = get_runtime_device()
    seed_everything(int(args.seed))
    print(f"Using device: {device}")
    args.action_mode = _resolve_action_mode(args)
    dataset = _prepare_dataset(args)
    train_indices, val_indices = split_indices(len(dataset), args.val_frac, args.seed)
    train_indices = maybe_truncate_indices(train_indices, int(args.max_train_samples), seed=args.seed)
    val_indices = maybe_truncate_indices(val_indices, int(args.max_val_samples), seed=args.seed + 1)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed))
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type in {"cuda", "xpu"}),
        drop_last=True,
        worker_init_fn=_seed_dataloader_worker,
        generator=loader_generator,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type in {"cuda", "xpu"}),
            drop_last=False,
            worker_init_fn=_seed_dataloader_worker,
        )

    action_dim = int(args.action_chunk_size) * 2
    backbone = CNNBackbone(in_channels=3, feature_dim=args.cnn_feature_dim)
    classifier = DirectChunkPolicyHead(
        in_dim=args.cnn_feature_dim,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        action_dim=action_dim,
        dropout=args.dropout,
        output_tanh=args.action_output_tanh,
    )
    model = CNNBCPolicy(backbone=backbone, classifier=classifier).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.ckpt_dir)
    if is_rank0():
        print(
            f"CNN BC: dataset={args.dataset} "
            f"| windows total={len(dataset)} train={len(train_dataset)} val={0 if val_dataset is None else len(val_dataset)} "
            f"| cnn_feature_dim={args.cnn_feature_dim} frame_stack={args.seq_len} stride={args.frame_stride} "
            f"chunk={args.action_chunk_size} action_mode={args.action_mode}"
        )
        bc_stats = _build_bc_stats(
            args,
            dataset=dataset,
            cnn_feature_dim=args.cnn_feature_dim,
            train_size=len(train_dataset),
            val_size=0 if val_dataset is None else len(val_dataset),
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(bc_stats, _stats_sidecar_path(ckpt_dir / "best.pt"))
        torch.save(bc_stats, _stats_sidecar_path(ckpt_dir / "latest.pt"))
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode=get_wandb_mode(args),
            config={**vars(args), **bc_stats},
        )

    best_val = float("inf")
    step = 0
    start_epoch = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        step, resumed_epoch = load_ckpt(resume_path, model=model, opt=opt, scaler=None)
        start_epoch = resumed_epoch + 1
        if is_rank0():
            print(
                f"Resumed training from {resume_path} (step={step}, finished_epoch={resumed_epoch + 1}, starting_epoch={start_epoch + 1})"
            )

    t0 = time.time()
    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        num_batches = 0
        for batch in loader:
            images = normalize_image_batch(batch["image"]).to(device, non_blocking=True)
            target_actions = batch["action"].to(device, non_blocking=True).to(torch.float32)
            target_actions = _action_scale_for_training(
                target_actions,
                normalize_actions=bool(args.normalize_actions),
                action_scale=float(args.action_scale),
            )

            pred = model(images)
            loss = F.mse_loss(pred, target_actions)
            mae = torch.mean(torch.abs(pred - target_actions))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item())
            epoch_mae += float(mae.item())
            num_batches += 1
            step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_mae = epoch_mae / max(num_batches, 1)
        metrics: dict[str, float] = {
            "train/loss": avg_loss,
            "train/mae": avg_mae,
            "train/epoch": float(epoch + 1),
            "train/time_hours": (time.time() - t0) / 3600.0,
        }
        if val_loader is not None:
            val_metrics = _evaluate_validation_scaled(
                model,
                val_loader,
                device=device,
                normalize_actions=bool(args.normalize_actions),
                action_scale=float(args.action_scale),
                max_batches=int(args.val_max_batches),
            )
            metrics.update(val_metrics)
            val_loss = float(val_metrics.get("validation/loss", avg_loss))
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(
                    ckpt_dir / "best.pt",
                    step=step,
                    epoch=epoch,
                    model=model,
                    opt=opt,
                    scaler=None,
                    args=args,
                )
                if is_rank0():
                    torch.save(bc_stats, _stats_sidecar_path(ckpt_dir / "best.pt"))

        if is_rank0():
            wandb.log(metrics, step=epoch + 1)
            print(
                f"epoch {epoch + 1:04d}/{int(args.epochs):04d} "
                f"| loss={avg_loss:.6f} | mae={avg_mae:.6f} "
                f"| val={metrics.get('validation/loss', float('nan')):.6f}"
            )

        if (epoch + 1) % int(args.save_every) == 0:
            save_ckpt(
                ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                step=step,
                epoch=epoch,
                model=model,
                opt=opt,
                scaler=None,
                args=args,
            )

    save_ckpt(
        ckpt_dir / "latest.pt",
        step=step,
        epoch=max(int(args.epochs) - 1, 0),
        model=model,
        opt=opt,
        scaler=None,
        args=args,
    )
    if is_rank0():
        torch.save(bc_stats, _stats_sidecar_path(ckpt_dir / "latest.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN Behavioural Cloning training directly from image pixels.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 (.h5) or NPZ dataset.")
    parser.add_argument("--seq_len", type=int, default=3)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--action_chunk_size", type=int, default=1)
    parser.add_argument("--cnn_feature_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--action_output_tanh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--val_max_batches", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--image_hw", type=int, nargs=2, default=None, help="Optional image resize (H W) for NPZ dataset.")
    parser.add_argument("--normalize_actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--swm_action_scale", type=float, default=100.0)
    parser.add_argument(
        "--action_mode",
        type=str,
        default="auto",
        choices=["auto", "relative", "absolute", "swm_relative"],
    )
    parser.add_argument("--ckpt_dir", type=str, default="local_models/behavior_cloning/cnn_bc")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default="pusht-cnn-bc")
    parser.add_argument("--wandb_run_name", type=str, default="cnn-bc")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
