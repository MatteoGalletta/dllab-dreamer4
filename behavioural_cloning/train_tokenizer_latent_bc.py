#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
    CachedFeatureDataset,
    PushTSequenceDataset,
    TokenizerBackbone,
    build_feature_cache,
    get_runtime_device,
    get_wandb_mode,
    is_rank0,
    load_feature_cache,
    load_tokenizer_encoder,
    maybe_truncate_indices,
    normalize_image_batch,
    resolve_tokenizer_path,
    save_ckpt,
    seed_everything,
    split_indices,
    tokenizer_feature_cache_path,
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
    ):
        self.npz_path = str(npz_path)
        self.seq_len = int(seq_len)
        self.action_chunk_size = int(action_chunk_size)
        self.frame_stride = int(frame_stride)
        self.raw_seq_len = (self.seq_len - 1) * self.frame_stride + self.action_chunk_size
        self.image_hw = None if image_hw is None else (int(image_hw[0]), int(image_hw[1]))

        with np.load(self.npz_path, allow_pickle=True) as data:
            image_key = self._resolve_first_key(data, ("images", "pixels", "observations", "obs"))
            action_key = self._resolve_first_key(data, ("actions", "action"))
            self.images = np.asarray(data[image_key])
            self.actions = np.asarray(data[action_key], dtype=np.float32)
            self.episode_starts, self.episode_ends = self._resolve_episode_bounds(data, num_samples=len(self.actions))

        if len(self.images) != len(self.actions):
            raise ValueError(
                f"Image/action length mismatch in {self.npz_path}: images={len(self.images)} actions={len(self.actions)}"
            )

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


class TokenizerLatentBCPolicy(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        frame_stack: int,
        action_dim: int,
        hidden_dim: int,
        action_chunk_size: int,
    ):
        super().__init__()
        if int(action_chunk_size) < 1:
            raise ValueError("action_chunk_size must be at least 1")
        self.latent_dim = int(latent_dim)
        self.frame_stack = int(frame_stack)
        self.action_dim = int(action_dim)
        self.action_chunk_size = int(action_chunk_size)
        input_dim = self.latent_dim * self.frame_stack
        output_dim = self.action_dim * self.action_chunk_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), output_dim),
        )

    def forward(self, stacked_latents: torch.Tensor) -> torch.Tensor:
        if stacked_latents.ndim != 3:
            raise ValueError(
                f"Expected stacked latents with shape (B, T, D), got {tuple(stacked_latents.shape)}"
            )
        batch_size, frames, latent_dim = stacked_latents.shape
        if frames != self.frame_stack:
            raise ValueError(f"Expected frame_stack={self.frame_stack}, got {frames}")
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        flattened = stacked_latents.reshape(batch_size, -1)
        flat_actions = self.net(flattened)
        return flat_actions.view(batch_size, self.action_chunk_size * self.action_dim)


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
) -> dict[str, float]:
    if loader is None:
        return {}

    was_training = model.training
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True).to(torch.float32)
            target_actions = batch["action"].to(device, non_blocking=True).to(torch.float32)
            target_actions = _action_scale_for_training(
                target_actions,
                normalize_actions=normalize_actions,
                action_scale=action_scale,
            )
            pred = model(features)
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


def _prepare_cached_dataset(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Dataset, int, str]:
    dataset_path = str(args.dataset)
    if dataset_path.lower().endswith(".npz"):
        from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt

        tokenizer_path = resolve_tokenizer_path(args.tokenizer_ckpt_name)
        _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
        full_dataset = PushTNPZSequenceDataset(
            dataset_path,
            seq_len=args.seq_len,
            action_chunk_size=args.action_chunk_size,
            frame_stride=args.frame_stride,
            image_hw=(int(tokenizer_info["H"]), int(tokenizer_info["W"])),
        )
    else:
        full_dataset = PushTSequenceDataset(
            h5_path=dataset_path,
            seq_len=args.seq_len,
            action_chunk_size=args.action_chunk_size,
            frame_stride=args.frame_stride,
        )
    tokenizer_path = resolve_tokenizer_path(args.tokenizer_ckpt_name)
    encoder = load_tokenizer_encoder(tokenizer_path)
    backbone = TokenizerBackbone(
        encoder,
        patch=int(encoder.patch),
        output_dim=int(encoder.n_latents) * int(encoder.bottleneck_proj.out_features),
    )
    cache_path = tokenizer_feature_cache_path(args, tokenizer_path)
    cached_features = None
    if args.rebuild_latent_cache or not cache_path.exists():
        cached_features = build_feature_cache(
            full_dataset,
            tokenizer_backbone=backbone.to(device),
            cache_path=cache_path,
            device=device,
            batch_size=args.latent_cache_batch_size or args.batch_size,
            image_batch_normalizer=normalize_image_batch,
            metadata={
                "dataset": str(args.dataset),
                "tokenizer_ckpt": str(tokenizer_path),
                "seq_len": int(args.seq_len),
                "frame_stride": int(args.frame_stride),
                "action_chunk_size": int(args.action_chunk_size),
                "raw_feature_dim": int(backbone.raw_feature_dim),
                "cache_type": "tokenizer_raw_latents",
            },
        )
    if cached_features is None:
        cached_features = load_feature_cache(cache_path)
    if cached_features is None:
        raise RuntimeError(f"Tokenizer latent cache missing or invalid: {cache_path}")
    dataset = CachedFeatureDataset(full_dataset, cached_features)
    return dataset, int(backbone.raw_feature_dim), str(tokenizer_path)


def train(args: argparse.Namespace):
    device, device_type = get_runtime_device()
    seed_everything(int(args.seed))
    print(f"Using device: {device}")
    args.action_mode = _resolve_action_mode(args)
    dataset, latent_dim, tokenizer_path = _prepare_cached_dataset(args, device=device)
    train_indices, val_indices = split_indices(len(dataset), args.val_frac, args.seed)
    train_indices = maybe_truncate_indices(train_indices, int(args.max_train_samples), seed=args.seed)
    val_indices = maybe_truncate_indices(val_indices, int(args.max_val_samples), seed=args.seed + 1)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type in {"cuda", "xpu"}),
        drop_last=True,
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
        )

    model = TokenizerLatentBCPolicy(
        latent_dim=latent_dim,
        frame_stack=args.seq_len,
        action_dim=2,
        hidden_dim=args.hidden_dim,
        action_chunk_size=args.action_chunk_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if is_rank0():
        print(
            f"Tokenizer-latent BC: dataset={args.dataset} tokenizer={tokenizer_path} "
            f"| windows total={len(dataset)} train={len(train_dataset)} val={0 if val_dataset is None else len(val_dataset)} "
            f"| latent_dim={latent_dim} frame_stack={args.seq_len} stride={args.frame_stride} "
            f"chunk={args.action_chunk_size} action_mode={args.action_mode}"
        )
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode=get_wandb_mode(args),
            config=vars(args),
        )

    ckpt_dir = Path(args.ckpt_dir)
    best_val = float("inf")
    step = 0
    t0 = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        num_batches = 0
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True).to(torch.float32)
            target_actions = batch["action"].to(device, non_blocking=True).to(torch.float32)
            target_actions = _action_scale_for_training(
                target_actions,
                normalize_actions=bool(args.normalize_actions),
                action_scale=float(args.action_scale),
            )

            pred = model(features)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal tokenizer-latent BC in the style of the other group.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tokenizer_ckpt_name", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=3)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--action_chunk_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=256)
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
    parser.add_argument("--latent_cache_batch_size", type=int, default=0)
    parser.add_argument("--rebuild_latent_cache", action="store_true")
    parser.add_argument("--normalize_actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--action_mode", type=str, default="auto", choices=["auto", "relative", "absolute"])
    parser.add_argument("--ckpt_dir", type=str, default="local_models/behavior_cloning/tokenizer_latent_bc")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default="pusht-tokenizer-latent-bc")
    parser.add_argument("--wandb_run_name", type=str, default="tokenizer-latent-bc")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
