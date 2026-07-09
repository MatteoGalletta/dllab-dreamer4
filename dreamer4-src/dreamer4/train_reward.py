# train_reward.py
import os
import time
import random
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.PushTDataLoader import PushTSequenceDataset
from model import Encoder, temporal_patchify  # reuse tokenizer's encoder + patchify

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------
# Reward labeling
# ----------------------------------------------------------------------------
def compute_terminal_reward_labels(ep_len: int, tail_frac: float = 0.1, min_tail: int = 3) -> np.ndarray:
    """
    Fallback reward: assumes expert demos end in success. Labels the final
    `tail_frac` fraction of each episode (at least `min_tail` frames) as
    reward=1, everything before as reward=0.

    Replace this with compute_coverage_reward_labels() once we confirm the
    `state` layout — this is a placeholder that only encodes "episode is
    ending", not "task is actually solved" at each timestep.
    """
    tail = max(min_tail, int(round(ep_len * tail_frac)))
    tail = min(tail, ep_len)
    labels = np.zeros(ep_len, dtype=np.float32)
    labels[ep_len - tail:] = 1.0
    return labels


def cluster_episode_goals(h5_path: str, xy_tol: float = 10.0) -> tuple[dict, list]:
    """
    Memory-efficient goal estimation.
    1. Increases xy_tol slightly (e.g., from 1.0 to 10.0 pixels) to cleanly group
       episodes into shared starting scene zones.
    2. Uses native structures safely without building bloated nested dictionary arrays.
    """
    import h5py

    # 1. Read only the small metadata arrays first to minimize overhead
    with h5py.File(h5_path, "r") as f:
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        n_eps = len(ep_offset)

        starts = np.zeros((n_eps, 2), dtype=np.float32)
        finals = np.zeros((n_eps, 3), dtype=np.float32)

        # 2. Strided extraction of index locations instead of pulling massive blocks
        state_ds = f["state"]
        for i in range(n_eps):
            s = int(ep_offset[i])
            e = s + int(ep_len[i])

            # Read ONLY the single required 7D rows, NOT whole slices
            starts[i] = state_ds[s, 2:4]
            finals[i] = state_ds[e - 1, 2:5]

    # Cleanly cluster by rounded start pixel zones
    keys = [tuple(np.round(starts[i] / xy_tol).astype(int)) for i in range(n_eps)]

    scene_to_finals = {}
    for i, k in enumerate(keys):
        if k not in scene_to_finals:
            scene_to_finals[k] = []
        scene_to_finals[k].append(finals[i])

    # Standardize median evaluations
    scene_to_goal = {
        k: np.median(np.array(v, dtype=np.float32), axis=0)
        for k, v in scene_to_finals.items()
    }

    # Explicitly break reference pointers to free memory immediately
    del starts
    del finals
    import gc
    gc.collect()

    return scene_to_goal, keys


def compute_coverage_reward_labels(
    block_states: np.ndarray,
    goal_pose: np.ndarray,
    xy_norm: float = 100.0,
    theta_norm: float = 1.0,
    success_dist: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense shaped reward based on distance from the current block pose to the
    estimated scene goal pose. Returns both a continuous reward in [0, 1]
    (1 = at goal) and a binary success label (distance below threshold).

    block_states: (T, 3) [x, y, theta]
    goal_pose: (3,) [goal_x, goal_y, goal_theta]
    """
    dxy = (block_states[:, :2] - goal_pose[:2]) / xy_norm
    dtheta = (block_states[:, 2] - goal_pose[2]) / theta_norm
    dtheta = np.abs(np.mod(dtheta + np.pi, 2 * np.pi) - np.pi) / np.pi

    dist = np.sqrt((dxy ** 2).sum(axis=-1) + dtheta ** 2)
    dense_reward = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
    binary_reward = (dist < success_dist).astype(np.float32)
    return dense_reward, binary_reward


class PushTRewardDataset(PushTSequenceDataset):
    """
    Wraps PushTSequenceDataset and attaches a reward label per timestep
    (post action-chunk subsampling, so shape matches `image`/`state`: seq_len).

    reward_mode="coverage" (default, recommended): uses the goal pose
    estimated per-scene by cluster_episode_goals() and the distance-based
    reward from compute_coverage_reward_labels(). Handles genuine failure
    episodes correctly (they get low reward throughout, not artificially
    labeled successful like a terminal-tail heuristic would).

    reward_mode="terminal": crude fallback, labels the tail of every episode
    as reward=1 regardless of whether the push actually succeeded. Kept only
    for quick sanity checks / A-B comparison.
    """

    def __init__(
        self, h5_path, seq_len=50, action_chunk_size=5,
        reward_mode="coverage", tail_frac=0.1,
        xy_tol=1.0, xy_norm=100.0, success_dist=0.15,
    ):
        super().__init__(h5_path, seq_len=seq_len, action_chunk_size=action_chunk_size)
        self.reward_mode = reward_mode
        self.tail_frac = tail_frac
        self.xy_norm = xy_norm
        self.success_dist = success_dist

        if reward_mode == "coverage":
            self.scene_to_goal, self.ep_to_scene_key = cluster_episode_goals(h5_path, xy_tol=xy_tol)
            # precompute episode index for each frame, and cumulative episode starts
            self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

    def _episode_index_for_frame(self, frame_idx: int) -> int:
        # episode_ends is sorted; find first episode whose end exceeds frame_idx
        ep_idx = np.searchsorted(self.episode_ends, frame_idx, side="right")
        return int(ep_idx)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        seq_len = item["image"].shape[0]
        start_idx = self.valid_start_indices[idx]

        if self.reward_mode == "coverage":
            ep_idx = self._episode_index_for_frame(start_idx)
            scene_key = self.ep_to_scene_key[ep_idx]
            goal_pose = self.scene_to_goal[scene_key]  # (3,) [x,y,theta]

            # block pose is state dims [2:5], subsampled the same way as
            # image/state in the parent class (one per action chunk)
            block_states = item["state"][:, 2:5].numpy()  # (seq_len, 3)

            dense_reward, binary_reward = compute_coverage_reward_labels(
                block_states, goal_pose,
                xy_norm=self.xy_norm, success_dist=self.success_dist,
            )
            item["reward"] = torch.from_numpy(binary_reward)
            item["reward_dense"] = torch.from_numpy(dense_reward)

        elif self.reward_mode == "terminal":
            ep_end = None
            for e in self.episode_ends:
                if start_idx < e:
                    ep_end = e
                    break
            raw_end_idx = start_idx + self.raw_seq_len
            frames_to_ep_end = max(0, ep_end - raw_end_idx)
            steps_to_ep_end = frames_to_ep_end // self.action_chunk_size

            labels = np.zeros(seq_len, dtype=np.float32)
            tail = max(3, int(round(seq_len * self.tail_frac)))
            if steps_to_ep_end < tail:
                cutoff = seq_len - (tail - steps_to_ep_end)
                cutoff = max(0, min(seq_len, cutoff))
                labels[cutoff:] = 1.0
            item["reward"] = torch.from_numpy(labels)
            item["reward_dense"] = torch.from_numpy(labels)
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        return item


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RewardHead(nn.Module):
    """Small MLP on top of frozen tokenizer latents. Single-task (PushT has
    one task), so no task embedding needed — just per-frame binary logit."""

    def __init__(self, latent_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., latent_dim) -> logits: (...,)
        return self.net(z).squeeze(-1)


def pool_latents(z: torch.Tensor) -> torch.Tensor:
    """
    z from encoder is expected as (B, T, n_latents, d_bottleneck) or similar.
    Mean-pool over the latent-token axis to get one vector per frame:
    (B, T, d_bottleneck).
    Adjust this if your Encoder returns a different shape.
    """
    if z.dim() == 4:
        return z.mean(dim=2)
    return z  # already (B, T, D)


# ----------------------------------------------------------------------------
# Distributed / misc helpers (same as train_tokenizer.py)
# ----------------------------------------------------------------------------
def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def get_wandb_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "wandb_mode", "disabled") or "disabled"
    if mode == "online" and not os.environ.get("WANDB_API_KEY") and not (Path.home() / ".netrc").exists():
        return "disabled"
    return mode


def get_runtime_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "xpu"
    return torch.device("cpu"), "cpu"


def seed_everything(seed: int):
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.set_device(local_rank)
    return ddp, rank, world_size, local_rank


def load_tokenizer_encoder(ckpt_path: str, device: torch.device) -> Encoder:
    """Loads just the encoder weights from a Tokenizer checkpoint saved by
    train_tokenizer.py (which stores {"model": full_tokenizer_state_dict, ...})."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]

    n_patches = (args["H"] // args["patch"]) * (args["W"] // args["patch"])
    d_patch = args["patch"] * args["patch"] * args["C"]

    enc = Encoder(
        patch_dim=d_patch,
        d_model=args["d_model"],
        n_latents=args["n_latents"],
        n_patches=n_patches,
        n_heads=args["n_heads"],
        depth=args["depth"],
        d_bottleneck=args["d_bottleneck"],
        dropout=0.0,  # eval mode, no dropout
        mlp_ratio=args["mlp_ratio"],
        time_every=args["time_every"],
        mae_p_min=0.0,  # no masking at inference for reward features
        mae_p_max=0.0,
        scale_pos_embeds=args.get("scale_pos_embeds", False),
    )

    full_state = ckpt["model"]
    enc_state = {k[len("encoder."):]: v for k, v in full_state.items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_state, strict=True)
    enc.to(device)
    enc.eval()
    enc.requires_grad_(False)
    return enc, args


def save_ckpt(path: Path, *, step: int, epoch: int, model, opt, scaler, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, *, model, opt, scaler) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"], strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_pr_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device, device_type = get_runtime_device()
    if device.type in {"cuda", "xpu"}:
        device = torch.device(f"{device.type}:{local_rank}") if ddp else device

    seed_everything(args.seed + rank)

    # ---- frozen tokenizer encoder ----
    encoder, tok_args = load_tokenizer_encoder(args.tokenizer_ckpt, device)
    if is_rank0():
        print(f"Loaded tokenizer encoder from {args.tokenizer_ckpt} "
              f"(d_bottleneck={tok_args['d_bottleneck']})")

    # ---- data ----
    dataset = PushTRewardDataset(
        h5_path=args.dataset,
        seq_len=args.seq_len,
        action_chunk_size=args.action_chunk_size,
        reward_mode=args.reward_mode,
        tail_frac=args.tail_frac,
        xy_tol=args.xy_tol,
        xy_norm=args.xy_norm,
        success_dist=args.success_dist,
    )
    if is_rank0() and args.reward_mode == "coverage":
        n_scenes = len(dataset.scene_to_goal)
        print(f"Clustered {len(dataset.ep_to_scene_key)} episodes into {n_scenes} goal scenes")
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True, drop_last=False,
    )

    # ---- reward head ----
    reward_head = RewardHead(
        latent_dim=tok_args["d_bottleneck"], hidden=args.hidden, dropout=args.dropout,
    ).to(device)

    if ddp:
        reward_head = torch.nn.parallel.DistributedDataParallel(
            reward_head, device_ids=[local_rank], output_device=local_rank
        )

    opt = torch.optim.AdamW(reward_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device_type in {"cuda", "xpu"}
    scaler = GradScaler(device=device_type, enabled=use_amp)

    # pos_weight from empirical class balance, sampled from a few hundred
    # windows rather than guessed — coverage-reward positive rate depends on
    # success_dist and isn't knowable in closed form.
    if is_rank0():
        n_probe = min(500, len(train_set))
        probe_idx = np.random.choice(len(train_set), n_probe, replace=False)
        pos_frac = np.mean([train_set[i]["reward"].float().mean().item() for i in probe_idx])
        pos_frac = max(pos_frac, 1e-3)
        pos_weight_val = max(1.0, (1.0 - pos_frac) / pos_frac)
        print(f"Probed positive rate={pos_frac:.4f} -> pos_weight={pos_weight_val:.2f}")
    else:
        pos_weight_val = 1.0
    pos_weight_t = torch.tensor([pos_weight_val], device=device)
    if ddp:
        dist.broadcast(pos_weight_t, src=0)
    pos_weight = pos_weight_t.squeeze(0)

    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode=get_wandb_mode(args),
            config=vars(args),
        )

    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(Path(args.resume), model=reward_head, opt=opt, scaler=scaler)
        if is_rank0():
            print(f"[rank0] Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    reward_head.train()
    t0 = time.time()

    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                if step >= args.max_steps:
                    break

                x = batch["image"].to(device, non_blocking=True)      # (B,T,C,H,W)
                r = batch["reward"].to(device, non_blocking=True)     # (B,T)

                patches = temporal_patchify(x, tok_args["patch"])

                with torch.no_grad():
                    z, _ = encoder(patches)          # (B,T,n_latents,d_bottleneck) expected
                    z = pool_latents(z)               # (B,T,d_bottleneck)

                with autocast(device_type=device_type, enabled=use_amp):
                    logits = reward_head(z)           # (B,T)
                    loss = F.binary_cross_entropy_with_logits(logits, r, pos_weight=pos_weight)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: {loss}")

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                if is_rank0() and (step % args.log_every == 0):
                    precision, recall, f1 = compute_pr_metrics(logits.detach(), r)
                    wandb.log(
                        {
                            "loss/bce": float(loss.item()),
                            "train/precision": precision,
                            "train/recall": recall,
                            "train/f1": f1,
                            "train/pos_frac": float(r.mean().item()),
                            "lr": float(opt.param_groups[0]["lr"]),
                            "time/hrs": (time.time() - t0) / 3600.0,
                        },
                        step=step,
                    )

                if is_rank0() and (step % args.print_every == 0):
                    precision, recall, f1 = compute_pr_metrics(logits.detach(), r)
                    print(f"step {step:07d} | loss={loss.item():.4f} "
                          f"| P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

                if is_rank0() and args.eval_every > 0 and (step % args.eval_every == 0) and step > 0:
                    evaluate(reward_head, encoder, val_loader, tok_args, device, device_type, use_amp, step)
                    reward_head.train()

                if is_rank0() and args.save_every > 0 and (step % args.save_every == 0) and step > 0:
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(ckpt_path, step=step, epoch=epoch, model=reward_head, opt=opt, scaler=scaler, args=args)
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(latest, step=step, epoch=epoch, model=reward_head, opt=opt, scaler=scaler, args=args)

                step += 1

            start_epoch = epoch + 1

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(reward_head, encoder, val_loader, tok_args, device, device_type, use_amp, step):
    reward_head.eval()
    all_logits, all_labels = [], []
    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        r = batch["reward"].to(device, non_blocking=True)
        patches = temporal_patchify(x, tok_args["patch"])
        z, _ = encoder(patches)
        z = pool_latents(z)
        with autocast(device_type=device_type, enabled=use_amp):
            logits = reward_head(z)
        all_logits.append(logits.float().cpu())
        all_labels.append(r.float().cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    precision, recall, f1 = compute_pr_metrics(logits, labels)
    print(f"[eval @ step {step}] P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    wandb.log({"val/precision": precision, "val/recall": recall, "val/f1": f1}, step=step)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--action_chunk_size", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_frac", type=float, default=0.05)

    # reward labeling
    p.add_argument("--reward_mode", type=str, default="coverage", choices=["coverage", "terminal"])
    p.add_argument("--tail_frac", type=float, default=0.1, help="only used by reward_mode=terminal")
    p.add_argument("--xy_tol", type=float, default=1.0, help="tolerance (px) for clustering episodes into scenes by start block pose")
    p.add_argument("--xy_norm", type=float, default=100.0, help="normalizer for xy distance-to-goal, ~matches observed block xy range")
    p.add_argument("--success_dist", type=float, default=0.15, help="normalized-distance threshold below which block is considered at goal")

    # tokenizer
    p.add_argument("--tokenizer_ckpt", type=str, required=True)

    # reward head
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=50_000)

    # logging
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=1000)

    # wandb
    p.add_argument("--wandb_project", type=str, default="dreamer4-reward")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"])

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/reward_ckpts")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)

    train(p.parse_args())

# torchrun --nproc_per_node=8 train_reward.py \
#   --dataset /data2/ws1/lagandua-MySpace/pusht_expert_train.h5 \
#   --tokenizer_ckpt ./logs/tokenizer_ckpts/latest.pt \
#   --wandb_mode online