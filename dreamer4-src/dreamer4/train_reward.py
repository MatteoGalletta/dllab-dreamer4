# train_reward.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import random
import argparse
import sys
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from torch.utils.tensorboard import SummaryWriter

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
torch.backends.cudnn.benchmark = True

IS_WINDOWS = platform.system() == "Windows"


# ----------------------------------------------------------------------------
# Reward labeling
#
# Ground truth: PushT has a SINGLE FIXED goal pose (block position + angle)
# for the whole dataset, not a per-episode/per-scene goal. Confirmed via a
# teammate's probe-training script targeting the same dataset file, which
# hardcodes objective_pos=(256,256), objective_angle=pi/4, and defines
# success as pos_err<=20px AND angle_err<=pi/9 (separate thresholds, not a
# blended distance). This replaces the earlier cluster_episode_goals()
# approach, which *estimated* a goal per start-position cluster from a
# handful of episodes each (some of which are failures) -- that estimation
# noise was a real source of label noise/false positives. The empirical
# medians found via clustering (~251,260,0.78 and ~240-247,244,~0.7) land
# close to (256,256,0.785), consistent with this being the true fixed goal.
# ----------------------------------------------------------------------------
def angle_delta(angle: np.ndarray, reference: float) -> np.ndarray:
    """Circular angle difference, wrapped to [-pi, pi]."""
    return np.arctan2(np.sin(angle - reference), np.cos(angle - reference))


def compute_objective_reward(
    block_xy: np.ndarray,
    block_angle: np.ndarray,
    objective_pos: np.ndarray,
    objective_angle: float,
    pos_tol: float,
    angle_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized reward computation against the fixed global goal.

    binary: pos_err<=pos_tol AND angle_err<=angle_tol (matches the
        "objective_met" definition used elsewhere for this dataset).
    dense: product of two clipped-linear terms (position closeness x angle
        closeness), which softens the AND logic into something differentiable
        -- both dimensions need to be reasonably good for the reward to be
        high, rather than one bad dimension being averaged away by a good one
        (the failure mode of the earlier sqrt(dx^2+dtheta^2) blended metric).

    block_xy: (N, 2), block_angle: (N,)
    Returns: (dense (N,) float32 in [0,1], binary (N,) float32 in {0,1})
    """
    pos_err = np.linalg.norm(block_xy - objective_pos[None, :], axis=-1)
    angle_err = np.abs(angle_delta(block_angle, objective_angle))

    binary = (pos_err <= pos_tol) & (angle_err <= angle_tol)

    dense_pos = np.clip(1.0 - pos_err / (2.0 * pos_tol), 0.0, 1.0)
    dense_angle = np.clip(1.0 - angle_err / (2.0 * angle_tol), 0.0, 1.0)
    dense = dense_pos * dense_angle

    return dense.astype(np.float32), binary.astype(np.float32)


class PushTRewardDataset(PushTSequenceDataset):
    """
    Wraps PushTSequenceDataset and attaches both a binary and a dense reward
    label per timestep, computed against the fixed global goal pose.

    Rewards for the ENTIRE dataset are precomputed once (vectorized) at
    init time from state[:, 2:5] (block x, y, theta) -- this is both more
    accurate (no per-scene estimation noise) and much faster than computing
    per-window at __getitem__ time (no h5 re-reads, no episode/scene lookup
    per sample).
    """

    def __init__(
        self, h5_path, seq_len=50, action_chunk_size=5,
        objective_pos=(256.0, 256.0), objective_angle=float(np.pi / 4),
        pos_tol=20.0, angle_tol=float(np.pi / 9),
    ):
        super().__init__(h5_path, seq_len=seq_len, action_chunk_size=action_chunk_size)

        self.objective_pos = np.asarray(objective_pos, dtype=np.float32)
        self.objective_angle = float(objective_angle)
        self.pos_tol = float(pos_tol)
        self.angle_tol = float(angle_tol)

        import h5py
        with h5py.File(h5_path, "r") as f:
            state_block = f["state"][:, 2:5]  # (N, 3): block x, y, theta -- full column read

        self.dense_reward_all, self.binary_reward_all = compute_objective_reward(
            state_block[:, :2], state_block[:, 2],
            self.objective_pos, self.objective_angle, self.pos_tol, self.angle_tol,
        )
        # valid_start_indices set by parent __init__; keep as numpy for fast
        # vectorized indexing (used by the balanced sampler and pos_weight
        # calc in train()).
        self.valid_start_indices = np.asarray(self.valid_start_indices, dtype=np.int64)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        start_idx = int(self.valid_start_indices[idx])
        frame_idx = start_idx + np.arange(self.seq_len) * self.action_chunk_size
        item["reward"] = torch.from_numpy(self.binary_reward_all[frame_idx])
        item["reward_dense"] = torch.from_numpy(self.dense_reward_all[frame_idx])
        return item


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RewardHead(nn.Module):
    """Small MLP on top of frozen tokenizer latents. Single-task (PushT has
    one task), so no task embedding needed -- just per-frame binary logit.
    The same logit is used for BCE (binary success), sigmoid-MSE (dense
    coverage regression), and pairwise ranking (ordering) losses."""

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
        return self.net(z).squeeze(-1)


def pool_latents(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 4:
        return z.mean(dim=2)
    return z


def pairwise_ranking_loss(
    logits: torch.Tensor,
    dense_reward: torch.Tensor,
    num_pairs: int = 512,
    margin: float = 0.1,
    min_gap: float = 0.05,
) -> torch.Tensor:
    """
    Contrastive/ranking auxiliary loss (Bradley-Terry style): supervises the
    RAW LOGIT ORDERING against the dense reward for randomly sampled pairs,
    rather than only fitting a hard binary threshold.
    """
    N = logits.shape[0]
    if N < 2:
        return logits.new_zeros(())

    idx1 = torch.randint(0, N, (num_pairs,), device=logits.device)
    idx2 = torch.randint(0, N, (num_pairs,), device=logits.device)
    gap = dense_reward[idx1] - dense_reward[idx2]
    valid = gap.abs() > min_gap
    if valid.sum() == 0:
        return logits.new_zeros(())

    target = torch.sign(gap[valid])
    loss = F.margin_ranking_loss(logits[idx1][valid], logits[idx2][valid], target, margin=margin)
    return loss


# ----------------------------------------------------------------------------
# Distributed / misc helpers
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
        backend = "gloo" if IS_WINDOWS else "nccl"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.set_device(local_rank)
    return ddp, rank, world_size, local_rank


def load_tokenizer_encoder(ckpt_path: str, device: torch.device) -> Encoder:
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
        dropout=0.0,
        mlp_ratio=args["mlp_ratio"],
        time_every=args["time_every"],
        mae_p_min=0.0,
        mae_p_max=0.0,
        scale_pos_embeds=args.get("scale_pos_embeds", False),
    )

    full_state = ckpt["model"]
    enc_state = {k[len("encoder."):]: v for k, v in full_state.items() if k.startswith("encoder.")}
    enc.load_state_dict(enc_state, strict=True)
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


def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Rank-based ROC-AUC, no sklearn dependency. Threshold-independent
    measure of ranking quality -- useful alongside P/R/F1@0.5 to tell apart
    'the model can't separate the classes' from 'the decision threshold is
    miscalibrated' (the latter is fixable for free, see select_best_threshold)."""
    y = labels.astype(np.int64)
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def select_best_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Sweeps candidate thresholds (val-set probability quantiles) and picks
    the one maximizing F1. The model's natural decision boundary often isn't
    at 0.5 given class imbalance -- this recalibrates for free at eval time,
    no retraining needed, and directly addresses 'too many false positives'
    if that's a calibration issue rather than a ranking-quality issue."""
    y = labels.astype(np.float32)
    p = probs.astype(np.float32)
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    candidates = np.concatenate([[0.5], candidates])
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        preds = (p >= t).astype(np.float32)
        tp = float(((preds == 1) & (y == 1)).sum())
        fp = float(((preds == 1) & (y == 0)).sum())
        fn = float(((preds == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device, device_type = get_runtime_device()
    if device.type in {"cuda", "xpu"}:
        device = torch.device(f"{device.type}:{local_rank}") if ddp else device

    seed_everything(args.seed + rank)

    if is_rank0() and device_type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} | {props.total_memory / 1e9:.1f} GB VRAM")
        if props.total_memory < 10e9:
            print("Detected <10GB VRAM: if you hit CUDA OOM, lower --batch_size "
                  "and/or --seq_len first.")

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
        objective_pos=(args.objective_x, args.objective_y),
        objective_angle=args.objective_angle,
        pos_tol=args.objective_pos_tol,
        angle_tol=args.objective_angle_tol,
    )
    if is_rank0():
        print(f"Fixed goal: pos=({args.objective_x:.1f}, {args.objective_y:.1f}) "
              f"angle={args.objective_angle:.3f} rad | "
              f"pos_tol={args.objective_pos_tol:.1f}px angle_tol={args.objective_angle_tol:.3f} rad")
        print(f"Global per-frame positive rate: {dataset.binary_reward_all.mean():.4f}")

    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    # Exact per-frame positive rate over the TRAIN split (vectorized, no h5/
    # image reads needed) -- used both for pos_weight and the balanced
    # sampler. Replaces the earlier approach of probing 500 random windows
    # via full __getitem__ calls (noisier estimate, and needlessly loaded
    # images just to compute a label statistic).
    train_indices = np.asarray(train_set.indices, dtype=np.int64)
    train_starts = dataset.valid_start_indices[train_indices]
    frame_offsets = np.arange(args.seq_len) * args.action_chunk_size
    frame_idx_matrix = train_starts[:, None] + frame_offsets[None, :]  # (N_train, seq_len)

    window_has_positive = dataset.binary_reward_all[frame_idx_matrix].any(axis=1)
    pos_frac_exact = float(dataset.binary_reward_all[frame_idx_matrix].mean())
    pos_frac_exact = max(pos_frac_exact, 1e-6)
    pos_weight_val = max(1.0, (1.0 - pos_frac_exact) / pos_frac_exact)

    if is_rank0():
        print(f"Train split exact positive-frame rate={pos_frac_exact:.4f} -> pos_weight={pos_weight_val:.2f}")
        print(f"{window_has_positive.mean()*100:.2f}% of training windows contain >=1 positive frame")

    pos_weight = torch.tensor(pos_weight_val, device=device)

    # ---- balanced sampler (simple data-balancing trick #1) ----
    # Oversamples windows that contain at least one success frame. This is a
    # DIFFERENT mechanism than pos_weight: pos_weight only reweights the loss
    # gradient, but doesn't change what a batch actually contains -- with
    # positives concentrated in the last few frames of each episode, many
    # batches see zero positives by pure chance (this was the root cause of
    # the noisy per-step train precision observed earlier). Oversampling
    # actually changes batch composition, giving more stable gradient signal.
    if ddp:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    elif args.balance_sampler:
        weights = np.where(window_has_positive, args.positive_oversample, 1.0).astype(np.float64)
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights), num_samples=len(train_set), replacement=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=max(0, args.num_workers // 2), pin_memory=(device_type == "cuda"), drop_last=False,
    )

    # ---- reward head ----
    reward_head = RewardHead(
        latent_dim=tok_args["d_bottleneck"], hidden=args.hidden, dropout=args.dropout,
    ).to(device)

    if ddp:
        reward_head = torch.nn.parallel.DistributedDataParallel(
            reward_head, device_ids=[local_rank] if device_type == "cuda" else None,
            output_device=local_rank if device_type == "cuda" else None,
        )

    opt = torch.optim.AdamW(reward_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device_type in {"cuda", "xpu"}
    scaler = GradScaler(device=device_type, enabled=use_amp)

    writer = None
    if is_rank0():
        from datetime import datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = os.path.join(args.ckpt_dir, "tensorboard_logs", f"run_{current_time}")
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"--> TensorBoard logging initialized at: {tb_log_dir}")
        print(f"    Run: tensorboard --logdir {os.path.join(args.ckpt_dir, 'tensorboard_logs')}")

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
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                if step >= args.max_steps:
                    break

                x = batch["image"].to(device, non_blocking=True)
                r = batch["reward"].to(device, non_blocking=True)
                r_dense = batch["reward_dense"].to(device, non_blocking=True)

                patches = temporal_patchify(x, tok_args["patch"])

                with torch.no_grad():
                    z, _ = encoder(patches)
                    z = pool_latents(z)

                with autocast(device_type=device_type, enabled=use_amp):
                    logits = reward_head(z)
                    bce = F.binary_cross_entropy_with_logits(logits, r, pos_weight=pos_weight)

                    logits_flat = logits.reshape(-1)
                    dense_flat = r_dense.reshape(-1)

                    dense_mse = (
                        F.mse_loss(torch.sigmoid(logits_flat), dense_flat)
                        if args.use_dense_aux else logits.new_zeros(())
                    )
                    rank_loss = (
                        pairwise_ranking_loss(
                            logits_flat, dense_flat,
                            num_pairs=args.ranking_pairs,
                            margin=args.ranking_margin,
                            min_gap=args.ranking_min_gap,
                        ) if args.use_ranking_loss else logits.new_zeros(())
                    )

                    loss = bce + args.dense_weight * dense_mse + args.ranking_weight * rank_loss

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: {loss}")

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                if is_rank0() and (step % args.log_every == 0):
                    precision, recall, f1 = compute_pr_metrics(logits.detach(), r)

                    writer.add_scalar("loss/total", float(loss.item()), global_step=step)
                    writer.add_scalar("loss/bce", float(bce.item()), global_step=step)
                    if args.use_dense_aux:
                        writer.add_scalar("loss/dense_mse", float(dense_mse.item()), global_step=step)
                    if args.use_ranking_loss:
                        writer.add_scalar("loss/ranking", float(rank_loss.item()), global_step=step)
                    writer.add_scalar("train/precision", precision, global_step=step)
                    writer.add_scalar("train/recall", recall, global_step=step)
                    writer.add_scalar("train/f1", f1, global_step=step)
                    writer.add_scalar("train/pos_frac", float(r.mean().item()), global_step=step)
                    writer.add_scalar("lr", float(opt.param_groups[0]["lr"]), global_step=step)
                    writer.flush()

                if is_rank0() and (step % args.print_every == 0):
                    precision, recall, f1 = compute_pr_metrics(logits.detach(), r)
                    print(f"step {step:07d} | loss={loss.item():.4f} (bce={bce.item():.4f} "
                          f"dense={float(dense_mse):.4f} rank={float(rank_loss):.4f}) "
                          f"| P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

                if is_rank0() and args.eval_every > 0 and (step % args.eval_every == 0) and step > 0:
                    evaluate(reward_head, encoder, val_loader, tok_args, device, device_type,
                             use_amp, step, pos_weight, args, writer)
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
def evaluate(reward_head, encoder, val_loader, tok_args, device, device_type, use_amp, step,
             pos_weight, args, writer=None):
    reward_head.eval()
    all_logits, all_labels, all_dense = [], [], []
    total_loss, total_count = 0.0, 0

    for i, batch in enumerate(val_loader):
        if args.max_eval_batches > 0 and i >= args.max_eval_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        r = batch["reward"].to(device, non_blocking=True)
        r_dense = batch["reward_dense"].to(device, non_blocking=True)
        patches = temporal_patchify(x, tok_args["patch"])
        z, _ = encoder(patches)
        z = pool_latents(z)
        with autocast(device_type=device_type, enabled=use_amp):
            logits = reward_head(z)
            bce = F.binary_cross_entropy_with_logits(logits, r, pos_weight=pos_weight, reduction="sum")
        total_loss += float(bce.item())
        total_count += r.numel()
        all_logits.append(logits.float().cpu())
        all_labels.append(r.float().cpu())
        all_dense.append(r_dense.float().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    dense = torch.cat(all_dense)
    probs = torch.sigmoid(logits)

    val_loss = total_loss / max(1, total_count)
    val_mse = F.mse_loss(probs, dense).item()
    precision, recall, f1 = compute_pr_metrics(logits, labels)

    auc = roc_auc(labels.numpy().reshape(-1), probs.numpy().reshape(-1))
    best_t, best_f1 = select_best_threshold(labels.numpy().reshape(-1), probs.numpy().reshape(-1))

    print(f"[eval @ step {step}] bce_loss={val_loss:.4f} dense_mse={val_mse:.4f} "
          f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
          f"roc_auc={auc:.3f} best_thresh={best_t:.3f} (f1={best_f1:.3f})")

    if is_rank0() and writer is not None:
        writer.add_scalar("val/loss", val_loss, global_step=step)
        writer.add_scalar("val/dense_mse", val_mse, global_step=step)
        writer.add_scalar("val/precision", precision, global_step=step)
        writer.add_scalar("val/recall", recall, global_step=step)
        writer.add_scalar("val/f1", f1, global_step=step)
        writer.add_scalar("val/roc_auc", auc, global_step=step)
        writer.add_scalar("val/best_threshold", best_t, global_step=step)
        writer.add_scalar("val/best_f1", best_f1, global_step=step)
        writer.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--action_chunk_size", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_frac", type=float, default=0.05)

    # reward labeling -- fixed global goal (see header comment). Defaults
    # match a teammate's probe-training script targeting the same dataset
    # file; double check these against your env config if unsure, but they
    # are consistent with the empirical goal estimates found earlier.
    p.add_argument("--objective_x", type=float, default=256.0)
    p.add_argument("--objective_y", type=float, default=256.0)
    p.add_argument("--objective_angle", type=float, default=float(np.pi / 4))
    p.add_argument("--objective_pos_tol", type=float, default=20.0, help="pixels")
    p.add_argument("--objective_angle_tol", type=float, default=float(np.pi / 9), help="radians (default 20 deg)")

    # data balancing
    p.add_argument("--balance_sampler", action="store_true", default=True)
    p.add_argument("--no_balance_sampler", dest="balance_sampler", action="store_false")
    p.add_argument("--positive_oversample", type=float, default=5.0,
                    help="sampling weight multiplier for windows containing >=1 positive frame")

    # tokenizer
    p.add_argument("--tokenizer_ckpt", type=str, required=True)

    # reward head
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # auxiliary losses (dense regression + pairwise ranking / contrastive)
    p.add_argument("--use_dense_aux", action="store_true", default=True)
    p.add_argument("--no_dense_aux", dest="use_dense_aux", action="store_false")
    p.add_argument("--dense_weight", type=float, default=0.5)
    p.add_argument("--use_ranking_loss", action="store_true", default=True)
    p.add_argument("--no_ranking_loss", dest="use_ranking_loss", action="store_false")
    p.add_argument("--ranking_weight", type=float, default=0.5)
    p.add_argument("--ranking_pairs", type=int, default=512)
    p.add_argument("--ranking_margin", type=float, default=0.1)
    p.add_argument("--ranking_min_gap", type=float, default=0.05)

    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=50_000)

    # logging
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--max_eval_batches", type=int, default=20)

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/reward_ckpts")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)

    train(p.parse_args())

# Windows / single-GPU (RTX 3070 Ti) usage -- no torchrun needed:
# python train_reward.py ^
#   --dataset C:\path\to\pusht_expert_train.h5 ^
#   --tokenizer_ckpt C:\path\to\tokenizer_ckpts\latest.pt ^
#   --batch_size 16 --num_workers 4
