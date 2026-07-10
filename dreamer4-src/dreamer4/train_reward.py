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
from torch.utils.data import DataLoader, DistributedSampler

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
# Fixed input shapes every step (same seq_len/H/W) -> benchmark mode picks
# faster cudnn kernels after a short warmup. Meaningful win on consumer GPUs.
torch.backends.cudnn.benchmark = True

IS_WINDOWS = platform.system() == "Windows"


# ----------------------------------------------------------------------------
# Reward labeling
# ----------------------------------------------------------------------------
def cluster_episode_goals(h5_path: str, xy_tol: float = 10.0) -> tuple[dict, list]:
    """
    Estimates a goal pose (block x, y, theta) per "scene" (a cluster of
    episodes sharing an identical/near-identical start block position -- this
    dataset resets to a small number of fixed start configs, each implying a
    fixed goal). Goal is estimated as the *median* final block pose across
    all episodes in a scene cluster, which is robust to individual failed
    episodes (confirmed present in this dataset -- e.g. incomplete pushes)
    pulling the estimate off.

    Reads only the 7 scalar values needed per episode (start/final block
    pose), not full state trajectories, to keep memory low.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        n_eps = len(ep_offset)

        starts = np.zeros((n_eps, 2), dtype=np.float32)
        finals = np.zeros((n_eps, 3), dtype=np.float32)

        state_ds = f["state"]
        for i in range(n_eps):
            s = int(ep_offset[i])
            e = s + int(ep_len[i])
            starts[i] = state_ds[s, 2:4]
            finals[i] = state_ds[e - 1, 2:5]

    keys = [tuple(np.round(starts[i] / xy_tol).astype(int)) for i in range(n_eps)]

    scene_to_finals = {}
    for i, k in enumerate(keys):
        scene_to_finals.setdefault(k, []).append(finals[i])

    scene_to_goal = {
        k: np.median(np.array(v, dtype=np.float32), axis=0)
        for k, v in scene_to_finals.items()
    }

    del starts, finals
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

    dist_ = np.sqrt((dxy ** 2).sum(axis=-1) + dtheta ** 2)
    dense_reward = np.clip(1.0 - dist_, 0.0, 1.0).astype(np.float32)
    binary_reward = (dist_ < success_dist).astype(np.float32)
    return dense_reward, binary_reward


class PushTRewardDataset(PushTSequenceDataset):
    """
    Wraps PushTSequenceDataset and attaches both a binary and a dense reward
    label per timestep, using the goal pose estimated per-scene by
    cluster_episode_goals(). Handles genuine failure episodes correctly
    (they get low reward throughout, confirmed via manual inspection of
    episodes 80 and 199 in this dataset).
    """

    def __init__(
        self, h5_path, seq_len=50, action_chunk_size=5,
        xy_tol=10.0, xy_norm=100.0, success_dist=0.15,
    ):
        super().__init__(h5_path, seq_len=seq_len, action_chunk_size=action_chunk_size)
        self.xy_norm = xy_norm
        self.success_dist = success_dist
        self.scene_to_goal, self.ep_to_scene_key = cluster_episode_goals(h5_path, xy_tol=xy_tol)
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

    def _episode_index_for_frame(self, frame_idx: int) -> int:
        ep_idx = np.searchsorted(self.episode_ends, frame_idx, side="right")
        return int(ep_idx)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        start_idx = self.valid_start_indices[idx]

        ep_idx = self._episode_index_for_frame(start_idx)
        scene_key = self.ep_to_scene_key[ep_idx]
        goal_pose = self.scene_to_goal[scene_key]  # (3,) [x,y,theta]

        block_states = item["state"][:, 2:5].numpy()  # (seq_len, 3)

        dense_reward, binary_reward = compute_coverage_reward_labels(
            block_states, goal_pose,
            xy_norm=self.xy_norm, success_dist=self.success_dist,
        )
        item["reward"] = torch.from_numpy(binary_reward)
        item["reward_dense"] = torch.from_numpy(dense_reward)
        return item


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RewardHead(nn.Module):
    """Small MLP on top of frozen tokenizer latents. Single-task (PushT has
    one task), so no task embedding needed -- just per-frame binary logit.
    The same logit is used for BCE (binary success), sigmoid-MSE (dense
    coverage regression), and pairwise ranking (ordering) losses -- no need
    for separate heads since all three targets live on [0, 1]."""

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
    """
    z from encoder is expected as (B, T, n_latents, d_bottleneck) or similar.
    Mean-pool over the latent-token axis to get one vector per frame:
    (B, T, d_bottleneck). Adjust if your Encoder returns a different shape.
    """
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
    Contrastive/ranking auxiliary loss (Bradley-Terry style, as used to train
    RLHF reward models): rather than only fitting a hard binary threshold,
    this supervises the RAW LOGIT ORDERING against the dense coverage-to-goal
    reward for randomly sampled pairs of frames. This directly targets the
    failure mode observed during training -- noisy precision caused by
    frames right around the ambiguous success boundary (e.g. dense=0.59 but
    still labeled 0) -- because it provides a training signal everywhere
    along the trajectory, not just at the binary cutoff.

    logits: (N,) flattened batch*time predicted logits
    dense_reward: (N,) flattened ground-truth dense reward in [0, 1]
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
        # NCCL is not available on Windows at all -- fall back to gloo there.
        # On a single 3070 Ti this path shouldn't trigger (world_size=1 for
        # plain `python train_reward.py`), but it's a landmine if torchrun is
        # ever used by habit, so guard it defensively.
        backend = "gloo" if IS_WINDOWS else "nccl"
        dist.init_process_group(backend=backend, init_method="env://")
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
                  "and/or --seq_len first (encoder is frozen, so no gradient "
                  "memory there -- OOM most likely comes from activation memory "
                  "in the frozen encoder's forward pass at large batch*seq_len).")

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
        xy_tol=args.xy_tol,
        xy_norm=args.xy_norm,
        success_dist=args.success_dist,
    )
    if is_rank0():
        n_scenes = len(dataset.scene_to_goal)
        print(f"Clustered {len(dataset.ep_to_scene_key)} episodes into {n_scenes} goal scenes")

    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    # On Windows, multiprocessing workers use 'spawn' (not 'fork'), which is
    # slower to start and re-imports the module per worker. num_workers=8 from
    # a Linux-tuned default can actually be SLOWER here than a lower count.
    # If the DataLoader appears to hang on startup, try --num_workers 0 first
    # to isolate whether it's a worker-spawn issue vs something else.
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
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

    # pos_weight from empirical class balance, sampled from a few hundred
    # windows rather than guessed.
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
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                if step >= args.max_steps:
                    break

                x = batch["image"].to(device, non_blocking=True)          # (B,T,C,H,W)
                r = batch["reward"].to(device, non_blocking=True)         # (B,T)
                r_dense = batch["reward_dense"].to(device, non_blocking=True)  # (B,T)

                patches = temporal_patchify(x, tok_args["patch"])

                with torch.no_grad():
                    z, _ = encoder(patches)
                    z = pool_latents(z)

                with autocast(device_type=device_type, enabled=use_amp):
                    logits = reward_head(z)  # (B,T)
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

    val_loss = total_loss / max(1, total_count)
    val_mse = F.mse_loss(torch.sigmoid(logits), dense).item()
    precision, recall, f1 = compute_pr_metrics(logits, labels)

    print(f"[eval @ step {step}] bce_loss={val_loss:.4f} dense_mse={val_mse:.4f} "
          f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    if is_rank0() and writer is not None:
        writer.add_scalar("val/loss", val_loss, global_step=step)
        writer.add_scalar("val/dense_mse", val_mse, global_step=step)
        writer.add_scalar("val/precision", precision, global_step=step)
        writer.add_scalar("val/recall", recall, global_step=step)
        writer.add_scalar("val/f1", f1, global_step=step)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--action_chunk_size", type=int, default=5)
    # Windows spawns workers slower than Linux forks; 4 is a safer default
    # than 8 for a single consumer machine. Set to 0 first if the DataLoader
    # seems to hang on startup, to isolate the issue.
    p.add_argument("--num_workers", type=int, default=4)
    # 3070 Ti has 8GB VRAM. The reward head itself is tiny; memory pressure
    # comes from activations in the frozen encoder's forward pass at
    # batch_size * seq_len frames per step. 16 is a safer starting point than
    # 32 -- raise it only if nvidia-smi shows comfortable headroom.
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_frac", type=float, default=0.05)

    # reward labeling
    p.add_argument("--xy_tol", type=float, default=10.0, help="tolerance (px) for clustering episodes into scenes by start block pose")
    p.add_argument("--xy_norm", type=float, default=100.0, help="normalizer for xy distance-to-goal, ~matches observed block xy range")
    p.add_argument("--success_dist", type=float, default=0.15, help="normalized-distance threshold below which block is considered at goal")

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
    p.add_argument("--ranking_pairs", type=int, default=512, help="random pairs sampled per batch for the ranking loss")
    p.add_argument("--ranking_margin", type=float, default=0.1)
    p.add_argument("--ranking_min_gap", type=float, default=0.05, help="minimum dense-reward gap between a pair to count as a valid ranking signal")

    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=50_000)

    # logging
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--max_eval_batches", type=int, default=20,
                    help="cap on val batches per eval call (0 = full val set). "
                         "20 batches * batch_size frames is plenty for a stable "
                         "P/R/F1/loss estimate without paying for the full val set every time.")

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