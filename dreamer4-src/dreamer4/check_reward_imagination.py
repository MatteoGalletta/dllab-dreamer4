"""
test: play expert trajectories INSIDE the dynamics model
(conditioned on the real action sequence, but with imagined/generated
latents instead of real encoded frames after some context length), and see
at what frame the reward classifier flags success. Compare this to:
  (a) the "simulator baseline" -- reward head on the REAL encoded frames
  (b) the ground-truth state-based reward (gt_binary/gt_dense)

This tests whether the world model's imagined rollouts are accurate enough
that a reward signal derived from them stays trustworthy.

Usage:
    python check_reward_in_imagination.py \
        --dataset /path/to/pusht_expert_train.h5 \
        --tokenizer_ckpt ./logs/tokenizer_ckpts/latest.pt \
        --dynamics_ckpt ./logs/dynamics_ckpts/latest.pt \
        --reward_ckpt ./logs/reward_ckpts/latest.pt \
        --episodes 0 1 2 80 199 \
        --ctx_length 8

"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import importlib.util

CURRENT_DIR = Path(__file__).resolve().parent
LOCAL_DYNAMICS_PATH = CURRENT_DIR / "train_dynamics.py"

spec = importlib.util.spec_from_file_location("local_train_dynamics", LOCAL_DYNAMICS_PATH)
local_dyn_module = importlib.util.module_from_spec(spec)
sys.modules["local_train_dynamics"] = local_dyn_module
spec.loader.exec_module(local_dyn_module)

load_frozen_tokenizer_from_pt_ckpt = local_dyn_module.load_frozen_tokenizer_from_pt_ckpt
sample_autoregressive_packed_sequence = local_dyn_module.sample_autoregressive_packed_sequence
make_tau_schedule = local_dyn_module.make_tau_schedule

from model import Dynamics, pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck, temporal_patchify

from train_reward import RewardHead, pool_latents, compute_objective_reward


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_dynamics_from_ckpt(ckpt_path: str, device: torch.device, tokenizer_ckpt_override: str = None):
    """Reconstructs the Dynamics model exactly from a checkpoint saved by
    train_dynamics.py -- all hyperparams needed are in ckpt['args']."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dyn_args = ckpt["args"]

    # Use explicitly passed tokenizer checkpoint if available to avoid relative path errors
    tokenizer_path = tokenizer_ckpt_override if tokenizer_ckpt_override is not None else dyn_args["tokenizer_ckpt"]

    # Filter None override values so we don't clobber the tokenizer's own saved args with unset CLI defaults.
    override = {
        k: dyn_args[k] for k in ("H", "W", "C", "patch")
        if dyn_args.get(k) is not None
    }
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        tokenizer_path, device=device, override=override
    )

    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    packing_factor = int(dyn_args["packing_factor"])
    assert n_latents % packing_factor == 0
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor

    dyn = Dynamics(
        d_model=dyn_args["d_model_dyn"],
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=dyn_args["n_register"],
        n_agent=dyn_args["n_agent"],
        n_heads=dyn_args["n_heads"],
        depth=dyn_args["dyn_depth"],
        k_max=dyn_args["k_max"],
        dropout=0.0,
        mlp_ratio=dyn_args["mlp_ratio"],
        time_every=dyn_args["time_every"],
        space_mode=dyn_args["space_mode"],
        scale_pos_embeds=dyn_args.get("scale_pos_embeds", False),
    ).to(device)
    dyn.load_state_dict(ckpt["dynamics"], strict=True)
    dyn.eval()
    for p in dyn.parameters():
        p.requires_grad_(False)

    return dyn, dyn_args, encoder, decoder, tok_args, packing_factor


def load_episode(h5_path: str, ep_idx: int, action_chunk_size: int):
    """Reproduces PushTSequenceDataset's transform (one frame per action
    chunk, actions grouped per chunk) but over a FULL episode rather than a
    fixed-length window."""
    with h5py.File(h5_path, "r") as f:
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        start = int(ep_offset[ep_idx])
        raw_len = int(ep_len[ep_idx])
        end = start + raw_len

        images = f["pixels"][start:end]
        actions_raw = f["action"][start:end]
        states = f["state"][start:end]

    seq_len = raw_len // action_chunk_size
    used = seq_len * action_chunk_size

    images = images[:used][::action_chunk_size]  # (seq_len,H,W,3)
    states = states[:used][::action_chunk_size]  # (seq_len,7)
    actions = actions_raw[:used].reshape(seq_len, -1)  # (seq_len, action_chunk_size*2)

    frames = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0  # (seq_len,C,H,W)
    actions = torch.from_numpy(actions).float()
    return frames, actions, states


@torch.no_grad()
def run_episode_in_imagination(
        ep_idx, h5_path, dyn, dyn_args, encoder, decoder, tok_args, packing_factor,
        reward_head, device, ctx_length, schedule, eval_d,
):
    frames, actions, states = load_episode(h5_path, ep_idx, dyn_args["action_chunk_size"])
    T = frames.shape[0]
    ctx_length = min(ctx_length, T - 1)
    horizon = T - ctx_length

    frames = frames.unsqueeze(0).to(device)  # (1,T,C,H,W)
    patch = int(tok_args.get("patch", 4))

    # ---- action padding, exactly mirroring the training loop ----
    if dyn_args.get("use_actions", False):
        raw_actions = actions.unsqueeze(0).to(device).clamp(-1, 1)  # (1,T,A_raw)
        actions_padded = torch.zeros((1, T, 16), device=device, dtype=torch.float32)
        actions_padded[..., : raw_actions.shape[-1]] = raw_actions
        act_mask = torch.zeros(16, device=device, dtype=torch.float32)
        act_mask[: raw_actions.shape[-1]] = 1.0
    else:
        actions_padded = None
        act_mask = None

    # ---- encode the FULL real sequence once (ground truth latents) ----
    patches = temporal_patchify(frames, patch)
    z_btLd, _ = encoder(patches)  # (1,T,n_latents,d_bottleneck)  -- also our "real" latents for baseline
    n_spatial = z_btLd.shape[2] // packing_factor
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)

    # ---- imagine forward from ctx_length using the REAL action sequence ----
    sched = make_tau_schedule(k_max=dyn_args["k_max"], schedule=schedule, d=eval_d)
    z_imagined_packed = sample_autoregressive_packed_sequence(
        dyn,
        z_gt_packed=z_gt_packed,
        ctx_length=ctx_length,
        horizon=horizon,
        k_max=dyn_args["k_max"],
        sched=sched,
        actions=actions_padded,
        act_mask=act_mask,
    )  # (1,T,Sz,Dz) -- context frames copied verbatim, rest imagined

    # ---- reward head on both real and imagined latents ----
    z_real_unpacked = unpack_spatial_to_bottleneck(z_gt_packed, k=packing_factor)
    z_imagined_unpacked = unpack_spatial_to_bottleneck(z_imagined_packed, k=packing_factor)

    pred_real = torch.sigmoid(reward_head(pool_latents(z_real_unpacked))).squeeze(0).cpu().numpy()
    pred_imagined = torch.sigmoid(reward_head(pool_latents(z_imagined_unpacked))).squeeze(0).cpu().numpy()

    return pred_real, pred_imagined, states, ctx_length


def episode_count(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        return len(f["ep_offset"])


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)


def run_aggregate(
        h5_path, dyn, dyn_args, encoder, decoder, tok_args, packing_factor,
        reward_head, device, ctx_length, schedule, eval_d, threshold,
        objective_pos, objective_angle, objective_pos_tol, objective_angle_tol,
        episode_indices,
):
    imag_success_flags, real_success_flags, gt_success_flags = [], [], []
    offsets_matched = []  # imagined-vs-gt frame offset, only when both cross
    horizons = []  # imagined horizon length per episode (n - ctx_len)
    skipped = 0

    for i, ep_idx in enumerate(episode_indices):
        try:
            pred_real, pred_imagined, states, ctx_len = run_episode_in_imagination(
                ep_idx, h5_path, dyn, dyn_args, encoder, decoder, tok_args, packing_factor,
                reward_head, device, ctx_length, schedule, eval_d,
            )
        except Exception as e:
            skipped += 1
            print(f"  [skip] episode {ep_idx}: {e}")
            continue

        n = min(len(pred_real), len(pred_imagined), len(states))
        block_states = states[:n, 2:5]
        gt_dense, gt_binary = compute_objective_reward(
            block_states[:, :2], block_states[:, 2],
            objective_pos, objective_angle, objective_pos_tol, objective_angle_tol,
        )

        imag_success = float(np.any(pred_imagined[:n] >= threshold))
        real_success = float(np.any(pred_real[:n] >= threshold))
        gt_success = float(np.any(gt_binary))

        imag_success_flags.append(imag_success)
        real_success_flags.append(real_success)
        gt_success_flags.append(gt_success)
        horizons.append(max(0, n - ctx_len))

        if imag_success and gt_success:
            imag_idx = int(np.where(pred_imagined[:n] >= threshold)[0][0])
            gt_idx = int(np.where(gt_binary >= 0.5)[0][0])
            offsets_matched.append(imag_idx - gt_idx)

        if (i + 1) % 20 == 0:
            print(f"  ...processed {i + 1}/{len(episode_indices)} episodes")

    imag_success_flags = np.array(imag_success_flags)
    real_success_flags = np.array(real_success_flags)
    gt_success_flags = np.array(gt_success_flags)
    horizons = np.array(horizons)

    n_done = len(gt_success_flags)
    print(f"\n=== Aggregate results over {n_done} episodes (skipped {skipped}) ===")
    print(f"Ground-truth positive rate (episodes that ever succeed): {gt_success_flags.mean():.3f}")

    print("\n-- REAL-frame baseline (reward head on real encoded frames) vs ground truth --")
    real_cm = confusion_counts(real_success_flags, gt_success_flags)
    print(f"  TP={real_cm['tp']} FP={real_cm['fp']} FN={real_cm['fn']} TN={real_cm['tn']} | "
          f"P={real_cm['precision']:.3f} R={real_cm['recall']:.3f} F1={real_cm['f1']:.3f}")

    print("\n-- IMAGINED rollout (dynamics model + reward head) vs ground truth --")
    imag_cm = confusion_counts(imag_success_flags, gt_success_flags)
    print(f"  TP={imag_cm['tp']} FP={imag_cm['fp']} FN={imag_cm['fn']} TN={imag_cm['tn']} | "
          f"P={imag_cm['precision']:.3f} R={imag_cm['recall']:.3f} F1={imag_cm['f1']:.3f}")
    print(f"  -> False positives (imagination hallucinated success): {imag_cm['fp']}/{n_done}")
    print(f"  -> False negatives (imagination missed a real success): {imag_cm['fn']}/{n_done}")

    # ---- horizon-stratified breakdown ----
    gt_pos_mask = gt_success_flags == 1
    if gt_pos_mask.sum() > 0:
        print("\n-- Recall (on imagination) stratified by imagined horizon length --")
        print("   (only episodes where gt_success=1 are included -- this is recall, not full P/R/F1)")
        h_pos = horizons[gt_pos_mask]
        imag_pos = imag_success_flags[gt_pos_mask]

        n_buckets = 4
        edges = np.quantile(h_pos, np.linspace(0, 1, n_buckets + 1))
        edges[-1] += 1e-6  # ensure the max value falls in the last bucket
        edges = np.unique(edges)  # guard against duplicate edges

        print(f"   {'horizon range':<18}{'n_episodes':>12}{'recall':>10}")
        for b in range(len(edges) - 1):
            lo, hi = edges[b], edges[b + 1]
            mask = (h_pos >= lo) & (h_pos < hi)
            n_bucket = int(mask.sum())
            if n_bucket == 0:
                continue
            bucket_recall = float(imag_pos[mask].mean())
            print(f"   [{lo:.0f}, {hi:.0f}){'':<8}{n_bucket:>12}{bucket_recall:>10.3f}")

        # simple correlation check: horizon length vs. miss (1=missed)
        missed = 1.0 - imag_pos
        if h_pos.std() > 0 and missed.std() > 0:
            corr = float(np.corrcoef(h_pos, missed)[0, 1])
            print(f"\n   correlation(horizon length, miss): {corr:+.3f} "
                  f"(positive => longer horizon, more misses => consistent with rollout drift)")

    if len(offsets_matched) > 0:
        offsets_matched = np.array(offsets_matched)
        print(f"\n-- Timing, for the {len(offsets_matched)} episodes where imagination correctly detects success --")
        print(f"  mean offset (imagined - gt): {offsets_matched.mean():+.2f} frames | "
              f"std: {offsets_matched.std():.2f} | "
              f"range: [{offsets_matched.min():+d}, {offsets_matched.max():+d}]")

    return imag_cm, real_cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--tokenizer_ckpt", type=str, required=True,
                   help="Tokenizer checkpoint path (now actively used to override broken relative paths in dynamics ckpt)")
    p.add_argument("--dynamics_ckpt", type=str, required=True)
    p.add_argument("--reward_ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 80, 199])
    p.add_argument("--ctx_length", type=int, default=8,
                   help="number of real (encoded) context frames before imagination takes over")
    p.add_argument("--schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.25)
    p.add_argument("--objective_x", type=float, default=256.0)
    p.add_argument("--objective_y", type=float, default=256.0)
    p.add_argument("--objective_angle", type=float, default=float(np.pi / 4))
    p.add_argument("--objective_pos_tol", type=float, default=20.0)
    p.add_argument("--objective_angle_tol", type=float, default=float(np.pi / 9))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--aggregate", action="store_true",
                   help="run over many episodes and report a confusion matrix instead of per-episode detail")
    p.add_argument("--num_episodes", type=int, default=100,
                   help="number of random episodes to sample when --aggregate is set (ignored if --episodes is explicitly meaningful and --aggregate not set)")
    p.add_argument("--random_seed", type=int, default=0)
    args = p.parse_args()

    device = get_device()

    # Pass the CLI's tokenizer_ckpt explicitly to override relative metadata paths
    dyn, dyn_args, encoder, decoder, tok_args, packing_factor = load_dynamics_from_ckpt(
        args.dynamics_ckpt, device, tokenizer_ckpt_override=args.tokenizer_ckpt
    )
    print(f"Loaded dynamics model from {args.dynamics_ckpt} "
          f"(k_max={dyn_args['k_max']}, action_chunk_size={dyn_args['action_chunk_size']}, "
          f"use_actions={dyn_args.get('use_actions', False)})")

    reward_ckpt = torch.load(args.reward_ckpt, map_location="cpu")
    reward_head = RewardHead(latent_dim=tok_args["d_bottleneck"], hidden=256).to(device)
    reward_head.load_state_dict(reward_ckpt["model"])
    reward_head.eval()

    objective_pos = np.array([args.objective_x, args.objective_y], dtype=np.float32)

    if args.aggregate:
        rng = np.random.default_rng(args.random_seed)
        n_total = episode_count(args.dataset)
        n_sample = min(args.num_episodes, n_total)
        episode_indices = rng.choice(n_total, size=n_sample, replace=False)
        print(f"Sampling {n_sample} random episodes out of {n_total} total (seed={args.random_seed})")

        run_aggregate(
            args.dataset, dyn, dyn_args, encoder, decoder, tok_args, packing_factor,
            reward_head, device, args.ctx_length, args.schedule, args.eval_d, args.threshold,
            objective_pos, args.objective_angle, args.objective_pos_tol, args.objective_angle_tol,
            episode_indices,
        )
        return

    for ep_idx in args.episodes:
        pred_real, pred_imagined, states, ctx_length = run_episode_in_imagination(
            ep_idx, args.dataset, dyn, dyn_args, encoder, decoder, tok_args, packing_factor,
            reward_head, device, args.ctx_length, args.schedule, args.eval_d,
        )

        n = min(len(pred_real), len(pred_imagined), len(states))
        block_states = states[:n, 2:5]
        gt_dense, gt_binary = compute_objective_reward(
            block_states[:, :2], block_states[:, 2],
            objective_pos, args.objective_angle, args.objective_pos_tol, args.objective_angle_tol,
        )

        print(f"\n=== Episode {ep_idx} (len={n}, ctx_length={ctx_length}) ===")
        print(f"{'frame':<8}{'pred_real':>12}{'pred_imag':>12}{'gt_dense':>12}{'gt_binary':>10}{'in_ctx':>8}")
        checkpoints = sorted(set(list(range(0, n, max(1, n // 12))) + list(range(max(0, n - 6), n))))
        for t in checkpoints:
            in_ctx = "yes" if t < ctx_length else ""
            print(
                f"{t:<8}{pred_real[t]:>12.4f}{pred_imagined[t]:>12.4f}{gt_dense[t]:>12.4f}{gt_binary[t]:>10.0f}{in_ctx:>8}")

        # first frame (after context) each signal crosses threshold, if any
        def first_crossing(arr, thresh, start=0):
            idxs = np.where(arr[start:] >= thresh)[0]
            return int(idxs[0] + start) if len(idxs) > 0 else None

        real_cross = first_crossing(pred_real, args.threshold)
        imag_cross = first_crossing(pred_imagined, args.threshold)
        gt_cross = first_crossing(gt_binary, 0.5)

        print(f"  first frame >= {args.threshold}: real={real_cross}  imagined={imag_cross}  gt_binary={gt_cross}")
        if imag_cross is not None and gt_cross is not None:
            print(f"  imagined vs gt offset: {imag_cross - gt_cross:+d} frames")
        if imag_cross is not None and real_cross is not None:
            print(f"  imagined vs real-baseline offset: {imag_cross - real_cross:+d} frames")


if __name__ == "__main__":
    main()