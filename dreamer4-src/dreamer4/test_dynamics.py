import os

# 1. CRITICAL: This MUST be the absolute first line to prevent the OpenMP DLL crash!
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import importlib.util
from pathlib import Path

# 2. Get the exact folder where this test_dynamics.py script lives
SCRIPT_DIR = Path(__file__).resolve().parent  # .../dreamer4-src/dreamer4
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # .../dllab-dreamer4

# Add the project root to sys.path so 'data_pipeline' can be found
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 3. FORCE Python to load the sibling 'train_dynamics.py' directly by its absolute path
real_train_dynamics_path = SCRIPT_DIR / "train_dynamics.py"
if not real_train_dynamics_path.exists():
    raise FileNotFoundError(f"Could not find the real train_dynamics.py at {real_train_dynamics_path}")

spec = importlib.util.spec_from_file_location("local_train_dynamics", str(real_train_dynamics_path))
train_dynamics = importlib.util.module_from_spec(spec)
sys.modules["local_train_dynamics"] = train_dynamics
spec.loader.exec_module(train_dynamics)

# 4. Extract the functions directly from our hard-loaded module
load_frozen_tokenizer_from_pt_ckpt = train_dynamics.load_frozen_tokenizer_from_pt_ckpt
sample_autoregressive_packed_sequence = train_dynamics.sample_autoregressive_packed_sequence
decode_packed_to_frames = train_dynamics.decode_packed_to_frames
make_tau_schedule = train_dynamics.make_tau_schedule
get_runtime_device = train_dynamics.get_runtime_device

# 5. Now import the rest of your packages safely
import torch
import torch.nn.functional as F
import imageio
import argparse
import numpy as np
from torch.utils.data import DataLoader

from data_pipeline.PushTDataLoader import PushTSequenceDataset
from model import (
    temporal_patchify, pack_bottleneck_to_spatial, Dynamics
)


@torch.no_grad()
def main(args):
    device, _ = get_runtime_device()

    # Same Dataloading as Training
    print("Loading dataset...")
    dataset = PushTSequenceDataset(
        h5_path=args.dataset,
        seq_len=args.seq_len,
        action_chunk_size=args.action_chunk_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle to get random testing slices
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    batch = next(iter(loader))
    frames = batch["image"].to(device)
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    elif float(frames.max().item()) > 1.5:
        frames = frames.float() / 255.0

    raw_actions = batch["action"].to(device).clamp(-1, 1)
    if raw_actions.dim() == 2:
        raw_actions = raw_actions.unsqueeze(1).expand(-1, frames.shape[1], -1)

    actions = torch.zeros((*raw_actions.shape[:-1], 16), device=device, dtype=torch.float32)
    actions[..., : raw_actions.shape[-1]] = raw_actions.float()

    act_mask = torch.zeros(16, device=device, dtype=torch.float32)
    act_mask[: raw_actions.shape[-1]] = 1.0

    # Load Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_ckpt}...")
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(args.tokenizer_ckpt, device=device)
    encoder.eval()
    decoder.eval()


    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))

    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    # Load Dynamics Model
    print(f"Loading dynamics model from {args.dynamics_ckpt}...")
    dyn = Dynamics(
        d_model=args.d_model_dyn,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=args.n_register,
        n_agent=args.n_agent,
        n_heads=args.n_heads,
        depth=args.dyn_depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        space_mode=args.space_mode,
        scale_pos_embeds=args.scale_pos_embeds,
    ).to(device)

    ckpt = torch.load(args.dynamics_ckpt, map_location="cpu")
    dyn.load_state_dict(ckpt["dynamics"], strict=True)
    dyn.eval()

    # Prepare Evaluation Tensors
    B, T = frames.shape[:2]
    ctx_length = min(args.eval_ctx, T - 1)
    horizon = min(args.eval_horizon, T - ctx_length)
    T_eval = ctx_length + horizon

    frames_eval = frames[:, :T_eval]
    actions_eval = actions[:, :T_eval]

    print(f"Evaluating with Context={ctx_length}, Horizon={horizon}, Total Steps={T_eval}")

    # Extract Ground Truth Z
    patches = temporal_patchify(frames_eval, patch)
    z_btLd, _ = encoder(patches)
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=args.packing_factor)

    # Predict Z via Dynamics Rollout
    sched = make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    z_pred_packed = sample_autoregressive_packed_sequence(
        dyn,
        z_gt_packed=z_gt_packed,
        ctx_length=ctx_length,
        horizon=horizon,
        k_max=args.k_max,
        sched=sched,
        actions=actions_eval,
        act_mask=act_mask,
    )

    # --- QUANTITATIVE METRICS (Latent Z & Decoded Video Frame Accuracy) ---
    z_gt_h = z_gt_packed[:, ctx_length:ctx_length + horizon]
    z_pred_h = z_pred_packed[:, ctx_length:ctx_length + horizon]

    # 1. Latent Z Space Metrics
    z_gt_float = z_gt_h.float()
    z_pred_float = z_pred_h.float()
    mse_z_per_t = (z_pred_float - z_gt_float).pow(2).mean(dim=(0, 2, 3))
    var_z_per_t = z_gt_float.var(dim=(0, 2, 3), unbiased=False).clamp_min(1e-8)
    r2_z_per_t = torch.clamp((1.0 - (mse_z_per_t / var_z_per_t)) * 100.0, min=0.0)

    cos_sim_per_t = F.cosine_similarity(z_pred_float, z_gt_float, dim=-1).mean(dim=(0, 2))
    cos_sim_pct_per_t = torch.clamp(cos_sim_per_t * 100.0, min=0.0)

    # 2. Decode Frames for Frame-Space Accuracy & Qualitative Visualization
    print("\nDecoding frames for quantitative & qualitative evaluation...")
    pred_frames = decode_packed_to_frames(
        decoder,
        z_packed=z_pred_packed,
        H=H, W=W, C=C, patch=patch,
        packing_factor=args.packing_factor,
    )

    if horizon > 0:
        t_step1 = 0
        t_step15 = min(14, horizon - 1)
        t_end = horizon - 1

        # Frame Space Metrics over imagination horizon
        gt_frames_h = frames_eval[:, ctx_length:ctx_length + horizon].float()
        pred_frames_h = pred_frames[:, ctx_length:ctx_length + horizon].float()

        frame_mse = F.mse_loss(pred_frames_h, gt_frames_h)
        frame_var = torch.var(gt_frames_h, unbiased=False).clamp_min(1e-8)
        frame_r2_acc_pct = max(0.0, (1.0 - (frame_mse / frame_var)).item()) * 100.0
        frame_psnr = 10.0 * torch.log10(1.0 / frame_mse.clamp_min(1e-10)).item()

        # Overall Frame Pixel Accuracy
        frame_diff = torch.abs(pred_frames_h - gt_frames_h)
        frame_pacc_5pct = (frame_diff <= 0.05).float().mean().item() * 100.0
        frame_pacc_2pct = (frame_diff <= 0.02).float().mean().item() * 100.0

        # Per-timestep Frame Space Breakdown
        frame_mse_per_t = (pred_frames_h - gt_frames_h).pow(2).mean(dim=(0, 2, 3, 4))
        frame_var_per_t = gt_frames_h.var(dim=(0, 2, 3, 4), unbiased=False).clamp_min(1e-8)
        frame_r2_per_t = torch.clamp((1.0 - (frame_mse_per_t / frame_var_per_t)) * 100.0, min=0.0)

        frame_pacc_5pct_per_t = (frame_diff <= 0.05).float().mean(dim=(0, 2, 3, 4)) * 100.0
        frame_pacc_2pct_per_t = (frame_diff <= 0.02).float().mean(dim=(0, 2, 3, 4)) * 100.0

        print("\n=================== Quantitative Dynamics Evaluation ===================")
        print("--- Latent Space Z Accuracy ---")
        print(f"Overall Horizon Latent MSE:            {mse_z_per_t.mean().item():.6f}")
        print(f"Overall Horizon Latent R2 Accuracy:     {r2_z_per_t.mean().item():.2f}%")
        print(f"Overall Horizon Latent Cosine Sim:      {cos_sim_pct_per_t.mean().item():.2f}%")
        print(f"  - Step 1  (t_future=1)  Latent R2:     {r2_z_per_t[t_step1].item():.2f}% (MSE: {mse_z_per_t[t_step1].item():.6f})")
        print(f"  - Step 15 (t_future=15) Latent R2:     {r2_z_per_t[t_step15].item():.2f}% (MSE: {mse_z_per_t[t_step15].item():.6f})")
        print(f"  - Step {horizon:02d} (t_future={horizon}) Latent R2:     {r2_z_per_t[t_end].item():.2f}% (MSE: {mse_z_per_t[t_end].item():.6f})")

        print("\n--- Decoded Imagined Video Frame Accuracy ---")
        print(f"Overall Decoded Frame MSE:             {frame_mse.item():.6f}")
        print(f"Overall Decoded Frame PSNR:            {frame_psnr:.2f} dB")
        print(f"Overall Decoded Frame R2 Accuracy:     {frame_r2_acc_pct:.2f}%")
        print(f"Overall Decoded Frame Pixel Acc (tol=0.05): {frame_pacc_5pct:.2f}%")
        print(f"Overall Decoded Frame Pixel Acc (tol=0.02): {frame_pacc_2pct:.2f}%")
        print("\n--- Decoded Frame Timestep Breakdown ---")
        print(f"  - Step 1  (t_future=1):  Pixel Acc (tol=0.05)={frame_pacc_5pct_per_t[t_step1].item():.2f}%, Pixel Acc (tol=0.02)={frame_pacc_2pct_per_t[t_step1].item():.2f}%, Frame R2={frame_r2_per_t[t_step1].item():.2f}%")
        print(f"  - Step 15 (t_future=15): Pixel Acc (tol=0.05)={frame_pacc_5pct_per_t[t_step15].item():.2f}%, Pixel Acc (tol=0.02)={frame_pacc_2pct_per_t[t_step15].item():.2f}%, Frame R2={frame_r2_per_t[t_step15].item():.2f}%")
        print(f"  - Step {horizon:02d} (t_future={horizon}): Pixel Acc (tol=0.05)={frame_pacc_5pct_per_t[t_end].item():.2f}%, Pixel Acc (tol=0.02)={frame_pacc_2pct_per_t[t_end].item():.2f}%, Frame R2={frame_r2_per_t[t_end].item():.2f}%")
        print("=======================================================================\n")

        b_idx = 0  # Process first item in batch
        gt_video = frames_eval[b_idx].permute(0, 2, 3, 1).cpu().numpy()
        pred_video = pred_frames[b_idx].permute(0, 2, 3, 1).cpu().numpy()

        gt_video = (np.clip(gt_video, 0, 1) * 255).astype(np.uint8)
        pred_video = (np.clip(pred_video, 0, 1) * 255).astype(np.uint8)

        side_by_side = np.concatenate([gt_video, pred_video], axis=2)

        # Save the original GIF
        out_path = Path("dynamics_eval.gif")
        imageio.mimsave(out_path, side_by_side, fps=10, loop=0)
        print(f"Qualitative GIF saved to: {out_path.absolute()}")

        # Save individual frames to let us inspect the step-by-step timeline
        frames_dir = Path("eval_frames")
        frames_dir.mkdir(exist_ok=True)

        # Clear out old frames from previous runs so they don't mix up
        for old_file in frames_dir.glob("*.png"):
            old_file.unlink()

        for t_step in range(T_eval):
            # Determine if this step was real history or model's imagination
            phase_label = "REAL_CONTEXT" if t_step < ctx_length else f"IMAGINATION_step_{t_step - ctx_length}"
            frame_filename = frames_dir / f"frame_{t_step:02d}_{phase_label}.png"
            imageio.imwrite(frame_filename, side_by_side[t_step])

        print(f"\nSaved {T_eval} individual frames to: {frames_dir.absolute()}/")
        print("Go open that folder and inspect where the glitches start!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--action_chunk_size", type=int, default=5)

    # Checkpoints
    p.add_argument("--tokenizer_ckpt", type=str, required=True)
    p.add_argument("--dynamics_ckpt", type=str, required=True)

    # Dynamics Architecture Details
    p.add_argument("--d_model_dyn", type=int, default=1024)
    p.add_argument("--dyn_depth", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--n_register", type=int, default=4)
    p.add_argument("--n_agent", type=int, default=1)
    p.add_argument("--space_mode", type=str, default="wm_agent_isolated")
    p.add_argument("--scale_pos_embeds", action="store_true")

    # Eval settings
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--eval_ctx", type=int, default=8)
    p.add_argument("--eval_horizon", type=int, default=16)
    p.add_argument("--eval_schedule", type=str, default="shortcut")
    p.add_argument("--eval_d", type=float, default=0.25)

    main(p.parse_args())