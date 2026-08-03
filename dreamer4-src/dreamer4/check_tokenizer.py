import os
from pathlib import Path
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Dynamically add script folder (dreamer4) and repo root (dllab-dreamer4) to sys.path
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parents[1]

for path in (script_dir, repo_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# --- NOW your existing imports will work seamlessly ---
from data_pipeline.PushTDataLoader import PushTSequenceDataset
from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
    recon_loss_from_mae, lpips_on_mae_recon
)

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Imports from your project structure
from data_pipeline.PushTDataLoader import PushTSequenceDataset
from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
    recon_loss_from_mae, lpips_on_mae_recon
)

try:
    import lpips
except ImportError:
    lpips = None


class TokenizerEvaluator:
    def __init__(self, model, dataloader, device, patch_size=16, lpips_net="alex"):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.patch_size = patch_size
        
        self.model.eval()
        self.model.to(self.device)
        
        self.lpips_fn = None
        if lpips is not None:
            self.lpips_fn = lpips.LPIPS(net=lpips_net).to(self.device)
            self.lpips_fn.eval()
            self.lpips_fn.requires_grad_(False)

    @torch.no_grad()
    def evaluate(self, num_batches=None, eval_mae_masking=True):
        """
        Evaluates reconstruction quality.
        - eval_mae_masking=True: Evaluates masked reconstruction loss (MAE task).
        - eval_mae_masking=False: Evaluates full image autoencoding quality.
        """
        # Save original MAE parameters to restore later
        encoder = self.model.encoder if not hasattr(self.model, "module") else self.model.module.encoder
        orig_p_min, orig_p_max = encoder.mae.p_min, encoder.mae.p_max

        if not eval_mae_masking:
            encoder.mae.p_min = 0.0
            encoder.mae.p_max = 0.0

        total_full_mse = 0.0
        total_mae_mse = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        batches_run = 0

        for i, batch in enumerate(self.dataloader):
            if num_batches is not None and i >= num_batches:
                break

            # Fetch and normalize video sequence (B, T, C, H, W)
            x = batch["image"].to(self.device, non_blocking=True)
            if x.dtype == torch.uint8 or float(x.max().item()) > 1.5:
                x = x.to(torch.float32) / 255.0
            else:
                x = x.to(torch.float32)

            B, T, C, H, W = x.shape
            patches = temporal_patchify(x, self.patch_size)  # (B, T, Np, Dp)

            # Forward pass
            pred, mae_mask, keep_prob = self.model(patches)

            # 1. Full Frame Reconstruction MSE (across all patches)
            full_mse = F.mse_loss(pred.float(), patches.float())
            psnr = 10.0 * torch.log10(1.0 / full_mse.clamp_min(1e-10))

            total_full_mse += full_mse.item()
            total_psnr += psnr.item()

            # 2. Masked Patch MSE (if masking is enabled)
            if eval_mae_masking and mae_mask.any():
                mae_mse = recon_loss_from_mae(pred, patches, mae_mask)
                total_mae_mse += mae_mse.item()

            # 3. LPIPS Perceptual Loss
            if self.lpips_fn is not None:
                lp = lpips_on_mae_recon(
                    self.lpips_fn, pred, patches, mae_mask,
                    H=H, W=W, C=C, patch=self.patch_size,
                    subsample_frac=1.0
                )
                total_lpips += lp.item()

            batches_run += 1

        # Restore original encoder settings
        encoder.mae.p_min, encoder.mae.p_max = orig_p_min, orig_p_max

        return {
            "full_recon_mse": total_full_mse / batches_run,
            "psnr_db": total_psnr / batches_run,
            "masked_mae_mse": (total_mae_mse / batches_run) if eval_mae_masking else None,
            "lpips": (total_lpips / batches_run) if self.lpips_fn else None,
        }

    @torch.no_grad()
    def visualize_reconstruction(self, save_path="reconstruction.png", max_frames=6):
        """
        Saves a visual comparison grid: Original | Masked Input | Full Reconstruction
        """
        batch = next(iter(self.dataloader))
        x = batch["image"].to(self.device)
        if x.dtype == torch.uint8 or float(x.max().item()) > 1.5:
            x = x.to(torch.float32) / 255.0

        B, T, C, H, W = x.shape
        patches = temporal_patchify(x, self.patch_size)

        # Forward pass
        pred_btnd, mae_mask_btNp1, _ = self.model(patches)

        # Masked input in patch space
        masked_input_btnd = torch.where(mae_mask_btNp1, torch.zeros_like(patches), patches)

        # Convert back to image space
        masked_img = temporal_unpatchify(masked_input_btnd, H, W, C, self.patch_size)
        recon_img = temporal_unpatchify(pred_btnd, H, W, C, self.patch_size)

        num_frames = min(T, max_frames)
        fig, axes = plt.subplots(3, num_frames, figsize=(3 * num_frames, 8))

        for t in range(num_frames):
            # Original
            axes[0, t].imshow(x[0, t].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
            axes[0, t].set_title(f"Target T={t}")
            axes[0, t].axis("off")

            # Masked Input
            axes[1, t].imshow(masked_img[0, t].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
            axes[1, t].set_title(f"Masked Input T={t}")
            axes[1, t].axis("off")

            # Reconstruction
            axes[2, t].imshow(recon_img[0, t].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
            axes[2, t].set_title(f"Reconstruction T={t}")
            axes[2, t].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
        plt.close()


def load_model_from_ckpt(ckpt_path, args, device):
    n_patches = (args.H // args.patch) * (args.W // args.patch)
    d_patch = args.patch * args.patch * args.C

    enc = Encoder(
        patch_dim=d_patch,
        d_model=args.d_model,
        n_latents=args.n_latents,
        n_patches=n_patches,
        n_heads=args.n_heads,
        depth=args.depth,
        d_bottleneck=args.d_bottleneck,
        dropout=0.0,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        mae_p_min=args.mae_p_min,
        mae_p_max=args.mae_p_max,
        scale_pos_embeds=args.scale_pos_embeds,
    )
    dec = Decoder(
        d_bottleneck=args.d_bottleneck,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        n_latents=args.n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=0.0,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        scale_pos_embeds=args.scale_pos_embeds,
    )

    model = Tokenizer(enc, dec).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)

    # Strip DDP "module." prefix if saved during distributed training
    if list(state.keys())[0].startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation .h5 dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_eval_batches", type=int, default=20)

    # Model Params (Matches train_tokenizer.py)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--action_chunk_size", type=int, default=5)
    parser.add_argument("--H", type=int, default=224)
    parser.add_argument("--W", type=int, default=224)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--n_latents", type=int, default=16)
    parser.add_argument("--d_bottleneck", type=int, default=32)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--time_every", type=int, default=1)
    parser.add_argument("--mae_p_min", type=float, default=0.0)
    parser.add_argument("--mae_p_max", type=float, default=0.9)
    parser.add_argument("--scale_pos_embeds", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}...")

    # Dataloader setup
    dataset = PushTSequenceDataset(
        h5_path=args.dataset,
        seq_len=args.seq_len,
        action_chunk_size=args.action_chunk_size,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load Model
    model = load_model_from_ckpt(args.ckpt, args, device)
    evaluator = TokenizerEvaluator(model, dataloader, device, patch_size=args.patch)

    # 1. Evaluate MAE Masked Performance
    print("\n--- Masked MAE Metrics ---")
    masked_metrics = evaluator.evaluate(num_batches=args.num_eval_batches, eval_mae_masking=True)
    for k, v in masked_metrics.items():
        if v is not None:
            print(f"{k}: {v:.6f}")

    # 2. Evaluate Full Reconstruction Quality (Unmasked Autoencoding)
    print("\n--- Unmasked Bottleneck Quality ---")
    unmasked_metrics = evaluator.evaluate(num_batches=args.num_eval_batches, eval_mae_masking=False)
    for k, v in unmasked_metrics.items():
        if v is not None:
            print(f"{k}: {v:.6f}")

    # 3. Generate Image Visualizations
    evaluator.visualize_reconstruction("eval_visualization.png")