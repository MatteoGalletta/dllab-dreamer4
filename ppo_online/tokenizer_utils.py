from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_DREAMER4_DIR = Path(__file__).resolve().parent.parent / "dreamer4-src" / "dreamer4"
if str(_DREAMER4_DIR) not in sys.path:
    sys.path.insert(0, str(_DREAMER4_DIR))

from model import Decoder, Encoder, Tokenizer, temporal_patchify  # type: ignore  # noqa: E402


_TOKENIZER_CACHE: dict[tuple[str, str], tuple[Tokenizer, dict[str, int]]] = {}


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Tokenizer checkpoint must be a dictionary payload.")

    for key in ("state_dict", "model_state_dict", "model", "tokenizer", "module"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested

    tensor_values = [value for value in payload.values() if torch.is_tensor(value)]
    if tensor_values and len(tensor_values) == len(payload):
        return payload

    raise ValueError("Could not find a state_dict in tokenizer checkpoint.")


def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "tokenizer."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned[clean_key] = value
    return cleaned


def load_tokenizer_from_ckpt(tokenizer_ckpt: str, device: torch.device):
    cache_key = (str(Path(tokenizer_ckpt).resolve()), str(device))
    if cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    ckpt = torch.load(tokenizer_ckpt, map_location="cpu")
    args = ckpt.get("args", {}) or {}

    H = int(args.get("H", 128))
    W = int(args.get("W", 128))
    C = int(args.get("C", 3))
    patch = int(args.get("patch", 4))
    d_model = int(args.get("d_model", 256))
    n_heads = int(args.get("n_heads", 4))
    depth = int(args.get("depth", 8))
    n_latents = int(args.get("n_latents", 16))
    d_bottleneck = int(args.get("d_bottleneck", 32))
    dropout = float(args.get("dropout", 0.0))
    mlp_ratio = float(args.get("mlp_ratio", 4.0))
    time_every = int(args.get("time_every", 1))
    scale_pos_embeds = bool(args.get("scale_pos_embeds", True))

    if H % patch != 0 or W % patch != 0:
        raise ValueError(f"Tokenizer image size {(H, W)} must be divisible by patch {patch}.")

    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    encoder = Encoder(
        patch_dim=d_patch,
        d_model=d_model,
        n_latents=n_latents,
        n_patches=n_patches,
        n_heads=n_heads,
        depth=depth,
        d_bottleneck=d_bottleneck,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
        mae_p_min=0.0,
        mae_p_max=0.0,
        scale_pos_embeds=scale_pos_embeds,
    )
    decoder = Decoder(
        d_bottleneck=d_bottleneck,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        n_latents=n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
        scale_pos_embeds=scale_pos_embeds,
    )
    tokenizer = Tokenizer(encoder, decoder).to(device)
    tokenizer.load_state_dict(_clean_state_dict_keys(_extract_state_dict(ckpt)), strict=True)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    info = {
        "H": H,
        "W": W,
        "C": C,
        "patch": patch,
        "n_latents": n_latents,
        "d_bottleneck": d_bottleneck,
        "latent_dim": n_latents * d_bottleneck,
    }
    _TOKENIZER_CACHE[cache_key] = (tokenizer, info)
    return tokenizer, info


class TokenizerZEncoder:
    def __init__(self, tokenizer_ckpt: str, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer, self.info = load_tokenizer_from_ckpt(tokenizer_ckpt, self.device)

    @property
    def latent_dim(self) -> int:
        return int(self.info["latent_dim"])

    def preprocess_frame(self, frame_hwc: np.ndarray) -> torch.Tensor:
        frame = np.asarray(frame_hwc)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame shaped (H, W, 3), got {tuple(frame.shape)}")

        frame_tensor = torch.as_tensor(frame, device=self.device)
        if frame_tensor.dtype == torch.uint8:
            frame_tensor = frame_tensor.to(torch.float32) / 255.0
        else:
            frame_tensor = frame_tensor.to(torch.float32)
            if float(frame_tensor.max().item()) > 1.5:
                frame_tensor = frame_tensor / 255.0

        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        frame_tensor = F.interpolate(
            frame_tensor,
            size=(int(self.info["H"]), int(self.info["W"])),
            mode="bilinear",
            align_corners=False,
        )
        return frame_tensor.clamp(0.0, 1.0)

    @torch.inference_mode()
    def encode_frame(self, frame_hwc: np.ndarray) -> np.ndarray:
        frame = self.preprocess_frame(frame_hwc)
        patches = temporal_patchify(frame.unsqueeze(1), int(self.info["patch"]))
        z_btld, _ = self.tokenizer.encoder(patches)
        z_flat = z_btld.reshape(-1).detach().cpu().numpy().astype(np.float32)
        return z_flat
