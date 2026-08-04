#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from behavioural_cloning.train_base import (  # noqa: E402
    CNNBackbone,
    TokenizerBackbone,
    load_tokenizer_encoder,
    normalize_image_batch,
)


@dataclass
class EncoderBundle:
    name: str
    image_hw: tuple[int, int]
    feature_dim: int
    model: torch.nn.Module


def _load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path} is not a supported BC checkpoint.")
    return checkpoint


def _subset_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    subset = {}
    needle = f"{prefix}."
    for key, value in state_dict.items():
        if key.startswith(needle):
            subset[key[len(needle) :]] = value
    if not subset:
        raise KeyError(f"Could not find prefix '{prefix}' in checkpoint.")
    return subset


def load_cnn_encoder(checkpoint_path: str | Path, device: torch.device) -> EncoderBundle:
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint["model"]
    args = checkpoint.get("args", {}) or {}

    feature_dim = int(state_dict["backbone.proj.1.weight"].shape[0])
    in_channels = int(args.get("C", 3))
    image_hw = (int(args.get("H", 224)), int(args.get("W", 224)))

    backbone = CNNBackbone(in_channels=in_channels, feature_dim=feature_dim)
    backbone.load_state_dict(_subset_state_dict(state_dict, "backbone"), strict=True)
    backbone.to(device)
    backbone.eval()

    return EncoderBundle(
        name="CNN encoder",
        image_hw=image_hw,
        feature_dim=feature_dim,
        model=backbone,
    )


def load_tokenizer_encoder_bundle(
    latent_checkpoint_path: str | Path,
    tokenizer_checkpoint_path: str | Path,
    device: torch.device,
) -> EncoderBundle:
    checkpoint = _load_checkpoint(latent_checkpoint_path)
    state_dict = checkpoint["model"]
    args = checkpoint.get("args", {}) or {}

    feature_dim = int(args.get("tokenizer_feature_dim", state_dict["backbone.projector.1.weight"].shape[0]))
    encoder = load_tokenizer_encoder(str(tokenizer_checkpoint_path))
    backbone = TokenizerBackbone(
        encoder,
        patch=int(encoder.patch),
        output_dim=feature_dim,
    )
    backbone.load_state_dict(_subset_state_dict(state_dict, "backbone"), strict=True)
    backbone.to(device)
    backbone.eval()

    return EncoderBundle(
        name="Frozen tokenizer encoder",
        image_hw=(int(args.get("H", 224)), int(args.get("W", 224))),
        feature_dim=feature_dim,
        model=backbone,
    )


def load_npz_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, mmap_mode="r", allow_pickle=True) as data:
        images = np.asarray(data["images"])
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    return images, states, actions


def sample_indices(num_items: int, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    count = min(int(num_samples), int(num_items))
    return np.sort(rng.choice(num_items, size=count, replace=False))


def preprocess_images(images_hwc: np.ndarray, image_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(images_hwc)).permute(0, 3, 1, 2)
    x = normalize_image_batch(x)
    if tuple(x.shape[-2:]) != tuple(image_hw):
        x = F.interpolate(
            x,
            size=image_hw,
            mode="bilinear",
            align_corners=False,
        )
    return x.to(device=device, dtype=torch.float32, non_blocking=True)


@torch.inference_mode()
def encode_frames(
    bundle: EncoderBundle,
    images_hwc: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    features = []
    for start in range(0, len(images_hwc), int(batch_size)):
        end = min(start + int(batch_size), len(images_hwc))
        batch = preprocess_images(images_hwc[start:end], bundle.image_hw, device)
        encoded = bundle.model(batch.unsqueeze(1)).squeeze(1).detach().cpu().numpy().astype(np.float32)
        features.append(encoded)
    return np.concatenate(features, axis=0)


def pca_project(x: np.ndarray, dims: int = 2) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    coords = u[:, :dims] * s[:dims]
    explained = (s**2) / max(np.sum(s**2), 1e-12)
    return coords.astype(np.float32), explained[:dims].astype(np.float32)


def ridge_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    train_frac: float = 0.8,
    ridge: float = 1e-3,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(len(x))
    train_size = max(2, int(round(len(x) * float(train_frac))))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    if len(test_idx) == 0:
        test_idx = train_idx[-max(1, train_size // 5) :]
        train_idx = train_idx[: -len(test_idx)]

    x_train = x[train_idx].astype(np.float64)
    x_test = x[test_idx].astype(np.float64)
    y_train = y[train_idx].astype(np.float64)
    y_test = y[test_idx].astype(np.float64)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True)

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train_centered = y_train - y_mean

    eye = np.eye(x_train.shape[1], dtype=np.float64)
    weights = np.linalg.solve(x_train.T @ x_train + float(ridge) * eye, x_train.T @ y_train_centered)
    pred = x_test @ weights + y_mean

    mse = float(np.mean((pred - y_test) ** 2))
    mae = float(np.mean(np.abs(pred - y_test)))
    ss_res = float(np.sum((pred - y_test) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"mse": mse, "mae": mae, "r2": r2}


def nearest_neighbor_state_distance(features: np.ndarray, states: np.ndarray) -> float:
    features = np.asarray(features, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    sq = np.sum(features**2, axis=1, keepdims=True)
    distances = sq + sq.T - 2.0 * (features @ features.T)
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, np.inf)
    nn_idx = np.argmin(distances, axis=1)
    return float(np.mean(np.linalg.norm(states - states[nn_idx], axis=1)))


def build_scalar_views(states: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    agent_xy = states[:, :2]
    block_xy = states[:, 2:4]
    block_theta = states[:, 4]
    action_mag = np.linalg.norm(actions, axis=1)
    agent_block_dist = np.linalg.norm(agent_xy - block_xy, axis=1)
    return {
        "agent_block_dist": agent_block_dist,
        "action_mag": action_mag,
        "block_theta": block_theta,
    }


def _hex_from_rgb(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _lerp_color(value: float, anchors: list[tuple[float, tuple[int, int, int]]]) -> str:
    value = float(np.clip(value, 0.0, 1.0))
    for idx in range(len(anchors) - 1):
        left_t, left_rgb = anchors[idx]
        right_t, right_rgb = anchors[idx + 1]
        if left_t <= value <= right_t:
            if right_t <= left_t:
                return _hex_from_rgb(left_rgb)
            alpha = (value - left_t) / (right_t - left_t)
            rgb = tuple(int(round((1.0 - alpha) * a + alpha * b)) for a, b in zip(left_rgb, right_rgb))
            return _hex_from_rgb(rgb)
    return _hex_from_rgb(anchors[-1][1])


def _viridis(value: float) -> str:
    anchors = [
        (0.0, (68, 1, 84)),
        (0.33, (59, 82, 139)),
        (0.66, (33, 145, 140)),
        (1.0, (253, 231, 37)),
    ]
    return _lerp_color(value, anchors)


def _magma(value: float) -> str:
    anchors = [
        (0.0, (0, 0, 4)),
        (0.33, (82, 18, 123)),
        (0.66, (183, 55, 121)),
        (1.0, (252, 253, 191)),
    ]
    return _lerp_color(value, anchors)


def _normalize_for_color(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _project_to_panel(coords: np.ndarray, width: float, height: float, padding: float = 24.0) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]
    x_lo, x_hi = float(x.min()), float(x.max())
    y_lo, y_hi = float(y.min()), float(y.max())
    x_span = max(x_hi - x_lo, 1e-6)
    y_span = max(y_hi - y_lo, 1e-6)
    sx = (width - 2 * padding) / x_span
    sy = (height - 2 * padding) / y_span
    px = padding + (x - x_lo) * sx
    py = height - padding - (y - y_lo) * sy
    return np.stack([px, py], axis=1)


def _svg_scatter_panel(
    title: str,
    coords: np.ndarray,
    values: np.ndarray,
    color_fn,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    pts = _project_to_panel(coords, width, height)
    norm_values = _normalize_for_color(values)
    parts = [
        f'<g transform="translate({x0},{y0})">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#cccccc" stroke-width="1.2"/>',
        f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="16" font-family="Arial">{title}</text>',
    ]
    for (px, py), scalar in zip(pts, norm_values):
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.4" fill="{color_fn(float(scalar))}" '
            f'fill-opacity="0.86" stroke="none"/>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def _svg_bar_panel(
    metrics: dict[str, dict[str, float]],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    tasks = [("full_state", "Full state"), ("block_pose", "Block pose"), ("action_xy", "Action")]
    cnn_vals = [metrics["cnn"][task]["r2"] for task, _ in tasks]
    tok_vals = [metrics["tokenizer"][task]["r2"] for task, _ in tasks]
    y_min = min(0.0, min(cnn_vals + tok_vals))
    y_max = max(1e-6, max(cnn_vals + tok_vals))
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0

    def y_map(value: float) -> float:
        top = 38.0
        bottom = height - 36.0
        alpha = (value - y_min) / (y_max - y_min)
        return bottom - alpha * (bottom - top)

    base_y = y_map(0.0)
    slot_w = width / len(tasks)
    bar_w = slot_w * 0.24
    parts = [
        f'<g transform="translate({x0},{y0})">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#cccccc" stroke-width="1.2"/>',
        f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="16" font-family="Arial">Linear probe R²</text>',
        f'<line x1="42" y1="{base_y:.2f}" x2="{width-16:.1f}" y2="{base_y:.2f}" stroke="#666666" stroke-width="1"/>',
    ]
    for idx, (_, label) in enumerate(tasks):
        cx = slot_w * (idx + 0.5)
        for dx, value, color in ((-bar_w * 0.65, cnn_vals[idx], "#2D6CDF"), (bar_w * 0.65, tok_vals[idx], "#E17C05")):
            x = cx + dx - bar_w / 2
            y = min(base_y, y_map(value))
            h = abs(base_y - y_map(value))
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        parts.append(
            f'<text x="{cx:.2f}" y="{height-12:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>'
        )
    parts.extend(
        [
            f'<rect x="{width-132:.1f}" y="34" width="12" height="12" fill="#2D6CDF"/>',
            f'<text x="{width-114:.1f}" y="44" font-size="12" font-family="Arial">CNN</text>',
            f'<rect x="{width-132:.1f}" y="54" width="12" height="12" fill="#E17C05"/>',
            f'<text x="{width-114:.1f}" y="64" font-size="12" font-family="Arial">Tokenizer</text>',
            "</g>",
        ]
    )
    return "\n".join(parts)


def _svg_text_panel(
    metrics: dict[str, dict[str, float]],
    metadata: dict[str, object],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    lines = [
        f"Dataset: {metadata['dataset_name']}",
        f"Samples: {metadata['num_samples']}",
        f"CNN dim: {metadata['cnn_feature_dim']}",
        f"Tokenizer dim: {metadata['tokenizer_feature_dim']}",
        "",
        f"CNN PCA var: {metadata['cnn_pca_var'][0]:.2%}, {metadata['cnn_pca_var'][1]:.2%}",
        f"Tokenizer PCA var: {metadata['tokenizer_pca_var'][0]:.2%}, {metadata['tokenizer_pca_var'][1]:.2%}",
        "",
        f"CNN NN state dist: {metrics['cnn']['nn_state_dist']:.3f}",
        f"Tok NN state dist: {metrics['tokenizer']['nn_state_dist']:.3f}",
        "",
        "Higher probe R² and lower",
        "nearest-neighbor state distance",
        "mean that control-relevant",
        "information is easier to decode.",
    ]
    parts = [
        f'<g transform="translate({x0},{y0})">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#cccccc" stroke-width="1.2"/>',
        f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="16" font-family="Arial">Summary</text>',
    ]
    y = 50
    for line in lines:
        parts.append(f'<text x="18" y="{y}" font-size="14" font-family="Courier New">{line}</text>')
        y += 22
    parts.append("</g>")
    return "\n".join(parts)


def plot_comparison_svg(
    cnn_coords: np.ndarray,
    tok_coords: np.ndarray,
    scalars: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    metadata: dict[str, object],
    output_path: Path,
) -> None:
    width = 1600
    height = 980
    panel_w = 470
    panel_h = 390
    right_w = 520
    top = 60
    left = 40
    gap = 24

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7fb"/>',
        '<text x="800" y="34" text-anchor="middle" font-size="26" font-family="Arial" font-weight="bold">'
        'Representation comparison: CNN encoder vs frozen tokenizer encoder</text>',
        _svg_scatter_panel(
            "CNN PCA colored by agent-block distance",
            cnn_coords,
            scalars["agent_block_dist"],
            _viridis,
            x0=left,
            y0=top,
            width=panel_w,
            height=panel_h,
        ),
        _svg_scatter_panel(
            "Tokenizer PCA colored by agent-block distance",
            tok_coords,
            scalars["agent_block_dist"],
            _viridis,
            x0=left + panel_w + gap,
            y0=top,
            width=panel_w,
            height=panel_h,
        ),
        _svg_bar_panel(
            metrics,
            x0=left + 2 * (panel_w + gap),
            y0=top,
            width=right_w,
            height=panel_h,
        ),
        _svg_scatter_panel(
            "CNN PCA colored by action magnitude",
            cnn_coords,
            scalars["action_mag"],
            _magma,
            x0=left,
            y0=top + panel_h + gap,
            width=panel_w,
            height=panel_h,
        ),
        _svg_scatter_panel(
            "Tokenizer PCA colored by action magnitude",
            tok_coords,
            scalars["action_mag"],
            _magma,
            x0=left + panel_w + gap,
            y0=top + panel_h + gap,
            width=panel_w,
            height=panel_h,
        ),
        _svg_text_panel(
            metrics,
            metadata,
            x0=left + 2 * (panel_w + gap),
            y0=top + panel_h + gap,
            width=right_w,
            height=panel_h,
        ),
        "</svg>",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def plot_comparison(
    cnn_coords: np.ndarray,
    tok_coords: np.ndarray,
    scalars: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    metadata: dict[str, object],
    output_stem: Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        svg_path = output_stem.with_suffix(".svg")
        plot_comparison_svg(cnn_coords, tok_coords, scalars, metrics, metadata, svg_path)
        return svg_path

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.28, hspace=0.24)

    ax_cnn_dist = fig.add_subplot(gs[0, 0])
    ax_tok_dist = fig.add_subplot(gs[0, 1])
    ax_probe = fig.add_subplot(gs[0, 2])
    ax_cnn_act = fig.add_subplot(gs[1, 0])
    ax_tok_act = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2])

    scatter_kwargs = dict(s=10, alpha=0.85, linewidths=0)

    dist = scalars["agent_block_dist"]
    act = scalars["action_mag"]

    sc1 = ax_cnn_dist.scatter(cnn_coords[:, 0], cnn_coords[:, 1], c=dist, cmap="viridis", **scatter_kwargs)
    sc2 = ax_tok_dist.scatter(tok_coords[:, 0], tok_coords[:, 1], c=dist, cmap="viridis", **scatter_kwargs)
    sc3 = ax_cnn_act.scatter(cnn_coords[:, 0], cnn_coords[:, 1], c=act, cmap="magma", **scatter_kwargs)
    sc4 = ax_tok_act.scatter(tok_coords[:, 0], tok_coords[:, 1], c=act, cmap="magma", **scatter_kwargs)

    for ax, title in (
        (ax_cnn_dist, "CNN PCA colored by agent-block distance"),
        (ax_tok_dist, "Tokenizer PCA colored by agent-block distance"),
        (ax_cnn_act, "CNN PCA colored by action magnitude"),
        (ax_tok_act, "Tokenizer PCA colored by action magnitude"),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    cbar1 = fig.colorbar(sc1, ax=[ax_cnn_dist, ax_tok_dist], fraction=0.025, pad=0.02)
    cbar1.set_label("Agent-block distance")
    cbar2 = fig.colorbar(sc3, ax=[ax_cnn_act, ax_tok_act], fraction=0.025, pad=0.02)
    cbar2.set_label("Action magnitude")

    tasks = ["full_state", "block_pose", "action_xy"]
    labels = ["Full state", "Block pose", "Action"]
    x = np.arange(len(tasks))
    width = 0.34
    cnn_vals = [metrics["cnn"][task]["r2"] for task in tasks]
    tok_vals = [metrics["tokenizer"][task]["r2"] for task in tasks]
    ax_probe.bar(x - width / 2, cnn_vals, width, label="CNN", color="#2D6CDF")
    ax_probe.bar(x + width / 2, tok_vals, width, label="Tokenizer", color="#E17C05")
    ax_probe.axhline(0.0, color="black", linewidth=0.8)
    ax_probe.set_xticks(x, labels)
    ax_probe.set_ylabel("Linear probe $R^2$")
    ax_probe.set_title("How much task information is linearly accessible?")
    ax_probe.legend(frameon=False)

    ax_text.axis("off")
    text = (
        f"Dataset: {metadata['dataset_name']}\n"
        f"Samples: {metadata['num_samples']}\n"
        f"CNN dim: {metadata['cnn_feature_dim']}\n"
        f"Tokenizer dim: {metadata['tokenizer_feature_dim']}\n\n"
        f"CNN PCA var: {metadata['cnn_pca_var'][0]:.2%}, {metadata['cnn_pca_var'][1]:.2%}\n"
        f"Tokenizer PCA var: {metadata['tokenizer_pca_var'][0]:.2%}, {metadata['tokenizer_pca_var'][1]:.2%}\n\n"
        f"NN state dist\n"
        f"  CNN: {metrics['cnn']['nn_state_dist']:.3f}\n"
        f"  Tokenizer: {metrics['tokenizer']['nn_state_dist']:.3f}\n\n"
        "Interpretation:\n"
        "Higher probe R² and lower NN state distance suggest\n"
        "that useful control information is easier to recover\n"
        "from that representation."
    )
    ax_text.text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle("Representation comparison: CNN encoder vs frozen tokenizer encoder", fontsize=15, y=0.98)
    png_path = output_stem.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path


def summarize_metrics(
    cnn_features: np.ndarray,
    tokenizer_features: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    targets = {
        "full_state": states,
        "block_pose": states[:, 2:5],
        "action_xy": actions,
    }
    metrics = {"cnn": {}, "tokenizer": {}}
    for task_name, target in targets.items():
        metrics["cnn"][task_name] = ridge_probe(cnn_features, target, seed=seed)
        metrics["tokenizer"][task_name] = ridge_probe(tokenizer_features, target, seed=seed)
    metrics["cnn"]["nn_state_dist"] = nearest_neighbor_state_distance(cnn_features, states)
    metrics["tokenizer"]["nn_state_dist"] = nearest_neighbor_state_distance(tokenizer_features, states)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare CNN and tokenizer representation spaces on PushT frames.")
    parser.add_argument("--dataset", default="data/expert_trajectories/pusht_expert.npz")
    parser.add_argument("--cnn-checkpoint", default="local_models/behavior_cloning/cnn_strided_bc.pt")
    parser.add_argument("--tokenizer-bc-checkpoint", default="local_models/behavior_cloning/latent.pt")
    parser.add_argument("--tokenizer-checkpoint", default="local_models/tokenizer/tokenizer.pt")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="runs/representation_comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    images, states, actions = load_npz_dataset(args.dataset)
    indices = sample_indices(len(images), args.num_samples, args.seed)
    images = images[indices]
    states = states[indices]
    actions = actions[indices]

    cnn_bundle = load_cnn_encoder(args.cnn_checkpoint, device)
    tokenizer_bundle = load_tokenizer_encoder_bundle(
        args.tokenizer_bc_checkpoint,
        args.tokenizer_checkpoint,
        device,
    )

    cnn_features = encode_frames(cnn_bundle, images, batch_size=args.batch_size, device=device)
    tokenizer_features = encode_frames(tokenizer_bundle, images, batch_size=args.batch_size, device=device)

    cnn_coords, cnn_var = pca_project(cnn_features, dims=2)
    tok_coords, tok_var = pca_project(tokenizer_features, dims=2)
    metrics = summarize_metrics(cnn_features, tokenizer_features, states, actions, seed=args.seed)
    scalars = build_scalar_views(states, actions)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_stem = out_dir / "representation_comparison"
    metrics_path = out_dir / "representation_metrics.json"

    metadata = {
        "dataset_name": Path(args.dataset).name,
        "num_samples": int(len(indices)),
        "cnn_feature_dim": int(cnn_bundle.feature_dim),
        "tokenizer_feature_dim": int(tokenizer_bundle.feature_dim),
        "cnn_pca_var": [float(x) for x in cnn_var],
        "tokenizer_pca_var": [float(x) for x in tok_var],
        "cnn_checkpoint": str(Path(args.cnn_checkpoint).resolve()),
        "tokenizer_bc_checkpoint": str(Path(args.tokenizer_bc_checkpoint).resolve()),
        "tokenizer_checkpoint": str(Path(args.tokenizer_checkpoint).resolve()),
    }

    plot_path = plot_comparison(cnn_coords, tok_coords, scalars, metrics, metadata, plot_stem)
    payload = {"metadata": metadata, "metrics": metrics}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved figure: {plot_path}")
    print(f"Saved metrics: {metrics_path}")
    print()
    print("Linear probe summary (R^2):")
    for task_name in ("full_state", "block_pose", "action_xy"):
        cnn_r2 = metrics["cnn"][task_name]["r2"]
        tok_r2 = metrics["tokenizer"][task_name]["r2"]
        print(f"  {task_name:>10s} | CNN={cnn_r2:7.4f} | Tokenizer={tok_r2:7.4f}")
    print(
        f"Nearest-neighbor state distance | CNN={metrics['cnn']['nn_state_dist']:.4f} "
        f"| Tokenizer={metrics['tokenizer']['nn_state_dist']:.4f}"
    )


if __name__ == "__main__":
    main()
