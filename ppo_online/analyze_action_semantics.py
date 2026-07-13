#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze whether PushT dataset actions look absolute or delta-like.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the PushT HDF5 dataset.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200000,
        help="Maximum number of transitions to analyze.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="debug/action_semantics_summary.json",
        help="Where to save the JSON summary.",
    )
    return parser.parse_args()


def fit_affine(x: np.ndarray, y: np.ndarray) -> dict:
    ones = np.ones((x.shape[0], 1), dtype=np.float32)
    design = np.concatenate([x, ones], axis=1)
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ weights
    residual = y - pred
    mse = float(np.mean(np.square(residual)))
    var = float(np.var(y))
    r2 = float(1.0 - (np.var(residual) / var)) if var > 1e-12 else float("nan")
    return {
        "weights": weights[: x.shape[1]].tolist(),
        "bias": weights[x.shape[1] :].reshape(-1).tolist(),
        "mse": mse,
        "r2": r2,
    }


def summarize_fixed_mapping(action: np.ndarray, target: np.ndarray, scale: float, bias: float) -> dict:
    pred = action * scale + bias
    residual = target - pred
    mse = float(np.mean(np.square(residual)))
    var = float(np.var(target))
    r2 = float(1.0 - (np.var(residual) / var)) if var > 1e-12 else float("nan")
    return {"scale": scale, "bias": bias, "mse": mse, "r2": r2}


def fit_scalar_scale(action: np.ndarray, target: np.ndarray) -> dict:
    numerator = float(np.sum(action * target))
    denominator = float(np.sum(action * action))
    scale = numerator / denominator if denominator > 1e-12 else 0.0
    pred = action * scale
    residual = target - pred
    mse = float(np.mean(np.square(residual)))
    var = float(np.var(target))
    r2 = float(1.0 - (np.var(residual) / var)) if var > 1e-12 else float("nan")
    return {"scale": scale, "mse": mse, "r2": r2}


def fit_per_dim_scale(action: np.ndarray, target: np.ndarray) -> dict:
    scales = []
    mses = []
    r2s = []
    for dim in range(action.shape[1]):
        numerator = float(np.sum(action[:, dim] * target[:, dim]))
        denominator = float(np.sum(action[:, dim] * action[:, dim]))
        scale = numerator / denominator if denominator > 1e-12 else 0.0
        pred = action[:, dim] * scale
        residual = target[:, dim] - pred
        mse = float(np.mean(np.square(residual)))
        var = float(np.var(target[:, dim]))
        r2 = float(1.0 - (np.var(residual) / var)) if var > 1e-12 else float("nan")
        scales.append(scale)
        mses.append(mse)
        r2s.append(r2)
    return {
        "scales": scales,
        "mean_mse": float(np.mean(mses)),
        "mean_r2": float(np.nanmean(r2s)),
        "mse_per_dim": mses,
        "r2_per_dim": r2s,
    }


def evaluate_candidate_scales(action: np.ndarray, target: np.ndarray, candidates: list[float]) -> list[dict]:
    results = []
    for scale in candidates:
        pred = action * scale
        residual = target - pred
        mse = float(np.mean(np.square(residual)))
        var = float(np.var(target))
        r2 = float(1.0 - (np.var(residual) / var)) if var > 1e-12 else float("nan")
        results.append({"scale": float(scale), "mse": mse, "r2": r2})
    results.sort(key=lambda item: item["mse"])
    return results


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as dataset:
        actions = np.asarray(dataset["action"], dtype=np.float32)
        states = np.asarray(dataset["state"], dtype=np.float32)
        ep_offsets = np.asarray(dataset["ep_offset"], dtype=np.int64)
        ep_lens = np.asarray(dataset["ep_len"], dtype=np.int64)

    total_frames = actions.shape[0]
    if total_frames < 2:
        raise ValueError("Dataset must contain at least 2 frames.")

    episode_end_mask = np.zeros(total_frames - 1, dtype=bool)
    episode_ends = ep_offsets + ep_lens
    valid_episode_ends = episode_ends[(episode_ends > 0) & (episode_ends <= total_frames)]
    for end_idx in valid_episode_ends:
        if end_idx - 1 < episode_end_mask.shape[0]:
            episode_end_mask[end_idx - 1] = True

    valid_mask = ~episode_end_mask
    valid_indices = np.flatnonzero(valid_mask)
    if args.max_steps > 0:
        valid_indices = valid_indices[: min(args.max_steps, valid_indices.shape[0])]

    a_t = actions[valid_indices]
    agent_t = states[valid_indices, 0:2]
    agent_tp1 = states[valid_indices + 1, 0:2]
    delta_agent = agent_tp1 - agent_t
    block_t = states[valid_indices, 2:4]
    block_tp1 = states[valid_indices + 1, 2:4]
    delta_block = block_tp1 - block_t

    delta_scale_isotropic = fit_scalar_scale(a_t, delta_agent)
    delta_scale_per_dim = fit_per_dim_scale(a_t, delta_agent)
    candidate_scales = [1.0, 2.0, 4.0, 8.0, 12.0, 15.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 256.0]
    candidate_delta_scales = evaluate_candidate_scales(a_t, delta_agent, candidate_scales)

    summary = {
        "dataset_path": str(Path(args.dataset).resolve()),
        "num_total_frames": int(total_frames),
        "num_valid_transitions": int(valid_indices.shape[0]),
        "action_shape": list(actions.shape),
        "state_shape": list(states.shape),
        "action_stats": {
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "mean": actions.mean(axis=0).tolist(),
        },
        "agent_delta_stats": {
            "min": delta_agent.min(axis=0).tolist(),
            "max": delta_agent.max(axis=0).tolist(),
            "mean": delta_agent.mean(axis=0).tolist(),
        },
        "block_delta_stats": {
            "min": delta_block.min(axis=0).tolist(),
            "max": delta_block.max(axis=0).tolist(),
            "mean": delta_block.mean(axis=0).tolist(),
        },
        "tests": {
            "absolute_fixed_256_256": summarize_fixed_mapping(a_t, agent_tp1, scale=256.0, bias=256.0),
            "absolute_fixed_512_0": summarize_fixed_mapping(a_t, agent_tp1, scale=512.0, bias=0.0),
            "absolute_affine_fit_to_next_agent": fit_affine(a_t, agent_tp1),
            "delta_affine_fit_to_agent_delta": fit_affine(a_t, delta_agent),
            "delta_scalar_scale_fit": delta_scale_isotropic,
            "delta_per_dim_scale_fit": delta_scale_per_dim,
            "delta_candidate_scales": candidate_delta_scales,
        },
    }

    abs_r2 = summary["tests"]["absolute_affine_fit_to_next_agent"]["r2"]
    delta_r2 = summary["tests"]["delta_affine_fit_to_agent_delta"]["r2"]
    if np.isnan(abs_r2) or np.isnan(delta_r2):
        likely = "inconclusive"
    elif abs_r2 > delta_r2 + 0.05:
        likely = "absolute_like"
    elif delta_r2 > abs_r2 + 0.05:
        likely = "delta_like"
    else:
        likely = "ambiguous"
    summary["likely_semantics"] = likely

    out_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved action semantics summary to {out_path}")
    print(f"Valid transitions analyzed: {summary['num_valid_transitions']}")
    print(f"Likely semantics: {likely}")
    print(
        "R2 absolute vs delta:",
        summary["tests"]["absolute_affine_fit_to_next_agent"]["r2"],
        summary["tests"]["delta_affine_fit_to_agent_delta"]["r2"],
    )
    print("Best isotropic delta scale:", summary["tests"]["delta_scalar_scale_fit"]["scale"])
    print("Best per-dim delta scales:", summary["tests"]["delta_per_dim_scale_fit"]["scales"])
    print("Best candidate scales by MSE:", [item["scale"] for item in candidate_delta_scales[:5]])


if __name__ == "__main__":
    main()
