#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, make_pusht_env, resolve_pusht_env_id
from ppo_online.model_paths import resolve_tokenizer_path
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Compare BC dataset frames against live PushT env renders.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the BC HDF5 dataset.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="debug/dataset_env_compare",
        help="Directory where comparison images are saved.",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Number of comparisons to save.")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional tokenizer checkpoint path used to determine render resolution.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check-action-semantics",
        action="store_true",
        help="Also test whether dataset actions reproduce the next state/frame under different SWM action interpretations.",
    )
    return parser.parse_args()


def add_label(image_rgb: np.ndarray, text: str) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    canvas = np.full((image_bgr.shape[0] + 28, image_bgr.shape[1], 3), 255, dtype=np.uint8)
    canvas[28:] = image_bgr
    cv2.putText(canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def make_env(image_hw: tuple[int, int], seed: int, relative: bool | None = None):
    resolved_env_id = resolve_pusht_env_id(DEFAULT_PUSHT_ENV_ID)
    env = make_pusht_env(
        env_id=resolved_env_id,
        render_mode="rgb_array",
        image_height=int(image_hw[0]),
        image_width=int(image_hw[1]),
        relative=bool(relative) if relative is not None else False,
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=True,
        render_obs=False,
    )
    env.reset(seed=seed)
    return env, resolved_env_id


def try_reset_to_dataset_state(env, state: np.ndarray, env_id: str):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 5:
        return False

    del env_id
    option_candidates = [
        {"state": state.tolist()},
        {"state": state[:5].tolist()},
    ]

    seen = set()
    for options in option_candidates:
        key = json.dumps(options, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            env.reset(options=options)
            return True
        except Exception:
            continue
    return False


def compute_state_l2(pred_state: np.ndarray | None, target_state: np.ndarray | None) -> float | None:
    if pred_state is None or target_state is None:
        return None
    pred_state = np.asarray(pred_state, dtype=np.float32).reshape(-1)
    target_state = np.asarray(target_state, dtype=np.float32).reshape(-1)
    if pred_state.shape != target_state.shape:
        return None
    return float(np.linalg.norm(pred_state - target_state))


def compute_pixel_mae(pred_frame: np.ndarray | None, target_frame: np.ndarray | None) -> float | None:
    if pred_frame is None or target_frame is None:
        return None
    pred_frame = np.asarray(pred_frame, dtype=np.float32)
    target_frame = np.asarray(target_frame, dtype=np.float32)
    if pred_frame.shape != target_frame.shape:
        return None
    return float(np.mean(np.abs(pred_frame - target_frame)))


def step_from_dataset_action(
    env,
    env_id: str,
    state: np.ndarray,
    action: np.ndarray,
    image_hw: tuple[int, int],
):
    if not try_reset_to_dataset_state(env, state, env_id):
        return None

    obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
    frame = np.asarray(env.render(), dtype=np.uint8)
    if frame.shape[:2] != image_hw:
        frame = cv2.resize(frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)

    next_state = None
    if isinstance(obs, dict) and "state" in obs:
        next_state = np.asarray(obs["state"], dtype=np.float32)
    elif obs is not None:
        next_state = np.asarray(obs, dtype=np.float32)

    return {
        "frame": frame,
        "state": next_state,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    _, tok_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
    image_hw = (int(tok_info["H"]), int(tok_info["W"]))

    with h5py.File(args.dataset, "r") as dataset:
        pixels = dataset["pixels"]
        states = dataset["state"] if "state" in dataset else None
        actions = dataset["action"] if "action" in dataset else None
        num_frames = int(pixels.shape[0])
        num_samples = min(int(args.num_samples), num_frames)
        sample_indices = np.linspace(0, num_frames - 1, num=num_samples, dtype=int)
        matched_resets = 0

        env, resolved_env_id = make_env(image_hw=image_hw, seed=args.seed)
        semantic_envs: dict[str, gym.Env] = {}
        semantic_metrics: dict[str, list[dict[str, float]]] = {}
        if args.check_action_semantics and actions is not None and states is not None:
            semantic_envs = {
                "raw_relative": make_env(image_hw=image_hw, seed=args.seed, relative=True)[0],
                "raw_absolute": make_env(image_hw=image_hw, seed=args.seed, relative=False)[0],
                "scaled_absolute": make_env(image_hw=image_hw, seed=args.seed, relative=False)[0],
            }
            semantic_metrics = {name: [] for name in semantic_envs}

        for sample_no, frame_idx in enumerate(sample_indices):
            dataset_frame = np.asarray(pixels[frame_idx], dtype=np.uint8)
            if dataset_frame.shape[:2] != image_hw:
                dataset_frame = cv2.resize(dataset_frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)

            matched_state = False
            if states is not None:
                matched_state = try_reset_to_dataset_state(env, np.asarray(states[frame_idx]), resolved_env_id)
                matched_resets += int(matched_state)
            if not matched_state:
                env.reset(seed=args.seed + sample_no)

            env_frame = np.asarray(env.render(), dtype=np.uint8)
            if env_frame.shape[:2] != image_hw:
                env_frame = cv2.resize(env_frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)

            left = add_label(dataset_frame, f"dataset idx={frame_idx}")
            right = add_label(env_frame, f"env render matched={matched_state}")
            comparison = np.concatenate([left, right], axis=1)
            save_rgb(outdir / f"compare_{sample_no:02d}_idx_{frame_idx:07d}.png", comparison)

            if semantic_envs and frame_idx + 1 < num_frames:
                dataset_next_frame = np.asarray(pixels[frame_idx + 1], dtype=np.uint8)
                if dataset_next_frame.shape[:2] != image_hw:
                    dataset_next_frame = cv2.resize(
                        dataset_next_frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR
                    )
                dataset_state = np.asarray(states[frame_idx], dtype=np.float32)
                dataset_next_state = np.asarray(states[frame_idx + 1], dtype=np.float32)
                dataset_action = np.asarray(actions[frame_idx], dtype=np.float32)
                action_variants = {
                    "raw_relative": dataset_action,
                    "raw_absolute": dataset_action,
                    "scaled_absolute": np.clip((dataset_action + 1.0) * 256.0, 0.0, 512.0),
                }
                semantic_panels = [
                    add_label(dataset_frame, f"dataset idx={frame_idx}"),
                    add_label(dataset_next_frame, f"dataset next idx={frame_idx + 1}"),
                ]

                for mode_name, semantic_env in semantic_envs.items():
                    result = step_from_dataset_action(
                        semantic_env,
                        resolved_env_id,
                        dataset_state,
                        action_variants[mode_name],
                        image_hw,
                    )
                    if result is None:
                        continue
                    state_l2 = compute_state_l2(result["state"], dataset_next_state)
                    pixel_mae = compute_pixel_mae(result["frame"], dataset_next_frame)
                    metric_entry = {}
                    if state_l2 is not None:
                        metric_entry["state_l2"] = state_l2
                    if pixel_mae is not None:
                        metric_entry["pixel_mae"] = pixel_mae
                    semantic_metrics[mode_name].append(metric_entry)

                    label = mode_name
                    if state_l2 is not None:
                        label += f" state_l2={state_l2:.2f}"
                    if pixel_mae is not None:
                        label += f" mae={pixel_mae:.2f}"
                    semantic_panels.append(add_label(result["frame"], label))

                if len(semantic_panels) > 2:
                    semantics_image = np.concatenate(semantic_panels, axis=1)
                    save_rgb(outdir / f"action_semantics_{sample_no:02d}_idx_{frame_idx:07d}.png", semantics_image)

        env.close()
        for semantic_env in semantic_envs.values():
            semantic_env.close()

        summary = {
            "dataset_path": str(Path(args.dataset).resolve()),
            "tokenizer_checkpoint": tokenizer_path,
            "resolved_env_id": resolved_env_id,
            "tokenizer_resolution": {"H": image_hw[0], "W": image_hw[1]},
            "keys": sorted(list(dataset.keys())),
            "pixels_shape": list(pixels.shape),
            "states_shape": list(states.shape) if states is not None else None,
            "actions_shape": list(actions.shape) if actions is not None else None,
            "num_samples": num_samples,
            "matched_resets": matched_resets,
        }
        if semantic_metrics:
            summary["action_semantics"] = {}
            for mode_name, entries in semantic_metrics.items():
                if not entries:
                    continue
                state_l2_vals = [entry["state_l2"] for entry in entries if "state_l2" in entry]
                pixel_mae_vals = [entry["pixel_mae"] for entry in entries if "pixel_mae" in entry]
                summary["action_semantics"][mode_name] = {
                    "num_evaluated": len(entries),
                    "mean_state_l2": float(np.mean(state_l2_vals)) if state_l2_vals else None,
                    "mean_pixel_mae": float(np.mean(pixel_mae_vals)) if pixel_mae_vals else None,
                }
        if actions is not None:
            action_sample = np.asarray(actions[: min(10000, actions.shape[0])], dtype=np.float32)
            summary["action_stats"] = {
                "min": action_sample.min(axis=0).tolist(),
                "max": action_sample.max(axis=0).tolist(),
                "mean": action_sample.mean(axis=0).tolist(),
            }
        if states is not None:
            state_sample = np.asarray(states[: min(10000, states.shape[0])], dtype=np.float32)
            summary["state_stats"] = {
                "min": state_sample.min(axis=0).tolist(),
                "max": state_sample.max(axis=0).tolist(),
                "mean": state_sample.mean(axis=0).tolist(),
            }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved {num_samples} comparison images to {outdir}")
    print(f"Tokenizer resolution: {image_hw[0]}x{image_hw[1]}")
    print(f"Tokenizer checkpoint: {tokenizer_path}")
    print(f"Matched dataset state resets: {matched_resets}/{num_samples}")
    if semantic_metrics:
        print("Action semantics summary:")
        for mode_name, entries in semantic_metrics.items():
            if not entries:
                continue
            state_l2_vals = [entry["state_l2"] for entry in entries if "state_l2" in entry]
            pixel_mae_vals = [entry["pixel_mae"] for entry in entries if "pixel_mae" in entry]
            state_text = "n/a" if not state_l2_vals else f"{float(np.mean(state_l2_vals)):.3f}"
            pixel_text = "n/a" if not pixel_mae_vals else f"{float(np.mean(pixel_mae_vals)):.3f}"
            print(f"  {mode_name}: mean_state_l2={state_text} mean_pixel_mae={pixel_text}")
    print(f"Summary written to {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
