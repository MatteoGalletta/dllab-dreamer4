#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import gym_pusht
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

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


def make_env(image_hw: tuple[int, int], seed: int):
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
        observation_width=int(image_hw[1]),
        observation_height=int(image_hw[0]),
    )
    env.reset(seed=seed)
    return env


def try_reset_to_dataset_state(env, state: np.ndarray):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 5:
        return False
    try:
        env.reset(options={"reset_to_state": state[:5].tolist()})
        return True
    except Exception:
        return False


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

        env = make_env(image_hw=image_hw, seed=args.seed)

        for sample_no, frame_idx in enumerate(sample_indices):
            dataset_frame = np.asarray(pixels[frame_idx], dtype=np.uint8)
            if dataset_frame.shape[:2] != image_hw:
                dataset_frame = cv2.resize(dataset_frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)

            matched_state = False
            if states is not None:
                matched_state = try_reset_to_dataset_state(env, np.asarray(states[frame_idx]))
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

        env.close()

        summary = {
            "dataset_path": str(Path(args.dataset).resolve()),
            "tokenizer_checkpoint": tokenizer_path,
            "tokenizer_resolution": {"H": image_hw[0], "W": image_hw[1]},
            "keys": sorted(list(dataset.keys())),
            "pixels_shape": list(pixels.shape),
            "states_shape": list(states.shape) if states is not None else None,
            "actions_shape": list(actions.shape) if actions is not None else None,
            "num_samples": num_samples,
            "matched_resets": matched_resets,
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
    print(f"Summary written to {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
