#!/usr/bin/env python3
"""
Re-render pusht_expert.npz images at a higher resolution from stored states.

The diffusion-policy PushT expert set uses a fixed green-T goal at
PUSHT_FIXED_TARGET_POSE ([256, 256, pi/4]). Matching that goal — and avoiding
the fixed-target alignment wrapper, which warps expert states — is required for
faithful re-renders.

Preferred path (default): set each expert state into a PushT env and render at
the requested resolution. This keeps trajectories geometrically correct.

Alternative (--mode upsample): bilinear-upsample the original 96x96 frames.
Pixel-exact content, but soft / not a true high-res render.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_online.env_config import (  # noqa: E402
    DEFAULT_PUSHT_ENV_ID,
    PUSHT_FIXED_TARGET_POSE,
    make_pusht_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/expert_trajectories/pusht_expert.npz",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default="data/expert_trajectories/pusht_expert_224.npz",
    )
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument(
        "--mode",
        choices=("render", "upsample"),
        default="render",
        help="render=re-draw from states in env; upsample=resize original images.",
    )
    parser.add_argument(
        "--goal-pose",
        "--goal_pose",
        dest="goal_pose",
        type=float,
        nargs=3,
        default=PUSHT_FIXED_TARGET_POSE.tolist(),
        metavar=("X", "Y", "ANGLE"),
        help="Fixed PushT goal pose used by the expert dataset (default: 256 256 pi/4).",
    )
    parser.add_argument(
        "--verify-n",
        type=int,
        default=0,
        help="If >0, compare the first N re-rendered frames (downscaled) to the originals.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames (for debugging).",
    )
    return parser.parse_args()


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0.0, 255.0)
    if float(image.max()) <= 1.5:
        image = image * 255.0
    return image.astype(np.uint8)


def upsample_images(images: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    out = np.empty((len(images), height, width, 3), dtype=np.uint8)
    for i, image in enumerate(tqdm(images, desc="upsample")):
        frame = to_uint8_image(image)
        out[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return out


def make_goal_state(goal_pose: np.ndarray) -> np.ndarray:
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(3)
    # Agent pose is irrelevant for the green-T draw; block pose equals the goal.
    return np.array(
        [256.0, 256.0, goal_pose[0], goal_pose[1], goal_pose[2], 0.0, 0.0],
        dtype=np.float64,
    )


def render_images_from_states(
    states: np.ndarray,
    *,
    height: int,
    width: int,
    goal_pose: np.ndarray,
) -> np.ndarray:
    if height != width:
        raise ValueError(
            f"PushT uses a single square resolution; got height={height}, width={width}"
        )

    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(3)
    goal_state = make_goal_state(goal_pose)

    # Bare env: no fixed-target alignment (that rewrites expert states).
    env = make_pusht_env(
        env_id=DEFAULT_PUSHT_ENV_ID,
        render_mode="rgb_array",
        image_height=height,
        image_width=width,
        relative=False,
        sync_goal_pose=False,
        align_sampled_goal_to_fixed_target=False,
        render_obs=False,
        max_episode_steps=int(len(states) + 1),
    )
    env.reset(
        options={
            "state": np.asarray(states[0], dtype=np.float64),
            "goal_state": goal_state,
        }
    )
    unwrapped = env.unwrapped
    unwrapped.goal_pose = goal_pose.copy()
    unwrapped.goal_state = goal_state.copy()

    out = np.empty((len(states), height, width, 3), dtype=np.uint8)
    try:
        for i, state in enumerate(tqdm(states, desc="render")):
            unwrapped._set_state(np.asarray(state, dtype=np.float64))
            # Keep the green T pinned; variation resets can otherwise move it.
            unwrapped.goal_pose = goal_pose
            out[i] = np.asarray(unwrapped.render(), dtype=np.uint8)
    finally:
        env.close()
    return out


def verify_against_original(
    rendered: np.ndarray,
    original: np.ndarray,
    *,
    num_frames: int,
) -> None:
    import cv2

    n = min(int(num_frames), len(rendered), len(original))
    maes = []
    state_ok = True
    for i in range(n):
        gt = to_uint8_image(original[i])
        pred = rendered[i]
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        maes.append(float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))))
    print(
        f"verify: compared {n} frames | mean pixel MAE vs original "
        f"(after downscale if needed) = {float(np.mean(maes)):.4f}"
    )
    del state_ok


def main() -> None:
    args = parse_args()
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError("image_height/image_width must be positive")

    dataset = np.load(args.dataset)
    for key in ("states", "actions", "images", "episode_ends"):
        if key not in dataset:
            raise KeyError(f"{args.dataset} missing required key '{key}'")

    states = np.asarray(dataset["states"])
    actions = np.asarray(dataset["actions"])
    images = np.asarray(dataset["images"])
    episode_ends = np.asarray(dataset["episode_ends"])

    if len(states) != len(images) or len(states) != len(actions):
        raise ValueError(
            f"Length mismatch: states={len(states)} actions={len(actions)} images={len(images)}"
        )

    if args.max_frames is not None:
        n = max(0, int(args.max_frames))
        states = states[:n]
        actions = actions[:n]
        images = images[:n]
        episode_ends = episode_ends[episode_ends <= n]
        if len(episode_ends) == 0 or int(episode_ends[-1]) != n:
            episode_ends = np.concatenate([episode_ends, np.array([n], dtype=np.int64)])

    goal_pose = np.asarray(args.goal_pose, dtype=np.float64)
    print(
        f"Loaded {args.dataset} | frames={len(states)} episodes={len(episode_ends)} "
        f"mode={args.mode} resolution={args.image_height}x{args.image_width} "
        f"goal_pose={goal_pose.tolist()}"
    )

    if args.mode == "upsample":
        images_out = upsample_images(images, args.image_height, args.image_width)
    else:
        images_out = render_images_from_states(
            states,
            height=args.image_height,
            width=args.image_width,
            goal_pose=goal_pose,
        )

    if args.verify_n > 0:
        verify_against_original(images_out, images, num_frames=args.verify_n)

    output_path = Path(args.output_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        states=states,
        actions=actions,
        images=images_out,
        episode_ends=episode_ends,
    )
    print(
        f"Saved {output_path} | images={tuple(images_out.shape)} dtype={images_out.dtype}"
    )


if __name__ == "__main__":
    main()
