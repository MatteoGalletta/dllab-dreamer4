#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_online.env_config import ensure_swm_compat

ensure_swm_compat()

import stable_worldmodel as swm

from behavioural_cloning.eval_bc_exact import (
    BCImagePolicy,
    clean_state_dict_keys,
    infer_cnn_image_hw,
    save_video,
    resolve_model_config,
)
from ppo_online.model_paths import resolve_bc_prior_path, resolve_tokenizer_path
from ppo_online.render import extract_checkpoint_args, extract_checkpoint_state_dict, load_state_dict_safe
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt
from ppo_online.train import resolve_device


PUSHT_DATASET_DELTA_SCALE = 100.0
SWM_PUSHT_RELATIVE_ACTION_SCALE = 100.0


def _to_swm_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape[0] >= 7:
        return state[:7].copy()
    if state.shape[0] == 5:
        return np.concatenate([state, np.zeros(2, dtype=np.float64)])
    raise ValueError(f"Expected PushT state with 5 or 7 values, got shape {state.shape}")


def _install_pusht_goal_pose_setter():
    from stable_worldmodel.envs.pusht.env import PushT

    def _set_goal_state_and_pose(self, goal_state):
        goal_state = _to_swm_state(goal_state)
        current_state = None
        if hasattr(self, "_get_obs"):
            try:
                current_state = np.asarray(self._get_obs(), dtype=np.float64).copy()
            except Exception:
                current_state = None
        self._set_goal_state(goal_state)
        self.goal_pose = goal_state[2:5].copy()
        if hasattr(self, "_goal") and current_state is not None and hasattr(self, "_set_state"):
            try:
                self._set_state(goal_state)
                self._goal = self.render()
            finally:
                self._set_state(current_state)

    PushT._set_goal_state_and_pose = _set_goal_state_and_pose


def combine_world_panel_videos(video_dir: str | Path, output_path: str | Path, fps: float | None = None):
    import cv2

    video_dir = Path(video_dir)
    output_path = Path(output_path)
    video_paths = sorted(video_dir.glob("env_*.mp4"), key=lambda path: int(path.stem.split("_")[-1]))
    if not video_paths:
        print(f"No swm.World env_*.mp4 videos found in {video_dir}; skipping combine.")
        return None

    captures = [cv2.VideoCapture(str(path)) for path in video_paths]
    writer = None
    try:
        first_frames = []
        for capture, path in zip(captures, video_paths):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read first frame from {path}")
            first_frames.append(frame)

        tile_h, tile_w = first_frames[0].shape[:2]
        source_fps = captures[0].get(cv2.CAP_PROP_FPS) or 15.0
        output_fps = float(fps or source_fps)
        cols = math.ceil(math.sqrt(len(video_paths)))
        rows = math.ceil(len(video_paths) / cols)
        grid_w = cols * tile_w
        grid_h = rows * tile_h

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = output_path.with_name(f"{output_path.stem}.raw{output_path.suffix}")
        writer = None
        opened_codec = None
        for codec in ("mp4v", "avc1", "H264"):
            candidate = cv2.VideoWriter(
                str(temp_output_path),
                cv2.VideoWriter_fourcc(*codec),
                output_fps,
                (grid_w, grid_h),
            )
            if candidate.isOpened():
                writer = candidate
                opened_codec = codec
                break
            candidate.release()
        if writer is None:
            raise RuntimeError(f"Could not open video writer for {output_path}")

        last_frames = first_frames
        while True:
            canvas = np.full((grid_h, grid_w, 3), 250, dtype=np.uint8)
            for idx, frame in enumerate(last_frames):
                row, col = divmod(idx, cols)
                y0, x0 = row * tile_h, col * tile_w
                canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = frame
            writer.write(canvas)

            any_active = False
            next_frames = []
            for capture, last_frame in zip(captures, last_frames):
                ok, frame = capture.read()
                if ok:
                    any_active = True
                    next_frames.append(frame)
                else:
                    next_frames.append(last_frame)
            if not any_active:
                break
            last_frames = next_frames
    finally:
        for capture in captures:
            capture.release()
        if writer is not None:
            writer.release()

    finalized_codec = opened_codec
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        cmd = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temp_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            finalized_codec = "libx264"
            temp_output_path.unlink(missing_ok=True)
        else:
            temp_output_path.replace(output_path)
    else:
        temp_output_path.replace(output_path)

    print(f"Saved combined swm.World video to: {output_path} using codec={finalized_codec}")
    return output_path


def _overlay_goal_reference(frame: np.ndarray, goal_frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    goal_frame = np.asarray(goal_frame, dtype=np.uint8)
    if frame.shape != goal_frame.shape:
        return frame.copy()

    # Isolate the bright green target from the goal frame and paint it on top
    # of the live agent frame so goal orientation stays visible even when the
    # current block nearly occludes the target underneath.
    goal_rgb = goal_frame.astype(np.int16)
    mask = (
        (goal_rgb[..., 1] >= 150)
        & (goal_rgb[..., 1] >= goal_rgb[..., 0] + 25)
        & (goal_rgb[..., 1] >= goal_rgb[..., 2] + 25)
    )
    if not np.any(mask):
        return frame.copy()

    blended = frame.astype(np.float32).copy()
    green_overlay = goal_frame.astype(np.float32)
    blended[mask] = 0.35 * blended[mask] + 0.65 * green_overlay[mask]
    return np.clip(blended, 0, 255).astype(np.uint8)


def install_world_panel_video_patch():
    from stable_worldmodel.plot.video_utils import save_video
    import stable_worldmodel.world.world as swm_world_module
    from PIL import Image, ImageDraw, ImageFont

    def _patched_save_panel_videos(video_dir, panels, fps: int = 15) -> None:
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

        labels = list(panels)
        n_envs = len(panels[labels[0]])

        sample = np.asarray(panels[labels[0]][0])
        h, w = sample.shape[1:3] if sample.ndim == 4 else sample.shape[:2]
        n = len(labels)
        pad, gap, lh = max(12, w // 14), max(10, w // 16), max(22, w // 9)
        cw = (2 * pad + n * w + (n - 1) * gap + 15) // 16 * 16
        ch = (2 * pad + h + lh + 15) // 16 * 16
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, w // 14))
        except OSError:
            font = ImageFont.load_default()
        y_text = pad + h + max(8, lh // 4)

        for i in range(n_envs):
            env_panels = [np.asarray(panels[label][i]) for label in labels]
            panel_by_label = {label: panel for label, panel in zip(labels, env_panels)}
            goal_panel = panel_by_label.get("goal")
            if goal_panel is not None and "agent" in panel_by_label:
                goal_frame = goal_panel[0] if goal_panel.ndim == 4 else goal_panel
                agent_panel = panel_by_label["agent"]
                if agent_panel.ndim == 4:
                    panel_by_label["agent"] = np.stack(
                        [_overlay_goal_reference(frame, goal_frame) for frame in agent_panel],
                        axis=0,
                    )
                else:
                    panel_by_label["agent"] = _overlay_goal_reference(agent_panel, goal_frame)
            env_panels = [panel_by_label[label] for label in labels]

            T = max((len(p) for p in env_panels if p.ndim == 4), default=1)
            composed = []
            for t in range(T):
                canvas = np.full((ch, cw, 3), 250, dtype=np.uint8)
                for j, panel in enumerate(env_panels):
                    frame = panel[min(t, len(panel) - 1)] if panel.ndim == 4 else panel
                    x = pad + j * (w + gap)
                    canvas[pad : pad + h, x : x + w] = frame
                img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(img)
                for j, label in enumerate(labels):
                    bbox = draw.textbbox((0, 0), label, font=font)
                    x = pad + j * (w + gap) + w // 2 - (bbox[2] - bbox[0]) // 2
                    draw.text((x, y_text), label, fill=(130, 130, 130), font=font)
                composed.append(np.array(img))
            save_video(video_dir / f"env_{i}.mp4", composed, fps=fps)

    swm_world_module.save_panel_videos = _patched_save_panel_videos


class PushTH5WorldDataset:
    column_names = ("pixels", "state", "proprio", "action")

    def __init__(self, h5_path: str, image_size: tuple[int, int]):
        self.h5_path = str(h5_path)
        self.image_size = tuple(image_size)
        with h5py.File(self.h5_path, "r") as dataset:
            self.num_frames = int(dataset["pixels"].shape[0])
            self.episode_starts = np.asarray(dataset["ep_offset"][:], dtype=np.int64)
            self.episode_lens = np.asarray(dataset["ep_len"][:], dtype=np.int64)
            self.episode_ends = self.episode_starts + self.episode_lens

    def episode_length(self, episode_index: int) -> int:
        return int(self.episode_lens[int(episode_index)])

    def _resize_images(self, images: np.ndarray) -> np.ndarray:
        if images.shape[1:3] == self.image_size:
            return images
        import cv2

        return np.stack(
            [
                cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
                for image in images
            ],
            axis=0,
        )

    def load_chunk(self, episode_indices, start_steps, end_steps):
        chunks = []
        with h5py.File(self.h5_path, "r") as dataset:
            pixels_ds = dataset["pixels"]
            states_ds = dataset["state"]
            actions_ds = dataset["action"]
            for episode_index, start_step, end_step in zip(episode_indices, start_steps, end_steps):
                abs_start = int(self.episode_starts[int(episode_index)] + int(start_step))
                abs_end = int(self.episode_starts[int(episode_index)] + int(end_step))
                states = np.stack([_to_swm_state(state) for state in states_ds[abs_start:abs_end]], axis=0)
                images = self._resize_images(np.asarray(pixels_ds[abs_start:abs_end], dtype=np.uint8))
                actions = np.asarray(actions_ds[abs_start:abs_end], dtype=np.float32)
                proprio = np.concatenate([states[:, :2], states[:, -2:]], axis=-1)
                chunks.append(
                    {
                        "pixels": torch.as_tensor(images).permute(0, 3, 1, 2),
                        "state": states.copy(),
                        "proprio": proprio,
                        "action": actions.copy(),
                    }
                )
        return chunks


def sample_world_eval_starts(dataset: PushTH5WorldDataset, num_episodes: int, goal_offset_steps: int, seed: int):
    valid = []
    for episode_index in range(len(dataset.episode_ends)):
        max_start = dataset.episode_length(episode_index) - goal_offset_steps - 1
        for start_step in range(max_start + 1):
            valid.append((episode_index, start_step))
    if not valid:
        raise ValueError(f"No valid dataset starts for goal_offset_steps={goal_offset_steps}.")

    rng = np.random.default_rng(seed)
    replace = num_episodes > len(valid)
    sampled = rng.choice(len(valid), size=num_episodes, replace=replace)
    episode_indices, start_steps = zip(*(valid[int(idx)] for idx in sampled))
    return list(episode_indices), list(start_steps)


class BCWorldPolicy:
    def __init__(
        self,
        *,
        policy: BCImagePolicy,
        seq_len: int,
        frame_stride: int,
        action_chunk_size: int,
        device: torch.device,
        action_rescale_ratio: float,
    ):
        self.policy = policy
        self.seq_len = int(seq_len)
        self.frame_stride = max(1, int(frame_stride))
        self.action_chunk_size = int(action_chunk_size)
        self.device = device
        self.action_rescale_ratio = float(action_rescale_ratio)
        self.env = None
        self.frame_histories = None
        self.action_buffers = None

    def set_env(self, env):
        self.env = env
        max_history_len = (self.seq_len - 1) * self.frame_stride + 1
        self.frame_histories = [deque(maxlen=max_history_len) for _ in range(env.num_envs)]
        self.action_buffers = [deque() for _ in range(env.num_envs)]

    def _stack_history(self, env_index: int) -> np.ndarray:
        history = list(self.frame_histories[env_index])
        newest = len(history) - 1
        indices = [max(0, newest - i * self.frame_stride) for i in range(self.seq_len - 1, -1, -1)]
        return np.stack([history[idx] for idx in indices], axis=0)

    def get_action(self, info_dict, **kwargs):
        del kwargs
        if self.env is None:
            raise RuntimeError("BCWorldPolicy.set_env must be called before get_action")

        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None:
            needs_flush = np.asarray(needs_flush).reshape(-1)
            for env_index, should_flush in enumerate(needs_flush):
                if should_flush:
                    self.frame_histories[env_index].clear()
                    self.action_buffers[env_index].clear()

        pixels = np.asarray(info_dict["pixels"])[:, -1]
        actions = []
        for env_index in range(self.env.num_envs):
            self.frame_histories[env_index].append(np.asarray(pixels[env_index], dtype=np.uint8))
            if not self.action_buffers[env_index]:
                stacked_frames = self._stack_history(env_index)
                input_tensor = torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=self.device)
                with torch.no_grad():
                    action_chunk = self.policy.predict_action_chunk(input_tensor).squeeze(0).cpu().numpy()
                action_chunk = action_chunk * self.action_rescale_ratio
                self.action_buffers[env_index].extend(action_chunk)
            actions.append(self.action_buffers[env_index].popleft())
        return np.asarray(actions, dtype=np.float32)


class DatasetOracleWorldPolicy:
    def __init__(self, action_sequences: list[np.ndarray], action_rescale_ratio: float):
        self.env = None
        self.action_sequences = [np.asarray(sequence, dtype=np.float32) for sequence in action_sequences]
        self.action_buffers = None
        self.action_rescale_ratio = float(action_rescale_ratio)

    def set_env(self, env):
        self.env = env
        self.action_buffers = []
        for sequence in self.action_sequences[: env.num_envs]:
            scaled_sequence = np.asarray(sequence, dtype=np.float32) * self.action_rescale_ratio
            self.action_buffers.append(deque(np.asarray(action, dtype=np.float32) for action in scaled_sequence))
        while len(self.action_buffers) < env.num_envs:
            self.action_buffers.append(deque())

    def get_action(self, info_dict, **kwargs):
        del info_dict, kwargs
        if self.env is None:
            raise RuntimeError("DatasetOracleWorldPolicy.set_env must be called before get_action")
        actions = []
        for env_index in range(self.env.num_envs):
            if not self.action_buffers[env_index]:
                actions.append(np.zeros(2, dtype=np.float32))
                continue
            actions.append(np.asarray(self.action_buffers[env_index].popleft(), dtype=np.float32))
        return np.asarray(actions, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC checkpoint through swm.World like the other group's BC eval.")
    parser.add_argument("--checkpoint", type=str, default="local_models/behavior_cloning/bc_best.pt")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the PushT HDF5 dataset used for eval starts/goals.")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument(
        "--dataset-action-scale",
        type=float,
        default=PUSHT_DATASET_DELTA_SCALE,
        help="Pixels-per-unit implied by dataset relative actions.",
    )
    parser.add_argument(
        "--world-action-scale",
        type=float,
        default=SWM_PUSHT_RELATIVE_ACTION_SCALE,
        help="Pixels-per-unit used internally by the swm/PushT-v1 relative action space.",
    )
    parser.add_argument(
        "--policy-source",
        choices=("bc", "dataset_oracle"),
        default="bc",
        help="Use the BC checkpoint or replay ground-truth dataset actions under the same world eval setup.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device("auto")
    checkpoint_path = resolve_bc_prior_path(args.checkpoint)
    tokenizer_path = None
    image_hw = (224, 224)
    if args.policy_source == "bc":
        payload = load_state_dict_safe(checkpoint_path, torch.device("cpu"))
        state_dict = extract_checkpoint_state_dict(payload)
        if state_dict is None:
            raise ValueError(f"Unsupported BC checkpoint format in {checkpoint_path}")
        cleaned_state = clean_state_dict_keys(state_dict)
        ckpt_args = extract_checkpoint_args(payload)
        model_cfg = resolve_model_config(ckpt_args, cleaned_state)

        if model_cfg["tokenizer_name"] is not None:
            tokenizer_path = resolve_tokenizer_path(args.tokenizer_path or str(model_cfg["tokenizer_name"]))
            _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
            image_hw = (int(tokenizer_info["H"]), int(tokenizer_info["W"]))
        else:
            image_hw = infer_cnn_image_hw(model_cfg)

        policy = BCImagePolicy(
            image_shape=(image_hw[0], image_hw[1], 3),
            hidden_dim=model_cfg["hidden_dim"],
            dropout=model_cfg["dropout"],
            action_chunk_size=model_cfg["action_chunk_size"],
            seq_len=model_cfg["seq_len"],
            tokenizer_ckpt=tokenizer_path,
            tokenizer_feature_dim=model_cfg["tokenizer_feature_dim"],
            policy_style=model_cfg["policy_style"],
            temporal_layers=model_cfg["temporal_layers"],
            temporal_heads=model_cfg["temporal_heads"],
            temporal_context=model_cfg["temporal_context"],
            backbone_device=device,
            action_output_tanh=model_cfg["action_output_tanh"],
        ).to(device)
        policy.load_state_dict(cleaned_state, strict=True)
        policy.eval()
    else:
        model_cfg = {
            "seq_len": 1,
            "action_chunk_size": 1,
            "hidden_dim": 0,
            "dropout": 0.0,
            "temporal_layers": 0,
            "temporal_heads": 0,
            "tokenizer_name": None,
        }
        policy = None

    world_dataset = PushTH5WorldDataset(args.dataset, image_size=image_hw)
    episode_indices, start_steps = sample_world_eval_starts(
        world_dataset,
        args.episodes,
        args.goal_offset_steps,
        args.seed,
    )
    _install_pusht_goal_pose_setter()
    install_world_panel_video_patch()

    print(
        f"Using swm.World.evaluate(dataset=...) for policy_source={args.policy_source}: "
        f"episodes={args.episodes}, goal_offset_steps={args.goal_offset_steps}, "
        f"eval_budget={args.eval_budget}, seed={args.seed}."
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"tokenizer={tokenizer_path}")
    print(f"sampled episode indices={episode_indices}")
    print(f"sampled start steps={start_steps}")
    dataset_action_scale = float(
        args.dataset_action_scale
        if args.dataset_action_scale != PUSHT_DATASET_DELTA_SCALE or args.policy_source != "bc"
        else model_cfg.get("swm_action_scale", PUSHT_DATASET_DELTA_SCALE)
        if model_cfg.get("action_mode") == "swm_relative"
        else PUSHT_DATASET_DELTA_SCALE
    )
    action_rescale_ratio = dataset_action_scale / float(args.world_action_scale)
    print(
        f"relative action rescale ratio={action_rescale_ratio:.4f} "
        f"(dataset_scale={dataset_action_scale:.3f} / world_scale={float(args.world_action_scale):.3f})"
    )

    combine_world_video = args.video_path is not None and Path(args.video_path).suffix.lower() == ".mp4"
    with tempfile.TemporaryDirectory(prefix="pusht-world-video-") as tmp_video_dir:
        world_video_path = tmp_video_dir if combine_world_video else args.video_path
        if combine_world_video:
            print(f"Writing swm.World per-env videos to a temp dir before combining into {args.video_path}.")

        world = swm.World(
            "swm/PushT-v1",
            num_envs=args.episodes,
            image_shape=image_hw,
            max_episode_steps=2 * args.eval_budget,
        )
        if args.policy_source == "bc":
            world_policy = BCWorldPolicy(
                policy=policy,
                seq_len=model_cfg["seq_len"],
                frame_stride=model_cfg["frame_stride"],
                action_chunk_size=model_cfg["action_chunk_size"],
                device=device,
                action_rescale_ratio=action_rescale_ratio,
            )
        else:
            oracle_chunks = world_dataset.load_chunk(
                episode_indices,
                start_steps,
                [start_step + args.goal_offset_steps for start_step in start_steps],
            )
            oracle_action_sequences = [np.asarray(chunk["action"], dtype=np.float32) for chunk in oracle_chunks]
            world_policy = DatasetOracleWorldPolicy(
                action_sequences=oracle_action_sequences,
                action_rescale_ratio=action_rescale_ratio,
            )
        world.set_policy(world_policy)
        try:
            metrics = world.evaluate(
                dataset=world_dataset,
                episodes_idx=episode_indices,
                start_steps=start_steps,
                goal_offset=args.goal_offset_steps,
                eval_budget=args.eval_budget,
                callables=[
                    {
                        "method": "_set_state",
                        "args": {"state": {"value": "state"}},
                    },
                    {
                        "method": "_set_goal_state_and_pose",
                        "args": {"goal_state": {"value": "goal_state"}},
                    },
                ],
                video=world_video_path,
            )
        finally:
            world.close()

        if combine_world_video:
            combine_world_panel_videos(world_video_path, args.video_path)

    print(f"swm.World metrics: {metrics}")
    world_success_rate = float(metrics.get("success_rate", 0.0))
    print(f"World success rate: {world_success_rate:.2f}%")
    print(f"Success rate: {world_success_rate / 100.0:.4f}")


if __name__ == "__main__":
    main()
