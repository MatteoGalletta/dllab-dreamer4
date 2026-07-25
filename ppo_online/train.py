#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import gymnasium as gym
import gym_pusht
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from gymnasium.wrappers import FrameStackObservation

from .env_config import DEFAULT_PUSHT_ENV_ID, make_pusht_env, resolve_pusht_env_id
from .agent import PPOAgent
from .buffer import PPOVectorBuffer
from .model_paths import resolve_bc_prior_path, resolve_ppo_checkpoint_path, resolve_tokenizer_path
from .tokenizer_utils import TokenizerZEncoder, load_tokenizer_from_ckpt


def pool_latents(z_unpacked: torch.Tensor) -> torch.Tensor:
    """Collapses spatial patches by taking their spatial token mean."""
    return z_unpacked.mean(dim=-2)


class LatentContextSampler:
    def __init__(self, npz_path: str, action_chunk_size: int, device: torch.device):
        self.device = device
        self.action_chunk_size = action_chunk_size

        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]
        self.actions_raw = data["actions"]

        ends = data["episode_ends"]
        starts = np.zeros_like(ends)
        starts[1:] = ends[:-1]

        self.episodes = []
        for s, e in zip(starts, ends):
            raw_len = e - s
            seq_len = raw_len // action_chunk_size
            used = seq_len * action_chunk_size

            if seq_len >= 24:
                ep_imgs = self.images[s:s + used][::action_chunk_size]
                ep_acts = self.actions_raw[s:s + used].reshape(seq_len, -1)
                self.episodes.append((ep_imgs, ep_acts))

    def sample_context(self, batch_size: int, ctx_len: int = 24):
        batch_imgs, batch_acts = [], []
        indices = np.random.choice(len(self.episodes), size=batch_size, replace=True)

        for idx in indices:
            ep_imgs, ep_acts = self.episodes[idx]
            t_start = np.random.randint(0, len(ep_imgs) - ctx_len)

            batch_imgs.append(ep_imgs[t_start: t_start + ctx_len])
            batch_acts.append(ep_acts[t_start: t_start + ctx_len])

        imgs_tensor = torch.from_numpy(np.stack(batch_imgs)).permute(0, 1, 4, 2, 3).float() / 255.0
        acts_tensor = torch.from_numpy(np.stack(batch_acts)).float()

        return imgs_tensor.to(self.device), acts_tensor.to(self.device)


class StridedObservationStackWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, stack_size: int, frame_stride: int):
        super().__init__(env)
        self.stack_size = int(stack_size)
        self.frame_stride = max(1, int(frame_stride))
        self.history: deque[np.ndarray] = deque()
        base_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.repeat(np.expand_dims(base_space.low, axis=0), self.stack_size, axis=0),
            high=np.repeat(np.expand_dims(base_space.high, axis=0), self.stack_size, axis=0),
            dtype=base_space.dtype,
        )

    def _stack_history(self) -> np.ndarray:
        history = list(self.history)
        newest = len(history) - 1
        indices = [max(0, newest - i * self.frame_stride) for i in range(self.stack_size - 1, -1, -1)]
        return np.stack([history[idx] for idx in indices], axis=0)

    def observation(self, observation):
        self.history.append(np.asarray(observation, dtype=self.observation_space.dtype))
        return self._stack_history()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.append(np.asarray(observation, dtype=self.observation_space.dtype))
        return self._stack_history(), info


class ActionChunkingTemporalEnsembleWrapper(gym.Wrapper):
    """
    ACT-style action chunking:
    the policy predicts a horizon of relative deltas every step, and we execute
    a temporally ensembled primitive delta from the overlapping predictions.
    """

    def __init__(
        self,
        env: gym.Env,
        chunk_size: int = 5,
        max_step_pixels: float = 15.0,
        ensemble_decay: float = 0.5,
    ):
        super().__init__(env)
        self.chunk_size = chunk_size
        self.max_step_pixels = max_step_pixels
        self.ensemble_decay = ensemble_decay

        low = np.full((chunk_size * 2,), -1.0, dtype=np.float32)
        high = np.full((chunk_size * 2,), 1.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_eef = np.array([256.0, 256.0], dtype=np.float32)
        self.pending_chunks: deque[dict[str, Any]] = deque()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = extract_state_array(obs)
        self.current_eef = state[0:2].copy()
        self.pending_chunks.clear()
        return obs, info

    def _ensemble_delta(self) -> np.ndarray:
        candidates = []
        weights = []
        for age, entry in enumerate(reversed(self.pending_chunks)):
            offset = entry["offset"]
            if offset >= self.chunk_size:
                continue
            candidates.append(entry["chunk"][offset])
            weights.append(math.exp(-self.ensemble_decay * age))

        if not candidates:
            return np.zeros(2, dtype=np.float32)

        weights_np = np.asarray(weights, dtype=np.float32)
        weights_np /= max(weights_np.sum(), 1e-8)
        return np.sum(np.asarray(candidates, dtype=np.float32) * weights_np[:, None], axis=0)

    def step(self, macro_action):
        macro_action = np.asarray(macro_action, dtype=np.float32)
        macro_action = np.clip(macro_action, -1.0, 1.0).reshape(self.chunk_size, 2)
        self.pending_chunks.append({"chunk": macro_action * self.max_step_pixels, "offset": 0})

        delta = self._ensemble_delta()
        target_pos = np.clip(self.current_eef + delta, 0.0, 512.0)

        obs, reward, terminated, truncated, info = self.env.step(target_pos)
        state = extract_state_array(obs)
        self.current_eef = state[0:2].copy()

        for entry in self.pending_chunks:
            entry["offset"] += 1
        while self.pending_chunks and self.pending_chunks[0]["offset"] >= self.chunk_size:
            self.pending_chunks.popleft()

        info = dict(info)
        info["executed_delta"] = delta.astype(np.float32)
        info["num_active_chunks"] = len(self.pending_chunks)

        return obs, reward, terminated, truncated, info


class OpenLoopChunkExecutionWrapper(gym.Wrapper):
    """
    Execute a full predicted action chunk open-loop inside one PPO transition.

    This is closer to the offline-rl-with-le-wm setup where one PPO step
    corresponds to one chunk decision and the env advances ``chunk_size`` real
    steps internally.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        chunk_size: int,
        gamma: float,
        action_mode: str,
        action_output_tanh: bool = False,
        return_rendered_history: bool = False,
        target_height: int | None = None,
        target_width: int | None = None,
        stack_size: int = 1,
        frame_stride: int = 1,
    ):
        super().__init__(env)
        self.chunk_size = int(chunk_size)
        self.gamma = float(gamma)
        self.action_mode = str(action_mode)
        self.action_output_tanh = bool(action_output_tanh)
        self.return_rendered_history = bool(return_rendered_history)
        self.target_height = None if target_height is None else int(target_height)
        self.target_width = None if target_width is None else int(target_width)
        self.stack_size = max(1, int(stack_size))
        self.frame_stride = max(1, int(frame_stride))
        self.frame_history: deque[np.ndarray] | None = None

        if self.action_mode in {"relative", "swm_relative"} or (
            self.action_mode == "absolute" and self.action_output_tanh
        ):
            low = np.full((self.chunk_size * 2,), -1.0, dtype=np.float32)
            high = np.full((self.chunk_size * 2,), 1.0, dtype=np.float32)
        else:
            low = np.full((self.chunk_size * 2,), 0.0, dtype=np.float32)
            high = np.full((self.chunk_size * 2,), 512.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        if self.return_rendered_history:
            if self.target_height is None or self.target_width is None:
                raise ValueError("Rendered history mode requires target_height and target_width.")
            self.frame_history = deque(maxlen=(self.stack_size - 1) * self.frame_stride + 1)
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.stack_size, self.target_height, self.target_width, 3),
                dtype=np.uint8,
            )

    def _render_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("Expected renderable RGB frame for chunk history observations.")
        frame = np.asarray(frame, dtype=np.uint8)
        if (
            self.target_height is not None
            and self.target_width is not None
            and frame.shape[:2] != (self.target_height, self.target_width)
        ):
            frame_tensor = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            frame_tensor = F.interpolate(
                frame_tensor,
                size=(self.target_height, self.target_width),
                mode="bilinear",
                align_corners=False,
            )
            frame = frame_tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
        return frame

    def _append_current_frame(self) -> None:
        if self.frame_history is not None:
            self.frame_history.append(self._render_frame())

    def _stack_frame_history(self) -> np.ndarray:
        if self.frame_history is None or not self.frame_history:
            raise RuntimeError("Frame history is empty; call reset() before stepping.")
        history = list(self.frame_history)
        newest = len(history) - 1
        indices = [max(0, newest - i * self.frame_stride) for i in range(self.stack_size - 1, -1, -1)]
        return np.stack([history[idx] for idx in indices], axis=0)

    def _primitive_to_env_action(self, primitive: np.ndarray) -> np.ndarray:
        primitive = np.asarray(primitive, dtype=np.float32)
        if self.action_mode in {"relative", "swm_relative"}:
            return np.clip(primitive, -1.0, 1.0)
        if self.action_mode == "absolute" and self.action_output_tanh:
            return np.clip((primitive + 1.0) * 256.0, 0.0, 512.0)
        return np.clip(primitive, 0.0, 512.0)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.frame_history is not None:
            self.frame_history.clear()
            self._append_current_frame()
            return self._stack_frame_history(), info
        return observation, info

    def step(self, macro_action):
        macro_action = np.asarray(macro_action, dtype=np.float32).reshape(self.chunk_size, 2)
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        executed_actions = []

        for j, primitive in enumerate(macro_action):
            env_action = self._primitive_to_env_action(primitive)
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            total_reward += (self.gamma**j) * float(reward)
            executed_actions.append(env_action.astype(np.float32))
            self._append_current_frame()
            if terminated or truncated:
                break

        info = dict(info)
        info["executed_actions"] = np.asarray(executed_actions, dtype=np.float32)
        info["executed_primitives"] = np.asarray(macro_action[: len(executed_actions)], dtype=np.float32)
        info["num_executed_primitives"] = len(executed_actions)
        if self.frame_history is not None:
            obs = self._stack_frame_history()
        return obs, float(total_reward), terminated, truncated, info


class RunningMeanStd:
    """Welford-style running mean/variance shared across vector env rewards."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count


class RewardNormalizer:
    """Normalize chunk rewards by the running std of discounted returns."""

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, epsilon: float = 1e-8):
        self.rms = RunningMeanStd(shape=())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.epsilon = float(epsilon)

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=np.float64)
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        out = rewards / np.sqrt(self.rms.var + self.epsilon)
        self.returns = self.returns * (1.0 - dones)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray | float]:
        return {"var": self.rms.var, "mean": self.rms.mean, "count": self.rms.count}


class PushTDenseRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, env_id: str | None = None):
        super().__init__(env)
        self.target_pos = np.array([256.0, 256.0], dtype=np.float32)
        self.env_id = env_id or getattr(getattr(env, "spec", None), "id", "") or ""
        self.base_env = self._find_pusht_base_env()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _find_pusht_base_env(self):
        env = self.env
        visited = set()
        while env is not None and id(env) not in visited:
            visited.add(id(env))
            if hasattr(env, "window_size") and (
                hasattr(env, "block")
                or hasattr(env, "_setup")
                or hasattr(env, "_get_info")
            ):
                return env
            env = getattr(env, "env", None)
        return None

    def _rasterize_block_mask(self, pose: np.ndarray) -> np.ndarray | None:
        if self.base_env is None:
            self.base_env = self._find_pusht_base_env()
        if self.base_env is None or not hasattr(self.base_env, "block"):
            return None

        canvas_size = int(getattr(self.base_env, "window_size", 512))
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        position = np.asarray(pose[:2], dtype=np.float32)
        angle = float(pose[2])
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        for shape in self.base_env.block.shapes:
            if hasattr(shape, "get_vertices"):
                vertices = np.asarray([[vertex.x, vertex.y] for vertex in shape.get_vertices()], dtype=np.float32)
                world_vertices = (vertices @ rotation.T) + position[None, :]
                polygon = np.round(world_vertices).astype(np.int32)
                cv2.fillPoly(mask, [polygon], 255)
            elif hasattr(shape, "radius") and hasattr(shape, "offset"):
                offset = np.asarray([shape.offset.x, shape.offset.y], dtype=np.float32)
                center = (rotation @ offset) + position
                cv2.circle(mask, tuple(np.round(center).astype(np.int32)), int(round(float(shape.radius))), 255, -1)

        return mask

    def _compute_visual_coverage(self, info: dict) -> float | None:
        goal_pose = info.get("goal_pose")
        block_pose = info.get("block_pose")
        if goal_pose is None or block_pose is None:
            return None

        goal_mask = self._rasterize_block_mask(np.asarray(goal_pose, dtype=np.float32))
        block_mask = self._rasterize_block_mask(np.asarray(block_pose, dtype=np.float32))
        if goal_mask is None or block_mask is None:
            return None

        goal_pixels = goal_mask > 0
        if not np.any(goal_pixels):
            return None

        overlap_pixels = goal_pixels & (block_mask > 0)
        return float(overlap_pixels.sum() / max(1, goal_pixels.sum()))

    def _resolve_goal_block_pose(self, info: dict, state: np.ndarray) -> np.ndarray | None:
        goal_pose = info.get("goal_pose")
        if goal_pose is not None:
            goal_pose = np.asarray(goal_pose, dtype=np.float32).reshape(-1)
            if goal_pose.shape[0] >= 3:
                return goal_pose[:3]

        goal_state = info.get("goal_state")
        if goal_state is not None:
            goal_state = np.asarray(goal_state, dtype=np.float32).reshape(-1)
            if goal_state.shape[0] >= 5:
                return np.array([goal_state[2], goal_state[3], goal_state[4]], dtype=np.float32)

        if state.shape[0] >= 5:
            return np.array([self.target_pos[0], self.target_pos[1], state[4]], dtype=np.float32)
        return None

    def _angle_distance(self, angle_a: float, angle_b: float) -> float:
        diff = abs(angle_a - angle_b) % (2.0 * math.pi)
        return min(diff, (2.0 * math.pi) - diff)

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)

        state = extract_state_array(observation)
        eef_pos = state[0:2].astype(np.float32)
        block_pos = state[2:4].astype(np.float32)
        block_angle = float(state[4]) if state.shape[0] >= 5 else 0.0
        goal_block_pose = self._resolve_goal_block_pose(info, state)
        goal_block_pos = goal_block_pose[:2] if goal_block_pose is not None else self.target_pos
        goal_block_angle = float(goal_block_pose[2]) if goal_block_pose is not None else block_angle

        dist_reach = float(np.linalg.norm(eef_pos - block_pos))
        dist_push = float(np.linalg.norm(block_pos - goal_block_pos))
        angle_error = self._angle_distance(block_angle, goal_block_angle)

        r_push = math.exp(-dist_push / 100.0)
        dist_reach_eff = max(0.0, dist_reach - 60.0)
        r_reach = math.exp(-dist_reach_eff / 100.0)
        r_angle = math.exp(-angle_error / (math.pi / 6.0))

        info = dict(info)
        info["original_reward"] = float(original_reward)
        info["reach_reward"] = float(r_reach)
        info["push_reward"] = float(r_push)
        info["angle_reward"] = float(r_angle)
        info["distance_to_block"] = dist_reach
        info["distance_to_target"] = dist_push
        info["angle_error"] = angle_error

        dense_reward = (1.0 * r_reach) + (3.0 * r_push) + (1.0 * r_angle)
        visual_coverage = self._compute_visual_coverage(info)
        if visual_coverage is not None:
            info["coverage"] = float(visual_coverage)
        info["coverage_proxy"] = float(r_push)

        info["dense_reward"] = float(dense_reward)

        return observation, float(dense_reward), terminated, truncated, info


class PushTObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        observation = extract_state_array(observation)
        eef_x = np.clip(observation[0] / 512.0, 0.0, 1.0)
        eef_y = np.clip(observation[1] / 512.0, 0.0, 1.0)
        block_x = np.clip(observation[2] / 512.0, 0.0, 1.0)
        block_y = np.clip(observation[3] / 512.0, 0.0, 1.0)
        theta = float(observation[4])

        return np.array(
            [eef_x, eef_y, block_x, block_y, math.sin(theta), math.cos(theta)],
            dtype=np.float32,
        )


class TokenizerLatentObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, tokenizer_ckpt: str, tokenizer_device: str = "cpu"):
        super().__init__(env)
        self.latent_encoder = TokenizerZEncoder(tokenizer_ckpt=tokenizer_ckpt, device=tokenizer_device)
        latent_dim = self.latent_encoder.latent_dim
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(latent_dim,),
            dtype=np.float32,
        )

    def _render_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("Expected renderable RGB frame for tokenizer observation.")
        return np.asarray(frame, dtype=np.uint8)

    def observation(self, observation):
        del observation
        return self.latent_encoder.encode_frame(self._render_frame())


class RenderedImageObsWrapper(gym.ObservationWrapper):
    """
    Replace the state observation with rendered RGB frames resized to the
    tokenizer/BC training resolution so PPO can reuse the full BC image prior.
    """

    def __init__(
        self,
        env: gym.Env,
        tokenizer_ckpt: str | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
    ):
        super().__init__(env)
        if tokenizer_ckpt is not None:
            _, info = load_tokenizer_from_ckpt(tokenizer_ckpt, torch.device("cpu"))
            self.target_height = int(info["H"])
            self.target_width = int(info["W"])
        else:
            if target_height is None or target_width is None:
                raise ValueError("RenderedImageObsWrapper needs either tokenizer_ckpt or explicit target_height/target_width.")
            self.target_height = int(target_height)
            self.target_width = int(target_width)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.target_height, self.target_width, 3),
            dtype=np.uint8,
        )

    def _render_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("Expected renderable RGB frame for image observation.")
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape[:2] != (self.target_height, self.target_width):
            frame_tensor = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            frame_tensor = F.interpolate(
                frame_tensor,
                size=(self.target_height, self.target_width),
                mode="bilinear",
                align_corners=False,
            )
            frame = frame_tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
        return frame

    def observation(self, observation):
        del observation
        return self._render_frame()


def extract_state_array(observation: Any) -> np.ndarray:
    if isinstance(observation, dict):
        if "state" in observation:
            return np.asarray(observation["state"], dtype=np.float32)
        raise KeyError("Expected observation dict to contain a 'state' entry.")
    return np.asarray(observation, dtype=np.float32)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(value).squeeze())
    except Exception:
        return default


def extract_final_observation(infos: dict, env_index: int) -> np.ndarray | None:
    if "final_observation" in infos and "_final_observation" in infos:
        if infos["_final_observation"][env_index]:
            obs = infos["final_observation"][env_index]
            if obs is not None:
                return np.asarray(obs, dtype=np.float32)
    elif "final_info" in infos:
        try:
            final_info = infos["final_info"][env_index]
            if final_info is not None and "final_observation" in final_info:
                return np.asarray(final_info["final_observation"], dtype=np.float32)
        except Exception:
            pass
    return None


def extract_executed_primitives(infos: dict, env_index: int, default: int = 1) -> int:
    if "num_executed_primitives" in infos:
        try:
            value = int(np.asarray(infos["num_executed_primitives"][env_index]).squeeze())
            return max(1, value)
        except Exception:
            pass
    if "final_info" in infos:
        try:
            final_info = infos["final_info"][env_index]
            if final_info is not None and "num_executed_primitives" in final_info:
                return max(1, int(final_info["num_executed_primitives"]))
        except Exception:
            pass
    return max(1, int(default))


def add_timeout_bootstrap_rewards(rewards, terminations, truncations, infos, agent, device, gamma):
    rewards = np.asarray(rewards, dtype=np.float32).copy()
    timeout_mask = np.logical_and(truncations, np.logical_not(terminations))
    timeout_indices = np.where(timeout_mask)[0]

    if len(timeout_indices) == 0:
        return rewards, 0

    final_obs_list, valid_indices = [], []
    executed_steps = []
    for idx in timeout_indices:
        final_obs = extract_final_observation(infos, int(idx))
        if final_obs is not None:
            final_obs_list.append(final_obs)
            valid_indices.append(int(idx))
            executed_steps.append(extract_executed_primitives(infos, int(idx), default=1))

    missing_count = len(timeout_indices) - len(valid_indices)
    if not final_obs_list:
        return rewards, missing_count

    final_obs_tensor = torch.as_tensor(np.stack(final_obs_list), dtype=torch.float32, device=device)
    with torch.no_grad():
        final_values = agent.network.get_value(final_obs_tensor).squeeze(-1)

    final_values_np = final_values.detach().cpu().numpy().astype(np.float32)
    for idx, value, steps in zip(valid_indices, final_values_np, executed_steps):
        rewards[idx] += (gamma ** int(steps)) * float(value)

    return rewards, missing_count


def add_timeout_bootstrap_rewards_manual(
    rewards,
    terminations,
    truncations,
    final_observations,
    executed_steps,
    agent,
    device,
    gamma,
):
    rewards = np.asarray(rewards, dtype=np.float32).copy()
    terminations = np.asarray(terminations, dtype=bool)
    truncations = np.asarray(truncations, dtype=bool)
    timeout_indices = np.where(np.logical_and(truncations, np.logical_not(terminations)))[0]
    if len(timeout_indices) == 0:
        return rewards, 0

    final_obs_list = []
    valid_indices = []
    valid_steps = []
    for idx in timeout_indices:
        final_obs = final_observations[int(idx)]
        if final_obs is None:
            continue
        final_obs_list.append(np.asarray(final_obs, dtype=np.float32))
        valid_indices.append(int(idx))
        valid_steps.append(max(1, int(executed_steps[int(idx)])))

    missing_count = len(timeout_indices) - len(valid_indices)
    if not final_obs_list:
        return rewards, missing_count

    final_obs_tensor = torch.as_tensor(np.stack(final_obs_list), dtype=torch.float32, device=device)
    with torch.no_grad():
        final_values = agent.network.get_value(final_obs_tensor).squeeze(-1)

    final_values_np = final_values.detach().cpu().numpy().astype(np.float32)
    for idx, value, steps in zip(valid_indices, final_values_np, valid_steps):
        rewards[idx] += (gamma ** int(steps)) * float(value)

    return rewards, missing_count


@dataclass
class TrainConfig:
    env_id: str = DEFAULT_PUSHT_ENV_ID
    max_episode_steps: int = 300
    fixed_target: bool = True
    fixed_target_block_success: bool = True
    agent_block_coef: float = 0.0
    block_start_near_goal: bool = True
    block_start_radius: float = 200.0
    reward_mode: str = "sparse"
    action_mode: str = "absolute"
    num_envs: int = 8
    rollout_steps: int = 64
    total_timesteps: int = 1_000_000
    batch_size: int = 512
    ppo_epochs: int = 10
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    target_kl: float | None = 0.03
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    norm_reward: bool = True
    reward_clip: float = 10.0
    clip_vloss: bool = True

    chunk_size: int = 5
    max_step_pixels: float = 15.0
    ensemble_decay: float = 0.35
    actor_output_tanh: bool = True
    init_log_std: float = -2.0
    anneal_log_std: bool = False
    final_log_std: float = -3.5
    obs_stack_size: int = 3
    frame_stride: int = 5
    network_type: str = "bc_latent"
    image_height: int = 224
    image_width: int = 224
    actor_hidden_dim: int = 256
    actor_dropout: float = 0.05
    bc_pixel_residual: bool = False
    bc_pixel_residual_scale: float = 0.05

    bc_prior_path: str = "local_models/behavior_cloning/latest.pt"
    bc_stats_path: str | None = None
    prior_loss_coef: float = 0.0
    prior_loss_decay: float = 0.997
    bc_kl_penalty: bool = False
    bc_kl_penalty_coef: float = 0.0
    tokenizer_path: str = "logs/tokenizer_ckpts/latest.pt"
    tokenizer_device: str = "auto"

    seed: int = 42
    device: str = "auto"
    vector_env: str = "sync"
    save_path: str = "local_models/ppo_online/latest.pth"
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 10
    eval_episodes: int = 10
    eval_seed: int = 0
    norm_reward: bool = True
    reward_clip: float = 10.0
    clip_vloss: bool = True
    critic_warmup_ratio: float = 0.10  # NEU: 10% der Updates als Warmup


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.xpu.is_available():
            return torch.device("xpu")
        if not torch.cuda.is_available():
            return torch.device("cpu")

        try:
            major, minor = torch.cuda.get_device_capability(0)
            supported_arches = {
                arch.replace("sm_", "")
                for arch in torch.cuda.get_arch_list()
                if arch.startswith("sm_")
            }
            requested_arch = f"{major}{minor}"

            if supported_arches and requested_arch not in supported_arches:
                print(
                    "CUDA device detected but unsupported by this PyTorch build: "
                    f"sm_{requested_arch} not in {sorted(supported_arches)}. Falling back to CPU."
                )
                return torch.device("cpu")

            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as error:
            print(f"CUDA auto-detection failed ({error}). Falling back to CPU.")
            return torch.device("cpu")

    return torch.device(device_name)


def make_env(rank: int, seed: int, config: TrainConfig, render_mode: str | None = None):
    def _thunk():
        tokenizer_info = None
        if config.network_type == "bc_latent":
            _, tokenizer_info = load_tokenizer_from_ckpt(config.tokenizer_path, torch.device("cpu"))

        resolved_env_id = resolve_pusht_env_id(config.env_id)
        image_height = (
            int(tokenizer_info["H"])
            if tokenizer_info is not None
            else int(getattr(config, "image_height", 224))
        )
        image_width = (
            int(tokenizer_info["W"])
            if tokenizer_info is not None
            else int(getattr(config, "image_width", 224))
        )
        env = make_pusht_env(
            env_id=resolved_env_id,
            render_mode=render_mode or "rgb_array",
            image_height=image_height,
            image_width=image_width,
            max_episode_steps=int(config.max_episode_steps),
            relative=bool(config.action_mode in {"relative", "swm_relative"}),
            sync_goal_pose=True,
            align_sampled_goal_to_fixed_target=bool(config.fixed_target),
            fixed_target_block_success=bool(config.fixed_target_block_success),
            fixed_target_agent_block_coef=float(config.agent_block_coef),
            block_start_near_goal=bool(config.block_start_near_goal),
            block_start_radius=float(config.block_start_radius),
            reward_mode=str(config.reward_mode),
            render_obs=False,
        )
        if str(config.reward_mode) == "dense_shaped":
            env = PushTDenseRewardWrapper(env, env_id=resolved_env_id)
        env = OpenLoopChunkExecutionWrapper(
            env,
            chunk_size=config.chunk_size,
            gamma=config.gamma,
            action_mode=config.action_mode,
            action_output_tanh=bool(getattr(config, "action_output_tanh", False)),
            return_rendered_history=bool(config.network_type == "bc_pixels"),
            target_height=image_height if config.network_type == "bc_pixels" else None,
            target_width=image_width if config.network_type == "bc_pixels" else None,
            stack_size=config.obs_stack_size if config.network_type == "bc_pixels" else 1,
            frame_stride=config.frame_stride if config.network_type == "bc_pixels" else 1,
        )
        if config.network_type == "bc_pixels":
            pass
        elif config.network_type == "bc_latent":
            env = TokenizerLatentObsWrapper(
                env,
                tokenizer_ckpt=config.tokenizer_path,
                tokenizer_device=config.tokenizer_device,
            )
            env = StridedObservationStackWrapper(
                env,
                stack_size=config.obs_stack_size,
                frame_stride=config.frame_stride,
            )
        else:
            env = PushTObsWrapper(env)
        if str(config.vector_env) == "manual":
            env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return env

    return _thunk


def _extract_checkpoint_args(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    args = payload.get("args", {}) or {}
    if isinstance(args, dict):
        return args
    try:
        return vars(args)
    except TypeError:
        return {}


def extract_checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(payload, dict):
        if payload and all(isinstance(key, str) for key in payload.keys()):
            first_value = next(iter(payload.values()))
            if torch.is_tensor(first_value):
                return payload
        for key in ("state_dict", "model_state_dict", "network", "model", "actor", "agent"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                state_dict = extract_checkpoint_state_dict(nested)
                if state_dict is not None:
                    return state_dict
    return None


def _load_bc_contract_overrides(bc_prior_path: str) -> dict[str, Any]:
    try:
        payload = torch.load(bc_prior_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(bc_prior_path, map_location="cpu")
    except Exception as error:
        print(f"Warning: could not read BC checkpoint args from {bc_prior_path}: {error}")
        return {}

    ckpt_args = _extract_checkpoint_args(payload)
    state_dict = extract_checkpoint_state_dict(payload)
    overrides: dict[str, Any] = {}
    if ckpt_args.get("seq_len") is not None:
        overrides["obs_stack_size"] = int(ckpt_args["seq_len"])
    if ckpt_args.get("frame_stride") is not None:
        overrides["frame_stride"] = int(ckpt_args["frame_stride"])
    if ckpt_args.get("action_chunk_size") is not None:
        overrides["chunk_size"] = int(ckpt_args["action_chunk_size"])
    if ckpt_args.get("hidden_dim") is not None:
        overrides["actor_hidden_dim"] = int(ckpt_args["hidden_dim"])
    if ckpt_args.get("tokenizer_ckpt_name"):
        overrides["tokenizer_path"] = str(ckpt_args["tokenizer_ckpt_name"])
    if ckpt_args.get("action_mode") is not None:
        overrides["action_mode"] = str(ckpt_args["action_mode"])
    image_hw = ckpt_args.get("image_hw")
    if isinstance(image_hw, (list, tuple)) and len(image_hw) == 2:
        if image_hw[0] is not None and image_hw[1] is not None:
            overrides["image_height"] = int(image_hw[0])
            overrides["image_width"] = int(image_hw[1])
    if state_dict is not None:
        state_keys = list(state_dict.keys())
        if any(key.startswith("backbone.") or key.startswith("classifier.") for key in state_keys):
            overrides["network_type"] = "bc_pixels"
    return overrides


def _load_bc_stats_overrides(stats_path: str | None) -> dict[str, Any]:
    if not stats_path:
        return {}
    try:
        payload = torch.load(stats_path, map_location="cpu")
    except Exception as error:
        print(f"Warning: could not read BC stats from {stats_path}: {error}")
        return {}

    overrides: dict[str, Any] = {}
    if isinstance(payload, dict):
        if payload.get("frame_stack") is not None:
            overrides["obs_stack_size"] = int(payload["frame_stack"])
        if payload.get("frame_stride") is not None:
            overrides["frame_stride"] = int(payload["frame_stride"])
        if payload.get("action_chunk_size") is not None:
            overrides["chunk_size"] = int(payload["action_chunk_size"])
        if payload.get("hidden_dim") is not None:
            overrides["actor_hidden_dim"] = int(payload["hidden_dim"])
        if payload.get("latent_dim") is not None:
            overrides["latent_dim"] = int(payload["latent_dim"])
        if payload.get("image_normalization") is not None:
            overrides["bc_stats_image_normalization"] = payload["image_normalization"]
        if payload.get("action_mode") is not None:
            overrides["action_mode"] = str(payload["action_mode"])
    return overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on PushT with optional BC/tokenizer overrides.")
    parser.add_argument("--bc-prior-path", "--bc_checkpoint", dest="bc_prior_path", type=str, default=None)
    parser.add_argument("--bc-stats", type=str, default=None)
    parser.add_argument("--network-type", choices=("bc_latent", "bc_pixels", "mlp"), default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--vector-env", choices=("sync", "async", "manual"), default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--bc-pixel-residual", action="store_true")
    parser.add_argument("--no-bc-pixel-residual", action="store_true")
    parser.add_argument("--bc-pixel-residual-scale", type=float, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--action-chunk-size", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=None)
    parser.add_argument("--norm-reward", action="store_true")
    parser.add_argument("--no-norm-reward", action="store_true")
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--no-clip-vloss", action="store_true")
    parser.add_argument("--init-log-std", type=float, default=None)
    parser.add_argument("--final-log-std", type=float, default=None)
    parser.add_argument("--anneal-log-std", action="store_true")
    parser.add_argument("--no-anneal-log-std", action="store_true")
    parser.add_argument("--bc-kl-penalty", action="store_true")
    parser.add_argument("--no-bc-kl-penalty", action="store_true")
    parser.add_argument("--bc-kl-penalty-coef", type=float, default=None)
    parser.add_argument("--prior-loss-coef", type=float, default=None)
    parser.add_argument("--fixed-target", action="store_true")
    parser.add_argument("--no-fixed-target", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=("sparse", "dense", "dense_shaped"),
        default=None,
    )
    parser.add_argument(
        "--action-mode",
        choices=("absolute", "relative", "swm_relative"),
        default=None,
    )
    parser.add_argument("--block-start-radius", type=float, default=None)
    parser.add_argument("--agent-block-coef", type=float, default=None)
    parser.add_argument("--fixed-target-block-success", action="store_true")
    parser.add_argument("--no-fixed-target-block-success", action="store_true")
    parser.add_argument("--block-start-near-goal", action="store_true")
    parser.add_argument("--no-block-start-near-goal", action="store_true")
    parser.add_argument("--critic-warmup-ratio", type=float, default=None)  # NEU
    return parser.parse_args()


def evaluate_current_policy(
    agent: PPOAgent,
    config: TrainConfig,
    *,
    episodes: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    env = make_env(0, seed, config, render_mode="rgb_array")()
    returns = []
    coverages = []
    successes = []
    lengths = []
    try:
        with torch.no_grad():
            for episode_idx in range(int(episodes)):
                state, _ = env.reset(seed=int(seed) + episode_idx)
                done = False
                total_reward = 0.0
                step_count = 0
                final_info = {}
                terminated = False
                while not done:
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_mean = agent.network.actor_mean(state_tensor)
                    env_action = action_mean.squeeze(0).cpu().numpy()
                    state, reward, terminated, truncated, info = env.step(env_action)
                    total_reward += float(reward)
                    step_count += 1
                    final_info = dict(info)
                    done = bool(terminated or truncated)
                returns.append(float(total_reward))
                lengths.append(int(step_count))
                coverages.append(float(final_info.get("coverage", 0.0)))
                success = float(final_info.get("block_success", final_info.get("success", terminated)))
                successes.append(float(success))
    finally:
        env.close()

    return {
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_coverage": float(np.mean(coverages)) if coverages else float("nan"),
        "success_rate": float(np.mean(successes)) if successes else float("nan"),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
    }


def _manual_reset_envs(envs: list[gym.Env], seed: int) -> np.ndarray:
    states = []
    for rank, env in enumerate(envs):
        state, _ = env.reset(seed=int(seed) + rank)
        states.append(np.asarray(state))
    return np.stack(states)


def _manual_step_envs(
    envs: list[gym.Env],
    actions_np: np.ndarray,
    *,
    base_seed: int,
    global_env_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[np.ndarray | None], np.ndarray]:
    next_states = []
    rewards = []
    terminations = []
    truncations = []
    infos: list[dict[str, Any]] = []
    final_observations: list[np.ndarray | None] = []
    executed_steps: list[int] = []

    for env_index, (env, action) in enumerate(zip(envs, actions_np)):
        next_state, reward, terminated, truncated, info = env.step(action)
        info = dict(info)
        executed = max(1, int(info.get("num_executed_primitives", 1)))
        episode_done = bool(terminated or truncated)
        final_observation = np.asarray(next_state, dtype=np.float32) if episode_done else None
        if episode_done:
            reset_seed = int(base_seed + env_index + global_env_steps + 1)
            reset_state, reset_info = env.reset(seed=reset_seed)
            info["reset_info"] = reset_info
            next_state = reset_state

        next_states.append(np.asarray(next_state))
        rewards.append(float(reward))
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        infos.append(info)
        final_observations.append(final_observation)
        executed_steps.append(executed)

    return (
        np.stack(next_states),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(terminations, dtype=bool),
        np.asarray(truncations, dtype=bool),
        infos,
        final_observations,
        np.asarray(executed_steps, dtype=np.int32),
    )


def save_ppo_checkpoint(
    path: Path,
    *,
    agent: PPOAgent,
    config: TrainConfig,
    global_step: int,
    success_rate: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent": agent.network.state_dict(),
        "config": dict(vars(config)),
        "global_step": int(global_step),
        "success_rate": success_rate,
        "contract": {
            "frame_stack": int(config.obs_stack_size),
            "frame_stride": int(config.frame_stride),
            "action_chunk_size": int(config.chunk_size),
            "latent_dim": int(getattr(agent.network, "feature_dim", 512)),
            "hidden_dim": int(config.actor_hidden_dim),
            "action_dim": 2,
        },
    }
    reward_norm = getattr(config, "_reward_normalizer_state", None)
    if reward_norm is not None:
        payload["reward_norm"] = reward_norm
    torch.save(payload, path)


def train_pusht():
    args = parse_args()
    config = TrainConfig()
    if args.bc_prior_path is not None:
        config.bc_prior_path = args.bc_prior_path
    config.bc_prior_path = resolve_bc_prior_path(config.bc_prior_path)
    contract_overrides = _load_bc_contract_overrides(config.bc_prior_path)
    for key, value in contract_overrides.items():
        setattr(config, key, value)
    if args.bc_stats is not None:
        config.bc_stats_path = str(args.bc_stats)
    stats_overrides = _load_bc_stats_overrides(config.bc_stats_path)
    for key, value in stats_overrides.items():
        setattr(config, key, value)
    if args.network_type is not None:
        config.network_type = str(args.network_type)
    if args.tokenizer_path is not None:
        config.tokenizer_path = args.tokenizer_path
    if args.save_path is not None:
        config.save_path = args.save_path
    if args.vector_env is not None:
        config.vector_env = str(args.vector_env)
    if args.num_envs is not None:
        config.num_envs = int(args.num_envs)
    if args.num_chunks is not None:
        config.rollout_steps = int(args.num_chunks)
    if args.total_timesteps is not None:
        config.total_timesteps = int(args.total_timesteps)
    if args.hidden_dim is not None:
        config.actor_hidden_dim = int(args.hidden_dim)
    if args.bc_pixel_residual:
        config.bc_pixel_residual = True
    if args.no_bc_pixel_residual:
        config.bc_pixel_residual = False
    if args.bc_pixel_residual_scale is not None:
        config.bc_pixel_residual_scale = float(args.bc_pixel_residual_scale)
    if args.frame_stack is not None:
        config.obs_stack_size = int(args.frame_stack)
    if args.frame_stride is not None:
        config.frame_stride = int(args.frame_stride)
    if args.action_chunk_size is not None:
        config.chunk_size = int(args.action_chunk_size)
    if args.image_height is not None:
        config.image_height = int(args.image_height)
    if args.image_width is not None:
        config.image_width = int(args.image_width)
    if args.log_interval is not None:
        config.log_interval = int(args.log_interval)
    if args.save_interval is not None:
        config.save_interval = int(args.save_interval)
    if args.eval_interval is not None:
        config.eval_interval = int(args.eval_interval)
    if args.eval_episodes is not None:
        config.eval_episodes = int(args.eval_episodes)
    if args.learning_rate is not None:
        config.learning_rate = float(args.learning_rate)
    if args.max_episode_steps is not None:
        config.max_episode_steps = int(args.max_episode_steps)
    if args.eval_seed is not None:
        config.eval_seed = int(args.eval_seed)
    if args.reward_clip is not None:
        config.reward_clip = float(args.reward_clip)
    if args.norm_reward:
        config.norm_reward = True
    if args.no_norm_reward:
        config.norm_reward = False
    if args.clip_vloss:
        config.clip_vloss = True
    if args.no_clip_vloss:
        config.clip_vloss = False
    if args.init_log_std is not None:
        config.init_log_std = float(args.init_log_std)
    if args.final_log_std is not None:
        config.final_log_std = float(args.final_log_std)
    if args.anneal_log_std:
        config.anneal_log_std = True
    if args.no_anneal_log_std:
        config.anneal_log_std = False
    if args.bc_kl_penalty:
        config.bc_kl_penalty = True
    if args.no_bc_kl_penalty:
        config.bc_kl_penalty = False
    if args.bc_kl_penalty_coef is not None:
        config.bc_kl_penalty_coef = float(args.bc_kl_penalty_coef)
    if args.prior_loss_coef is not None:
        config.prior_loss_coef = float(args.prior_loss_coef)
    if args.fixed_target:
        config.fixed_target = True
    if args.no_fixed_target:
        config.fixed_target = False
    if args.reward_mode is not None:
        config.reward_mode = str(args.reward_mode)
    if args.action_mode is not None:
        config.action_mode = str(args.action_mode)
    if args.block_start_radius is not None:
        config.block_start_radius = float(args.block_start_radius)
    if args.agent_block_coef is not None:
        config.agent_block_coef = float(args.agent_block_coef)
    if args.fixed_target_block_success:
        config.fixed_target_block_success = True
    if args.no_fixed_target_block_success:
        config.fixed_target_block_success = False
    if args.block_start_near_goal:
        config.block_start_near_goal = True
    if args.no_block_start_near_goal:
        config.block_start_near_goal = False
    if config.network_type == "bc_latent" or args.tokenizer_path is not None:
        config.tokenizer_path = resolve_tokenizer_path(config.tokenizer_path)
    config.save_path = resolve_ppo_checkpoint_path(config.save_path)
    policy_device = resolve_device(config.device)
    tokenizer_device = resolve_device(config.tokenizer_device)
    config.device = str(policy_device)
    config.tokenizer_device = str(tokenizer_device)
    wandb.init(
        project="pusht-ppo",
        name=f"ppo_bc_chunk{config.chunk_size}_seed{config.seed}",
        config=config.__dict__,
        mode=args.wandb_mode,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = policy_device
    print(f"Training starts on policy_device={device}, tokenizer_device={tokenizer_device}")
    print(
        f"Model paths: bc_prior={config.bc_prior_path} "
        f"tokenizer={config.tokenizer_path} save={config.save_path}"
    )
    tokenizer_info = None
    if config.network_type == "bc_latent":
        _, tokenizer_info = load_tokenizer_from_ckpt(config.tokenizer_path, torch.device("cpu"))
    if tokenizer_info is not None:
        print(
            "Tokenizer/image setup: "
            f"H={tokenizer_info['H']} W={tokenizer_info['W']} "
            f"latent_dim={tokenizer_info['latent_dim']}"
        )
    else:
        print(
            "Image setup: "
            f"H={int(config.image_height)} W={int(config.image_width)} "
            f"network_type={config.network_type}"
        )
    print(
        "Resolved PPO contract: "
        f"obs_stack={config.obs_stack_size} frame_stride={config.frame_stride} "
        f"chunk={config.chunk_size} hidden_dim={config.actor_hidden_dim} "
        f"rollout_steps={config.rollout_steps} num_envs={config.num_envs} "
        f"max_episode_steps={config.max_episode_steps} fixed_target={config.fixed_target} "
        f"reward_mode={config.reward_mode} "
        f"block_start_radius={config.block_start_radius} action_mode={config.action_mode}"
    )

    env_fns = [make_env(rank, config.seed, config) for rank in range(config.num_envs)]
    manual_envs: list[gym.Env] | None = None
    if config.vector_env == "manual":
        manual_envs = [fn() for fn in env_fns]
        if device.type == "cpu" and tokenizer_device.type == "cpu":
            print("Warning: policy and tokenizer are both on CPU, so latent observation training will be slow.")
        states = _manual_reset_envs(manual_envs, config.seed)
        obs_shape = tuple(manual_envs[0].observation_space.shape)
        action_dim = int(manual_envs[0].action_space.shape[0])
        observation_dtype = manual_envs[0].observation_space.dtype
        envs = None
    else:
        if config.vector_env == "async":
            envs = gym.vector.AsyncVectorEnv(env_fns)
        else:
            envs = gym.vector.SyncVectorEnv(env_fns)
        envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

        if config.vector_env == "async":
            print("Warning: AsyncVectorEnv creates separate worker processes, so tokenizer models are duplicated per worker.")
        elif device.type == "cpu" and tokenizer_device.type == "cpu":
            print("Warning: policy and tokenizer are both on CPU, so latent observation training will be slow.")

        states, _ = envs.reset(seed=config.seed)
        obs_shape = tuple(envs.single_observation_space.shape)
        action_dim = int(envs.single_action_space.shape[0])
        observation_dtype = envs.single_observation_space.dtype

    state_dim = int(obs_shape[-1]) if config.network_type == "bc_latent" else int(np.prod(obs_shape))
    print(
        f"Observation setup: network_type={config.network_type} "
        f"obs_shape={obs_shape} state_dim={state_dim} action_dim={action_dim}"
    )

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config.learning_rate,
        clip_coef=config.clip_coef,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        actor_output_tanh=config.actor_output_tanh,
        bc_prior_path=config.bc_prior_path,
        prior_loss_coef=config.prior_loss_coef,
        prior_loss_decay=config.prior_loss_decay,
        bc_kl_penalty=config.bc_kl_penalty,
        bc_kl_penalty_coef=config.bc_kl_penalty_coef,
        prior_log_std_init=config.init_log_std,
        device=device,
        network_type=config.network_type,
        actor_hidden_dim=config.actor_hidden_dim,
        actor_dropout=config.actor_dropout,
        bc_pixel_residual=config.bc_pixel_residual,
        bc_pixel_residual_scale=config.bc_pixel_residual_scale,
        obs_shape=obs_shape,
        tokenizer_path=config.tokenizer_path,
        backbone_device=tokenizer_device,
    )
    if config.anneal_log_std:
        agent.network.log_std.requires_grad_(False)
        agent.set_log_std(config.init_log_std)
    if config.network_type == "bc_pixels" and agent.bc_pixel_architecture is not None:
        config.backbone_style = str(agent.bc_pixel_architecture["backbone_style"])
        config.feature_dim = int(agent.bc_pixel_architecture["feature_dim"])
        config.action_output_tanh = bool(agent.bc_pixel_architecture["action_output_tanh"])
    print(agent.prior_load_info.message)

    buffer = PPOVectorBuffer(
        buffer_size=config.rollout_steps,
        num_envs=config.num_envs,
        state_shape=obs_shape,
        action_dim=action_dim,
        device=device,
        state_dtype=observation_dtype,
    )
    reward_normalizer = (
        RewardNormalizer(
            num_envs=config.num_envs,
            gamma=config.gamma ** config.chunk_size,
            clip=config.reward_clip,
        )
        if config.norm_reward
        else None
    )

    steps_per_update = config.num_envs * config.rollout_steps * config.chunk_size
    num_updates = max(1, config.total_timesteps // steps_per_update)
    
    warmup_updates = int(num_updates * config.critic_warmup_ratio)
    
    global_env_steps = 0
    last_dones_for_gae = np.zeros(config.num_envs, dtype=np.float32)
    warned_missing_timeout_bootstrap = False
    best_eval_success = float("-inf")

    for update in range(num_updates):
        freeze_actor = update < warmup_updates
        frac = 1.0 - (update / max(1, num_updates))
        lr_now = frac * config.learning_rate if config.anneal_lr else config.learning_rate
        agent.optimizer.param_groups[0]["lr"] = lr_now
        agent.ent_coef = frac * config.ent_coef
        if config.anneal_log_std:
            scheduled_log_std = config.final_log_std + frac * (config.init_log_std - config.final_log_std)
            agent.set_log_std(scheduled_log_std)
        else:
            scheduled_log_std = float(agent.network.log_std.detach().mean().cpu())

        buffer.clear()
        reward_sum = 0.0
        reward_count = 0
        action_abs_max = 0.0
        rollout_returns = []
        rollout_coverages = []
        rollout_coverage_proxies = []
        rollout_chunks = []

        for _ in range(config.rollout_steps):
            state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions, logprobs, _, values = agent.network.get_action_and_value(state_tensor)

            actions_np = actions.detach().cpu().numpy().astype(np.float32)
            env_actions = actions_np
            action_abs_max = max(action_abs_max, float(np.max(np.abs(actions_np))))

            if manual_envs is not None:
                (
                    next_states,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                    final_observations,
                    executed_primitives,
                ) = _manual_step_envs(
                    manual_envs,
                    env_actions,
                    base_seed=config.seed,
                    global_env_steps=global_env_steps,
                )
                global_env_steps += int(np.sum(executed_primitives))
                rewards, missing_count = add_timeout_bootstrap_rewards_manual(
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    final_observations=final_observations,
                    executed_steps=executed_primitives,
                    agent=agent,
                    device=device,
                    gamma=config.gamma,
                )
            else:
                next_states, rewards, terminations, truncations, infos = envs.step(env_actions)
                executed_primitives = None
                if "num_executed_primitives" in infos:
                    try:
                        executed_primitives = np.asarray(infos["num_executed_primitives"], dtype=np.int32)
                    except Exception:
                        executed_primitives = None
                if executed_primitives is None:
                    global_env_steps += int(config.num_envs * config.chunk_size)
                else:
                    global_env_steps += int(np.sum(executed_primitives))

                rewards, missing_count = add_timeout_bootstrap_rewards(
                    rewards=np.asarray(rewards, dtype=np.float32),
                    terminations=np.asarray(terminations, dtype=bool),
                    truncations=np.asarray(truncations, dtype=bool),
                    infos=infos,
                    agent=agent,
                    device=device,
                    gamma=config.gamma,
                )

            if missing_count > 0 and not warned_missing_timeout_bootstrap:
                warned_missing_timeout_bootstrap = True
                print("Warning: TimeLimit bootstrapping is missing final_observation.")

            dones_for_gae = np.logical_or(terminations, truncations).astype(np.float32)
            if reward_normalizer is not None:
                rewards = reward_normalizer.normalize(rewards, dones_for_gae)

            if manual_envs is not None:
                for info in infos:
                    rollout_chunks.append(float(info.get("num_executed_primitives", config.chunk_size)))
                    episode = info.get("episode")
                    if episode is None:
                        continue
                    ep_return = as_float(episode.get("r", 0.0))
                    rollout_returns.append(ep_return)
                    if "coverage" in info:
                        rollout_coverages.append(float(info["coverage"]))
                    if "coverage_proxy" in info:
                        rollout_coverage_proxies.append(float(info["coverage_proxy"]))
            else:
                if "num_active_chunks" in infos:
                    try:
                        rollout_chunks.append(float(np.mean(np.asarray(infos["num_active_chunks"], dtype=np.float32))))
                    except Exception:
                        pass

                if "episode" in infos and "_episode" in infos:
                    for i, is_done in enumerate(infos["_episode"]):
                        if is_done:
                            ep_return = float(infos["episode"]["r"][i])
                            coverage = None
                            if "final_info" in infos and infos["final_info"][i] is not None:
                                coverage = infos["final_info"][i].get("coverage")
                            elif "coverage" in infos:
                                try:
                                    coverage = infos["coverage"][i]
                                except Exception:
                                    coverage = None

                            rollout_returns.append(ep_return)
                            if coverage is not None:
                                rollout_coverages.append(float(coverage))
                            coverage_proxy = None
                            if "final_info" in infos and infos["final_info"][i] is not None:
                                coverage_proxy = infos["final_info"][i].get("coverage_proxy")
                            elif "coverage_proxy" in infos:
                                try:
                                    coverage_proxy = infos["coverage_proxy"][i]
                                except Exception:
                                    coverage_proxy = None
                            if coverage_proxy is not None:
                                rollout_coverage_proxies.append(float(coverage_proxy))
                elif "final_info" in infos:
                    for final_info in infos["final_info"]:
                        if final_info is not None and "episode" in final_info:
                            ep_return = as_float(final_info["episode"].get("r", 0.0))
                            rollout_returns.append(ep_return)
                            if "coverage" in final_info:
                                rollout_coverages.append(float(final_info["coverage"]))
                            if "coverage_proxy" in final_info:
                                rollout_coverage_proxies.append(float(final_info["coverage_proxy"]))

            buffer.store(
                states,
                actions.detach(),
                logprobs.detach(),
                rewards,
                dones_for_gae,
                values.detach(),
            )

            reward_sum += float(np.sum(rewards))
            reward_count += int(np.size(rewards))
            states = next_states
            last_dones_for_gae = dones_for_gae

        with torch.no_grad():
            next_state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
            next_values = agent.network.get_value(next_state_tensor).squeeze(-1)

        advantages, returns = buffer.compute_returns_and_advantages(
            next_value=next_values,
            next_done=torch.as_tensor(last_dones_for_gae, dtype=torch.float32, device=device),
            gamma=config.gamma ** config.chunk_size,
            gae_lambda=config.gae_lambda,
        )

        stats = agent.update(
            buffer,
            advantages.flatten(),
            returns.flatten(),
            batch_size=config.batch_size,
            ppo_epochs=config.ppo_epochs,
            update_idx=update,
            clip_vloss=bool(config.clip_vloss),
            freeze_actor=freeze_actor,  # NEU: Flag übergeben
        )

        global_step = int(global_env_steps)
        mean_step_reward = reward_sum / max(1, reward_count)

        wandb_log_dict = {
            "global_step": global_step,
            "Environment/Mean_Step_Reward": mean_step_reward,
            "Environment/Max_Action": action_abs_max,
            "PPO/Policy_Loss": stats["policy_loss"],
            "PPO/Value_Loss": stats["value_loss"],
            "PPO/Entropy": stats["entropy"],
            "PPO/Approx_KL": stats["approx_kl"],
            "PPO/Clip_Fraction": stats["clipfrac"],
            "PPO/Prior_Loss": stats["prior_loss"],
            "PPO/BC_KL_Loss": stats["bc_kl_loss"],
            "PPO/Prior_Loss_Coef": stats["prior_loss_coef"],
            "PPO/BC_KL_Penalty_Coef": stats["bc_kl_penalty_coef"],
            "Hyperparameters/Learning_Rate": lr_now,
            "Hyperparameters/Entropy_Coef": agent.ent_coef,
            "Hyperparameters/Log_Std": scheduled_log_std,
        }

        if rollout_returns:
            wandb_log_dict["Environment/Mean_Episode_Return"] = sum(rollout_returns) / len(rollout_returns)
        if rollout_coverages:
            wandb_log_dict["Environment/Mean_Coverage"] = sum(rollout_coverages) / len(rollout_coverages)
            wandb_log_dict["Environment/Max_Coverage"] = max(rollout_coverages)
        if rollout_coverage_proxies:
            wandb_log_dict["Environment/Mean_Coverage_Proxy"] = (
                sum(rollout_coverage_proxies) / len(rollout_coverage_proxies)
            )
            wandb_log_dict["Environment/Max_Coverage_Proxy"] = max(rollout_coverage_proxies)
        if rollout_chunks:
            wandb_log_dict["Chunking/Mean_Active_Chunks"] = sum(rollout_chunks) / len(rollout_chunks)

        if config.eval_interval > 0 and (update + 1) % config.eval_interval == 0:
            eval_summary = evaluate_current_policy(
                agent,
                config,
                episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=device,
            )
            wandb_log_dict["Eval/Mean_Return"] = eval_summary["mean_return"]
            wandb_log_dict["Eval/Mean_Coverage"] = eval_summary["mean_coverage"]
            wandb_log_dict["Eval/Success_Rate"] = eval_summary["success_rate"]
            wandb_log_dict["Eval/Mean_Length"] = eval_summary["mean_length"]
            if eval_summary["success_rate"] > best_eval_success:
                best_eval_success = eval_summary["success_rate"]
                best_path = Path(config.save_path).with_name("best.pth")
                config._reward_normalizer_state = (
                    reward_normalizer.state_dict() if reward_normalizer is not None else None
                )
                save_ppo_checkpoint(
                    best_path,
                    agent=agent,
                    config=config,
                    global_step=global_step,
                    success_rate=float(eval_summary["success_rate"]),
                )
                wandb.save(str(best_path))
                print(
                    f"New best PPO checkpoint at update={update + 1}: "
                    f"success_rate={eval_summary['success_rate']:.3f} saved={best_path}"
                )

        wandb.log(wandb_log_dict, step=global_step)

        if (update + 1) % max(1, config.log_interval) == 0:
            mean_episode_return = wandb_log_dict.get("Environment/Mean_Episode_Return", float("nan"))
            mean_coverage = wandb_log_dict.get("Environment/Mean_Coverage", float("nan"))
            max_coverage = wandb_log_dict.get("Environment/Max_Coverage", float("nan"))
            mean_coverage_proxy = wandb_log_dict.get("Environment/Mean_Coverage_Proxy", float("nan"))
            mean_active_chunks = wandb_log_dict.get("Chunking/Mean_Active_Chunks", float("nan"))
            eval_success = wandb_log_dict.get("Eval/Success_Rate", float("nan"))
            print(
                f"update={update + 1}/{num_updates} "
                f"step={global_step} "
                f"episodes={len(rollout_returns)} "
                f"mean_return={mean_episode_return:.2f} "
                f"mean_cov={mean_coverage:.3f} "
                f"max_cov={max_coverage:.3f} "
                f"mean_cov_proxy={mean_coverage_proxy:.3f} "
                f"step_reward={mean_step_reward:.3f} "
                f"kl={stats['approx_kl']:.5f} "
                f"entropy={stats['entropy']:.3f} "
                f"log_std={scheduled_log_std:.3f} "
                f"chunks={mean_active_chunks:.2f} "
                f"eval_success={eval_success:.3f}"
            )

        if (update + 1) % max(1, config.save_interval) == 0:
            config._reward_normalizer_state = (
                reward_normalizer.state_dict() if reward_normalizer is not None else None
            )
            save_ppo_checkpoint(
                Path(config.save_path),
                agent=agent,
                config=config,
                global_step=global_step,
                success_rate=wandb_log_dict.get("Eval/Success_Rate"),
            )
            wandb.save(config.save_path)

    print("Saving final model weights...")
    config._reward_normalizer_state = reward_normalizer.state_dict() if reward_normalizer is not None else None
    save_ppo_checkpoint(
        Path(config.save_path),
        agent=agent,
        config=config,
        global_step=global_step,
        success_rate=best_eval_success if np.isfinite(best_eval_success) else None,
    )
    wandb.save(config.save_path)
    print("Training finished.")
    if envs is not None:
        envs.close()
    if manual_envs is not None:
        for env in manual_envs:
            env.close()
    wandb.finish()


if __name__ == "__main__":
    train_pusht()
