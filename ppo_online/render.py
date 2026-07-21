#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from collections import deque
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation

from ppo_online.env_config import make_pusht_env, resolve_pusht_env_id
from ppo_online.model_paths import resolve_bc_prior_path, resolve_ppo_checkpoint_path, resolve_tokenizer_path
from ppo_online.networks import BCPixelActorCritic, TokenizerLatentBCPPOActorCritic, VectorActorCritic
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt
from ppo_online.train import (
    ActionChunkingTemporalEnsembleWrapper,
    PushTDenseRewardWrapper,
    PushTObsWrapper,
    RenderedImageObsWrapper,
    TokenizerLatentObsWrapper,
    TrainConfig,
    extract_state_array,
    resolve_device,
)


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


def load_state_dict_safe(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_checkpoint_state_dict(payload):
    if isinstance(payload, dict):
        if payload and all(isinstance(k, str) for k in payload.keys()):
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


def extract_checkpoint_args(payload):
    if not isinstance(payload, dict):
        return {}
    args = payload.get("args", {}) or {}
    if isinstance(args, dict):
        return args
    try:
        return vars(args)
    except TypeError:
        return {}


def load_bc_prior_into_network(network: torch.nn.Module, path: str, device: torch.device, network_type: str):
    payload = load_state_dict_safe(path, device)
    state_dict = extract_checkpoint_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unsupported BC prior format in {path}")

    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "_orig_mod.", "network."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned[clean_key] = value

    if network_type == "bc_latent":
        policy_state = cleaned
        if any(key.startswith("classifier.") for key in cleaned):
            policy_state = {
                key[len("classifier.") :]: value
                for key, value in cleaned.items()
                if key.startswith("classifier.")
            }
        if not policy_state:
            raise ValueError(f"No latent BC policy weights found in BC prior {path}")
        incompatible = network.bc_policy.load_state_dict(policy_state, strict=False)
    elif network_type == "bc_pixels":
        actor_state = {
            key: value
            for key, value in cleaned.items()
            if key.startswith("backbone.") or key.startswith("classifier.")
        }
        if not actor_state:
            raise ValueError(f"No backbone/classifier weights found in BC prior {path}")
        incompatible = network.load_state_dict(actor_state, strict=False)
    else:
        raise ValueError(f"BC prior loading is only supported for bc_latent/bc_pixels, got {network_type}")

    print(
        f"Loaded BC prior from {path}. "
        f"missing={list(incompatible.missing_keys)[:6]} "
        f"unexpected={list(incompatible.unexpected_keys)[:6]}"
    )


class ChunkExecutionWrapper(gym.Wrapper):
    """
    Debug wrapper for BC checkpoints: interpret predicted action chunks without
    PPO's ACT-style temporal ensembling.
    """

    def __init__(self, env: gym.Env, chunk_size: int, mode: str, max_step_pixels: float, clip_actions: bool = True):
        super().__init__(env)
        self.chunk_size = chunk_size
        self.mode = mode
        self.max_step_pixels = max_step_pixels
        self.clip_actions = bool(clip_actions)
        low = np.full((chunk_size * 2,), -1.0, dtype=np.float32)
        high = np.full((chunk_size * 2,), 1.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.current_eef = np.array([256.0, 256.0], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = extract_state_array(obs)
        self.current_eef = state[0:2].copy()
        return obs, info

    def _map_primitive_to_env_action(self, primitive: np.ndarray) -> np.ndarray:
        if self.mode in {"first_absolute", "chunk_absolute"}:
            return np.clip((primitive + 1.0) * 256.0, 0.0, 512.0)
        if self.mode in {"first_delta", "chunk_delta"}:
            return np.clip(self.current_eef + primitive * self.max_step_pixels, 0.0, 512.0)
        if self.mode in {"first_relative", "chunk_relative"}:
            return np.asarray(primitive, dtype=np.float32)
        raise ValueError(f"Unsupported chunk execution mode: {self.mode}")

    def step(self, macro_action):
        macro_action = np.asarray(macro_action, dtype=np.float32)
        if self.clip_actions:
            macro_action = np.clip(macro_action, -1.0, 1.0)
        primitives = macro_action.reshape(self.chunk_size, 2)

        if self.mode.startswith("first_"):
            primitives = primitives[:1]

        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        executed_actions = []

        for primitive in primitives:
            env_action = self._map_primitive_to_env_action(primitive)
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            state = extract_state_array(obs)
            self.current_eef = state[0:2].copy()
            total_reward += float(reward)
            executed_actions.append(env_action.astype(np.float32))
            if terminated or truncated:
                break

        info = dict(info)
        info["executed_actions"] = np.asarray(executed_actions, dtype=np.float32)
        info["executed_primitives"] = primitives[: len(executed_actions)].astype(np.float32)
        return obs, float(total_reward), terminated, truncated, info


class TemporalEnsembleChunkExecutionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, chunk_size: int, mode: str, max_step_pixels: float, ensemble_decay: float, clip_actions: bool = True):
        super().__init__(env)
        self.chunk_size = chunk_size
        self.mode = mode
        self.max_step_pixels = max_step_pixels
        self.ensemble_decay = ensemble_decay
        self.clip_actions = bool(clip_actions)
        low = np.full((chunk_size * 2,), -1.0, dtype=np.float32)
        high = np.full((chunk_size * 2,), 1.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.current_eef = np.array([256.0, 256.0], dtype=np.float32)
        self.pending_chunks: deque[dict[str, object]] = deque()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = extract_state_array(obs)
        self.current_eef = state[0:2].copy()
        self.pending_chunks.clear()
        return obs, info

    def _ensemble_primitive(self) -> np.ndarray:
        candidates = []
        weights = []
        for age, entry in enumerate(reversed(self.pending_chunks)):
            offset = int(entry["offset"])
            if offset >= self.chunk_size:
                continue
            candidates.append(np.asarray(entry["chunk"], dtype=np.float32)[offset])
            weights.append(math.exp(-self.ensemble_decay * age))

        if not candidates:
            return np.zeros(2, dtype=np.float32)

        weights_np = np.asarray(weights, dtype=np.float32)
        weights_np /= max(weights_np.sum(), 1e-8)
        return np.sum(np.asarray(candidates, dtype=np.float32) * weights_np[:, None], axis=0)

    def _map_primitive_to_env_action(self, primitive: np.ndarray) -> np.ndarray:
        if self.mode == "ensemble_absolute":
            return np.clip((primitive + 1.0) * 256.0, 0.0, 512.0)
        if self.mode == "ensemble_delta":
            return np.clip(self.current_eef + primitive * self.max_step_pixels, 0.0, 512.0)
        if self.mode == "ensemble_relative":
            return np.asarray(primitive, dtype=np.float32)
        raise ValueError(f"Unsupported temporal ensemble mode: {self.mode}")

    def step(self, macro_action):
        macro_action = np.asarray(macro_action, dtype=np.float32)
        if self.clip_actions:
            macro_action = np.clip(macro_action, -1.0, 1.0)
        primitives = macro_action.reshape(self.chunk_size, 2)
        self.pending_chunks.append({"chunk": primitives, "offset": 0})

        primitive = self._ensemble_primitive()
        env_action = self._map_primitive_to_env_action(primitive)
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        state = extract_state_array(obs)
        self.current_eef = state[0:2].copy()

        for entry in self.pending_chunks:
            entry["offset"] = int(entry["offset"]) + 1
        while self.pending_chunks and int(self.pending_chunks[0]["offset"]) >= self.chunk_size:
            self.pending_chunks.popleft()

        info = dict(info)
        info["executed_actions"] = np.asarray([env_action], dtype=np.float32)
        info["executed_primitives"] = np.asarray([primitive], dtype=np.float32)
        info["num_active_chunks"] = len(self.pending_chunks)
        return obs, float(reward), terminated, truncated, info


class BCPriorRenderAdapter(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def actor_mean(self, image_obs: torch.Tensor) -> torch.Tensor:
        actions = self.policy(image_obs)
        if actions.ndim == 3:
            actions = actions[:, -1, :]
        return actions


def make_render_env(config: TrainConfig, video_folder: str, action_mode: str, record_video: bool = True, clip_actions: bool = True):
    tokenizer_info = None
    if config.network_type == "bc_latent":
        _, tokenizer_info = load_tokenizer_from_ckpt(config.tokenizer_path, torch.device("cpu"))

    resolved_env_id = resolve_pusht_env_id(config.env_id)
    image_height = int(tokenizer_info["H"]) if tokenizer_info is not None else int(getattr(config, "image_height", 224))
    image_width = int(tokenizer_info["W"]) if tokenizer_info is not None else int(getattr(config, "image_width", 224))
    relative = action_mode in {"first_relative", "chunk_relative", "ensemble_relative"}
    env = make_pusht_env(
        env_id=resolved_env_id,
        render_mode="rgb_array",
        image_height=image_height,
        image_width=image_width,
        relative=relative,
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=True,
        render_obs=False,
    )
    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix="ppo_pusht_test",
            episode_trigger=lambda episode_id: True,
            disable_logger=True,
        )
    env = PushTDenseRewardWrapper(env, env_id=resolved_env_id)
    if action_mode == "ppo_chunk":
        env = ActionChunkingTemporalEnsembleWrapper(
            env,
            chunk_size=config.chunk_size,
            max_step_pixels=config.max_step_pixels,
            ensemble_decay=config.ensemble_decay,
        )
    elif action_mode.startswith("ensemble_"):
        env = TemporalEnsembleChunkExecutionWrapper(
            env,
            chunk_size=config.chunk_size,
            mode=action_mode,
            max_step_pixels=config.max_step_pixels,
            ensemble_decay=config.ensemble_decay,
            clip_actions=clip_actions,
        )
    else:
        env = ChunkExecutionWrapper(
            env,
            chunk_size=config.chunk_size,
            mode=action_mode,
            max_step_pixels=config.max_step_pixels,
            clip_actions=clip_actions,
        )
    if config.network_type == "bc_pixels":
        env = RenderedImageObsWrapper(env, target_height=image_height, target_width=image_width)
        env = StridedObservationStackWrapper(
            env,
            stack_size=config.obs_stack_size,
            frame_stride=int(getattr(config, "frame_stride", 1)),
        )
    elif config.network_type == "bc_latent":
        env = TokenizerLatentObsWrapper(
            env,
            tokenizer_ckpt=config.tokenizer_path,
            tokenizer_device=config.tokenizer_device,
        )
        env = StridedObservationStackWrapper(
            env,
            stack_size=config.obs_stack_size,
            frame_stride=int(getattr(config, "frame_stride", 1)),
        )
    else:
        env = PushTObsWrapper(env)
    return env


def count_env_steps(info: dict) -> int:
    executed_actions = info.get("executed_actions")
    if executed_actions is None:
        return 1
    try:
        return max(1, int(len(executed_actions)))
    except TypeError:
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description="Render PPO or BC-prior policy on PushT.")
    parser.add_argument(
        "--source",
        choices=("ppo", "bc_prior"),
        default="bc_prior",
        help="Which weights to render.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional override checkpoint path. Defaults to TrainConfig save_path or bc_prior_path based on --source.",
    )
    parser.add_argument(
        "--action-mode",
        choices=(
            "auto",
            "ppo_chunk",
            "first_absolute",
            "first_delta",
            "first_relative",
            "chunk_absolute",
            "chunk_delta",
            "chunk_relative",
            "ensemble_absolute",
            "ensemble_delta",
            "ensemble_relative",
        ),
        default="auto",
        help="How to interpret the predicted action chunk during rendering.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording for faster debugging.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum number of env steps to run before stopping. 0 means no limit.",
    )
    parser.add_argument(
        "--max-step-pixels",
        type=float,
        default=None,
        help="Optional override for delta action scaling.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print rollout progress every N policy decisions. Default 0 disables intermediate logs.",
    )
    return parser.parse_args()


def render_agent_to_video():
    args = parse_args()
    config = TrainConfig()
    config.bc_prior_path = resolve_bc_prior_path(config.bc_prior_path)
    config.save_path = resolve_ppo_checkpoint_path(config.save_path)
    if args.max_step_pixels is not None:
        config.max_step_pixels = float(args.max_step_pixels)
    device = resolve_device(config.device)
    config.tokenizer_device = str(resolve_device(config.tokenizer_device))
    model_path = args.model_path or (config.bc_prior_path if args.source == "bc_prior" else config.save_path)
    if args.source == "bc_prior":
        model_path = resolve_bc_prior_path(model_path)
    else:
        model_path = resolve_ppo_checkpoint_path(model_path)

    if args.source == "bc_prior":
        from behavioural_cloning.eval_bc_exact import BCImagePolicy, clean_state_dict_keys, resolve_model_config

        bc_payload = load_state_dict_safe(model_path, torch.device("cpu"))
        bc_args = extract_checkpoint_args(bc_payload)
        bc_state_dict = extract_checkpoint_state_dict(bc_payload)
        if bc_state_dict is None:
            raise ValueError(f"Unsupported BC prior format in {model_path}")
        cleaned_bc_state = clean_state_dict_keys(bc_state_dict)
        bc_model_cfg = resolve_model_config(bc_args, cleaned_bc_state)
        tokenizer_name = bc_args.get("tokenizer_ckpt_name")
        policy_style = str(bc_model_cfg.get("policy_style", "sequence_classifier"))
        clip_bc_prior_actions = bool(bc_model_cfg.get("action_output_tanh", True))
        config.network_type = "bc_pixels"
        if tokenizer_name:
            config.tokenizer_path = resolve_tokenizer_path(str(tokenizer_name))
        config.image_height = int(bc_args.get("H", 224))
        config.image_width = int(bc_args.get("W", 224))
        if bc_args.get("seq_len") is not None:
            config.obs_stack_size = int(bc_args["seq_len"])
        config.frame_stride = int(bc_args.get("frame_stride", 1))
        if bc_args.get("action_chunk_size") is not None:
            config.chunk_size = int(bc_args["action_chunk_size"])
        if bc_args.get("hidden_dim") is not None:
            config.actor_hidden_dim = int(bc_args["hidden_dim"])
        if bc_args.get("dropout") is not None:
            config.actor_dropout = float(bc_args["dropout"])
        temporal_layers = int(bc_args.get("temporal_layers", 2))
        temporal_heads = int(bc_args.get("temporal_heads", 4))
        temporal_context = int(bc_args.get("temporal_context", 3))
    else:
        policy_style = "sequence_classifier"
        clip_bc_prior_actions = True
        config.tokenizer_path = resolve_tokenizer_path(config.tokenizer_path)
        temporal_layers = 2
        temporal_heads = 4
        temporal_context = 3
        config.frame_stride = int(getattr(config, "frame_stride", 1))

    action_mode = args.action_mode
    resolved_env_id = resolve_pusht_env_id(config.env_id)
    if action_mode == "auto":
        action_mode = "ensemble_relative" if args.source == "bc_prior" else "ppo_chunk"
    video_folder = "./videos"
    record_video = not args.no_video
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
    print(
        f"Render paths: source={args.source} model={model_path} "
        f"tokenizer={config.tokenizer_path} action_mode={action_mode} "
        f"record_video={record_video} max_steps={args.max_steps} "
        f"max_step_pixels={config.max_step_pixels} "
        f"network_type={config.network_type} obs_stack={config.obs_stack_size} "
        f"frame_stride={getattr(config, 'frame_stride', 1)} chunk_size={config.chunk_size}"
    )

    env = make_render_env(
        config,
        video_folder,
        action_mode=action_mode,
        record_video=record_video,
        clip_actions=(clip_bc_prior_actions if args.source == "bc_prior" else True),
    )

    obs_shape = tuple(env.observation_space.shape)
    state_dim = int(obs_shape[-1]) if config.network_type == "bc_latent" else int(np.prod(obs_shape))
    action_dim = int(env.action_space.shape[0])

    if args.source == "bc_prior":
        network = BCPriorRenderAdapter(
            BCImagePolicy(
                image_shape=obs_shape[1:] + (obs_shape[-1],) if False else (config.image_height, config.image_width, 3),
                hidden_dim=bc_model_cfg["hidden_dim"],
                dropout=bc_model_cfg["dropout"],
                action_chunk_size=bc_model_cfg["action_chunk_size"],
                seq_len=bc_model_cfg["seq_len"],
                tokenizer_ckpt=config.tokenizer_path,
                tokenizer_feature_dim=bc_model_cfg["tokenizer_feature_dim"],
                policy_style=bc_model_cfg["policy_style"],
                temporal_layers=bc_model_cfg["temporal_layers"],
                temporal_heads=bc_model_cfg["temporal_heads"],
                temporal_context=bc_model_cfg["temporal_context"],
                backbone_device=device,
                action_output_tanh=bc_model_cfg["action_output_tanh"],
            )
        ).to(device)
        network.policy.load_state_dict(cleaned_bc_state, strict=True)
        print(f"Loaded BC prior from {model_path}.")
    elif config.network_type == "bc_pixels":
        network = BCPixelActorCritic(
            image_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=config.actor_hidden_dim,
            dropout=config.actor_dropout,
            policy_style=policy_style,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            temporal_context=temporal_context,
        ).to(device)
    elif config.network_type == "bc_latent":
        network = TokenizerLatentBCPPOActorCritic(
            feature_dim=state_dim,
            frame_stack=obs_shape[0] if len(obs_shape) >= 1 else 1,
            action_dim=action_dim,
            action_chunk_size=config.chunk_size,
            hidden_dim=config.actor_hidden_dim,
            init_log_std=config.init_log_std,
        ).to(device)
    else:
        network = VectorActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_output_tanh=config.actor_output_tanh,
        ).to(device)

    if args.source != "bc_prior":
        payload = load_state_dict_safe(model_path, device)
        state_dict = extract_checkpoint_state_dict(payload)
        if state_dict is None:
            raise ValueError(f"Unsupported PPO checkpoint format in {model_path}")
        network.load_state_dict(state_dict)
        print(f"Loaded PPO checkpoint from {model_path}.")
    network.eval()

    state, _ = env.reset()
    done = False
    step_count = 0
    decision_count = 0
    total_reward = 0.0
    final_info = {}

    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_mean = network.actor_mean(state_tensor)

        env_action = action_mean.squeeze(0).cpu().numpy()
        if args.source != "bc_prior" or clip_bc_prior_actions:
            env_action = np.clip(env_action, -1.0, 1.0)
        state, reward, terminated, truncated, info = env.step(env_action)
        final_info = info
        total_reward += float(reward)
        done = terminated or truncated
        decision_count += 1
        step_count += count_env_steps(info)
        if args.print_every > 0 and (
            decision_count % args.print_every == 0 or done or (args.max_steps > 0 and step_count >= args.max_steps)
        ):
            coverage = info.get("coverage")
            coverage_proxy = info.get("coverage_proxy")
            coverage_text = "n/a" if coverage is None else f"{float(coverage):.3f}"
            proxy_text = "n/a" if coverage_proxy is None else f"{float(coverage_proxy):.3f}"
            print(
                f"decision={decision_count:04d} env_steps={step_count:04d} "
                f"reward={float(reward):8.3f} total_reward={total_reward:8.3f} "
                f"coverage={coverage_text} coverage_proxy={proxy_text}"
            )
        if args.max_steps > 0 and step_count >= args.max_steps:
            print(f"Stopped early after reaching max_steps={args.max_steps}.")
            break

    print(f"Episode length: {step_count}")
    print(f"Policy decisions: {decision_count}")
    print(f"Total dense reward: {total_reward:.3f}")
    if "coverage" in final_info:
        print(f"Coverage: {float(final_info['coverage']):.3f}")
    print(f"Rendered source: {args.source}")
    print(f"Action interpretation: {action_mode}")
    if record_video:
        print(f"Video saved to '{video_folder}'.")
    else:
        print("Video recording disabled.")

    env.close()


if __name__ == "__main__":
    render_agent_to_video()
