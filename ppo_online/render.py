#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation

from ppo_online.env_config import make_pusht_env_kwargs, resolve_pusht_env_id
from ppo_online.model_paths import resolve_bc_prior_path, resolve_ppo_checkpoint_path, resolve_tokenizer_path
from ppo_online.networks import BCPixelActorCritic, BCStyleLatentActorCritic, VectorActorCritic
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
        for key in ("state_dict", "model_state_dict", "network", "model", "actor"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                state_dict = extract_checkpoint_state_dict(nested)
                if state_dict is not None:
                    return state_dict
    return None


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
        classifier_state = {
            key[len("classifier.") :]: value
            for key, value in cleaned.items()
            if key.startswith("classifier.")
        }
        if not classifier_state:
            raise ValueError(f"No classifier weights found in BC prior {path}")
        incompatible = network.classifier.load_state_dict(classifier_state, strict=False)
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

    def __init__(self, env: gym.Env, chunk_size: int, mode: str, max_step_pixels: float):
        super().__init__(env)
        self.chunk_size = chunk_size
        self.mode = mode
        self.max_step_pixels = max_step_pixels
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
        primitives = np.clip(macro_action, -1.0, 1.0).reshape(self.chunk_size, 2)

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


def make_render_env(config: TrainConfig, video_folder: str, action_mode: str, record_video: bool = True):
    tokenizer_info = None
    if config.network_type in {"bc_pixels", "bc_latent"}:
        _, tokenizer_info = load_tokenizer_from_ckpt(config.tokenizer_path, torch.device("cpu"))

    resolved_env_id = resolve_pusht_env_id(config.env_id)
    image_height = int(tokenizer_info["H"]) if tokenizer_info is not None else None
    image_width = int(tokenizer_info["W"]) if tokenizer_info is not None else None
    env_kwargs = make_pusht_env_kwargs(
        resolved_env_id,
        render_mode="rgb_array",
        image_height=image_height,
        image_width=image_width,
    )
    if resolved_env_id.startswith("swm/") and action_mode in {"first_relative", "chunk_relative"}:
        env_kwargs["relative"] = True

    env = gym.make(resolved_env_id, **env_kwargs)
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
    else:
        env = ChunkExecutionWrapper(
            env,
            chunk_size=config.chunk_size,
            mode=action_mode,
            max_step_pixels=config.max_step_pixels,
        )
    if config.network_type == "bc_pixels":
        env = RenderedImageObsWrapper(env, tokenizer_ckpt=config.tokenizer_path)
        env = FrameStackObservation(env, stack_size=config.obs_stack_size)
    elif config.network_type == "bc_latent":
        env = TokenizerLatentObsWrapper(
            env,
            tokenizer_ckpt=config.tokenizer_path,
            tokenizer_device=config.tokenizer_device,
        )
        env = FrameStackObservation(env, stack_size=config.obs_stack_size)
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
    config.tokenizer_path = resolve_tokenizer_path(config.tokenizer_path)
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
    action_mode = args.action_mode
    resolved_env_id = resolve_pusht_env_id(config.env_id)
    if action_mode == "auto":
        if args.source == "bc_prior" and resolved_env_id.startswith("swm/"):
            action_mode = "chunk_relative"
        else:
            action_mode = "chunk_delta" if args.source == "bc_prior" else "ppo_chunk"
    video_folder = "./videos"
    record_video = not args.no_video
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
    print(
        f"Render paths: source={args.source} model={model_path} "
        f"tokenizer={config.tokenizer_path} action_mode={action_mode} "
        f"record_video={record_video} max_steps={args.max_steps} "
        f"max_step_pixels={config.max_step_pixels}"
    )

    env = make_render_env(config, video_folder, action_mode=action_mode, record_video=record_video)

    obs_shape = tuple(env.observation_space.shape)
    state_dim = int(obs_shape[-1]) if config.network_type == "bc_latent" else int(np.prod(obs_shape))
    action_dim = int(env.action_space.shape[0])

    if config.network_type == "bc_pixels":
        network = BCPixelActorCritic(
            image_shape=obs_shape,
            action_dim=action_dim,
            tokenizer_ckpt=config.tokenizer_path,
            hidden_dim=config.actor_hidden_dim,
            dropout=config.actor_dropout,
            backbone_device=config.tokenizer_device,
        ).to(device)
    elif config.network_type == "bc_latent":
        network = BCStyleLatentActorCritic(
            feature_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.actor_hidden_dim,
            dropout=config.actor_dropout,
            max_seq_len=obs_shape[0] if len(obs_shape) >= 1 else 64,
        ).to(device)
    else:
        network = VectorActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_output_tanh=config.actor_output_tanh,
        ).to(device)

    if args.source == "bc_prior":
        load_bc_prior_into_network(network, model_path, device, config.network_type)
    else:
        state_dict = load_state_dict_safe(model_path, device)
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

        env_action = np.clip(action_mean.squeeze(0).cpu().numpy(), -1.0, 1.0)
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
