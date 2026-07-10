#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation

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
        for prefix in ("module.", "network."):
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


def make_render_env(config: TrainConfig, video_folder: str):
    tokenizer_info = None
    if config.network_type in {"bc_pixels", "bc_latent"}:
        _, tokenizer_info = load_tokenizer_from_ckpt(config.tokenizer_path, torch.device("cpu"))

    env_kwargs = {
        "obs_type": "state",
        "render_mode": "rgb_array",
    }
    if tokenizer_info is not None:
        env_kwargs["observation_width"] = int(tokenizer_info["W"])
        env_kwargs["observation_height"] = int(tokenizer_info["H"])

    env = gym.make(
        "gym_pusht/PushT-v0",
        **env_kwargs,
    )
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="ppo_pusht_test",
        episode_trigger=lambda episode_id: True,
        disable_logger=True,
    )
    env = PushTDenseRewardWrapper(env)
    env = ActionChunkingTemporalEnsembleWrapper(
        env,
        chunk_size=config.chunk_size,
        max_step_pixels=config.max_step_pixels,
        ensemble_decay=config.ensemble_decay,
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
    return parser.parse_args()


def render_agent_to_video():
    args = parse_args()
    config = TrainConfig()
    config.bc_prior_path = resolve_bc_prior_path(config.bc_prior_path)
    config.tokenizer_path = resolve_tokenizer_path(config.tokenizer_path)
    config.save_path = resolve_ppo_checkpoint_path(config.save_path)
    device = resolve_device(config.device)
    config.tokenizer_device = str(resolve_device(config.tokenizer_device))
    model_path = args.model_path or (config.bc_prior_path if args.source == "bc_prior" else config.save_path)
    if args.source == "bc_prior":
        model_path = resolve_bc_prior_path(model_path)
    else:
        model_path = resolve_ppo_checkpoint_path(model_path)
    video_folder = "./videos"
    os.makedirs(video_folder, exist_ok=True)
    print(f"Render paths: source={args.source} model={model_path} tokenizer={config.tokenizer_path}")

    env = make_render_env(config, video_folder)

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
        step_count += 1

    print(f"Episode length: {step_count}")
    print(f"Total dense reward: {total_reward:.3f}")
    if "coverage" in final_info:
        print(f"Coverage: {float(final_info['coverage']):.3f}")
    print(f"Rendered source: {args.source}")
    print(f"Video saved to '{video_folder}'.")

    env.close()


if __name__ == "__main__":
    render_agent_to_video()
