#!/usr/bin/env python3

from __future__ import annotations

import os

import gymnasium as gym
import gym_pusht
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation

from ppo_online.networks import BCStyleLatentActorCritic, VectorActorCritic
from ppo_online.train import (
    ActionChunkingTemporalEnsembleWrapper,
    PushTDenseRewardWrapper,
    TokenizerLatentObsWrapper,
    TrainConfig,
    resolve_device,
)


def load_state_dict_safe(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def make_render_env(config: TrainConfig, video_folder: str):
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
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
    env = TokenizerLatentObsWrapper(
        env,
        tokenizer_ckpt=config.tokenizer_path,
        tokenizer_device=config.tokenizer_device,
    )
    env = FrameStackObservation(env, stack_size=config.obs_stack_size)
    return env


def render_agent_to_video():
    config = TrainConfig()
    device = resolve_device(config.device)
    config.tokenizer_device = str(resolve_device(config.tokenizer_device))
    model_path = config.save_path
    video_folder = "./videos"
    os.makedirs(video_folder, exist_ok=True)

    env = make_render_env(config, video_folder)

    obs_shape = tuple(env.observation_space.shape)
    state_dim = int(obs_shape[-1]) if config.network_type == "bc_latent" else int(np.prod(obs_shape))
    action_dim = int(env.action_space.shape[0])

    if config.network_type == "bc_latent":
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

    state_dict = load_state_dict_safe(model_path, device)
    network.load_state_dict(state_dict)
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
    print(f"Video saved to '{video_folder}'.")

    env.close()


if __name__ == "__main__":
    render_agent_to_video()
