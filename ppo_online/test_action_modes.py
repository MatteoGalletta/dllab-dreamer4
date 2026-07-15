#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np
import torch

from ppo_online.model_paths import resolve_bc_prior_path, resolve_ppo_checkpoint_path, resolve_tokenizer_path
from ppo_online.networks import BCPixelActorCritic, BCStyleLatentActorCritic, VectorActorCritic
from ppo_online.render import load_bc_prior_into_network, load_state_dict_safe, make_render_env
from ppo_online.train import TrainConfig, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple action interpretation modes for quick PushT BC/PPO debugging.")
    parser.add_argument("--source", choices=("ppo", "bc_prior"), default="bc_prior")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-step-pixels", type=float, default=None)
    parser.add_argument("--with-video", action="store_true", help="Record videos during the sweep.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["first_absolute", "first_delta", "chunk_absolute", "chunk_delta"],
        help="Action interpretation modes to test.",
    )
    return parser.parse_args()


def build_network(config: TrainConfig, obs_shape: tuple[int, ...], action_dim: int, device: torch.device):
    state_dim = int(obs_shape[-1]) if config.network_type == "bc_latent" else int(np.prod(obs_shape))

    if config.network_type == "bc_pixels":
        return BCPixelActorCritic(
            image_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=config.actor_hidden_dim,
            dropout=config.actor_dropout,
        ).to(device)
    if config.network_type == "bc_latent":
        return BCStyleLatentActorCritic(
            feature_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.actor_hidden_dim,
            dropout=config.actor_dropout,
        ).to(device)
    return VectorActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_output_tanh=config.actor_output_tanh,
    ).to(device)


def count_env_steps(info: dict) -> int:
    executed_actions = info.get("executed_actions")
    if executed_actions is None:
        return 1
    try:
        return max(1, int(len(executed_actions)))
    except TypeError:
        return 1


def run_single_mode(config: TrainConfig, args, mode: str, model_path: str, device: torch.device):
    env = make_render_env(
        config,
        video_folder="./videos",
        action_mode=mode,
        record_video=args.with_video,
    )

    obs_shape = tuple(env.observation_space.shape)
    action_dim = int(env.action_space.shape[0])
    network = build_network(config, obs_shape, action_dim, device)

    if args.source == "bc_prior":
        load_bc_prior_into_network(network, model_path, device, config.network_type)
    else:
        network.load_state_dict(load_state_dict_safe(model_path, device))

    network.eval()
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    decision_count = 0
    final_info = {}

    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_mean = network.actor_mean(state_tensor)

        env_action = np.clip(action_mean.squeeze(0).cpu().numpy(), -1.0, 1.0)
        state, reward, terminated, truncated, info = env.step(env_action)
        total_reward += float(reward)
        final_info = info
        decision_count += 1
        step_count += count_env_steps(info)
        done = terminated or truncated
        if args.max_steps > 0 and step_count >= args.max_steps:
            break

    env.close()
    coverage = final_info.get("coverage")
    coverage_proxy = final_info.get("coverage_proxy")
    return {
        "mode": mode,
        "steps": step_count,
        "decisions": decision_count,
        "reward": total_reward,
        "coverage": float(coverage) if coverage is not None else None,
        "coverage_proxy": float(coverage_proxy) if coverage_proxy is not None else None,
    }


def main():
    args = parse_args()
    config = TrainConfig()
    config.bc_prior_path = resolve_bc_prior_path(config.bc_prior_path)
    config.tokenizer_path = resolve_tokenizer_path(config.tokenizer_path)
    config.save_path = resolve_ppo_checkpoint_path(config.save_path)
    if args.max_step_pixels is not None:
        config.max_step_pixels = float(args.max_step_pixels)
    config.tokenizer_device = str(resolve_device(config.tokenizer_device))
    device = resolve_device(config.device)

    model_path = args.model_path or (config.bc_prior_path if args.source == "bc_prior" else config.save_path)
    if args.source == "bc_prior":
        model_path = resolve_bc_prior_path(model_path)
    else:
        model_path = resolve_ppo_checkpoint_path(model_path)

    print(
        f"Testing source={args.source} model={model_path} "
        f"max_steps={args.max_steps} max_step_pixels={config.max_step_pixels}"
    )
    for mode in args.modes:
        result = run_single_mode(config, args, mode, model_path, device)
        coverage_text = "n/a" if result["coverage"] is None else f"{result['coverage']:.3f}"
        proxy_text = None if result["coverage_proxy"] is None else f"{result['coverage_proxy']:.3f}"
        print(
            f"{result['mode']:>14} | steps={result['steps']:>4} "
            f"| decisions={result['decisions']:>4} "
            f"| reward={result['reward']:>10.3f} | coverage={coverage_text}"
            f"{'' if proxy_text is None else f' | coverage_proxy={proxy_text}'}"
        )


if __name__ == "__main__":
    main()
