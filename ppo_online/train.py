#!/usr/bin/env python3

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import gym_pusht
import numpy as np
import torch
import wandb
from gymnasium.wrappers import FrameStackObservation

from .agent import PPOAgent
from .buffer import PPOVectorBuffer
from .tokenizer_utils import TokenizerZEncoder


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
        obs = np.asarray(obs, dtype=np.float32)
        self.current_eef = obs[0:2].copy()
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
        obs = np.asarray(obs, dtype=np.float32)
        self.current_eef = obs[0:2].copy()

        for entry in self.pending_chunks:
            entry["offset"] += 1
        while self.pending_chunks and self.pending_chunks[0]["offset"] >= self.chunk_size:
            self.pending_chunks.popleft()

        info = dict(info)
        info["executed_delta"] = delta.astype(np.float32)
        info["num_active_chunks"] = len(self.pending_chunks)

        return obs, reward, terminated, truncated, info


class PushTDenseRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.target_pos = np.array([256.0, 256.0], dtype=np.float32)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)

        eef_pos = observation[0:2].astype(np.float32)
        block_pos = observation[2:4].astype(np.float32)

        dist_reach = float(np.linalg.norm(eef_pos - block_pos))
        dist_push = float(np.linalg.norm(block_pos - self.target_pos))

        r_push = math.exp(-dist_push / 100.0)
        dist_reach_eff = max(0.0, dist_reach - 60.0)
        r_reach = math.exp(-dist_reach_eff / 100.0)

        dense_reward = (1.0 * r_reach) + (3.0 * r_push) + (50.0 * float(original_reward))

        info = dict(info)
        info["dense_reward"] = float(dense_reward)
        info["original_reward"] = float(original_reward)
        info["coverage"] = float(original_reward)

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
        observation = np.asarray(observation, dtype=np.float32)
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


def add_timeout_bootstrap_rewards(rewards, terminations, truncations, infos, agent, device, gamma):
    rewards = np.asarray(rewards, dtype=np.float32).copy()
    timeout_mask = np.logical_and(truncations, np.logical_not(terminations))
    timeout_indices = np.where(timeout_mask)[0]

    if len(timeout_indices) == 0:
        return rewards, 0

    final_obs_list, valid_indices = [], []
    for idx in timeout_indices:
        final_obs = extract_final_observation(infos, int(idx))
        if final_obs is not None:
            final_obs_list.append(final_obs)
            valid_indices.append(int(idx))

    missing_count = len(timeout_indices) - len(valid_indices)
    if not final_obs_list:
        return rewards, missing_count

    final_obs_tensor = torch.as_tensor(np.stack(final_obs_list), dtype=torch.float32, device=device)
    with torch.no_grad():
        final_values = agent.network.get_value(final_obs_tensor).squeeze(-1)

    final_values_np = final_values.detach().cpu().numpy().astype(np.float32)
    for idx, value in zip(valid_indices, final_values_np):
        rewards[idx] += gamma * float(value)

    return rewards, missing_count


@dataclass
class TrainConfig:
    num_envs: int = 16
    rollout_steps: int = 256
    total_timesteps: int = 10_000_000
    batch_size: int = 1024
    ppo_epochs: int = 10
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    target_kl: float | None = 0.03
    max_grad_norm: float = 0.5

    chunk_size: int = 5
    max_step_pixels: float = 15.0
    ensemble_decay: float = 0.35
    actor_output_tanh: bool = True
    init_log_std: float = -1.0
    obs_stack_size: int = 8
    network_type: str = "bc_latent"
    actor_hidden_dim: int = 512
    actor_dropout: float = 0.05

    bc_prior_path: str = "ppo_online/bc_prior.pt"
    prior_loss_coef: float = 0.05
    prior_loss_decay: float = 0.997
    tokenizer_path: str = "ppo_online/tokenizer.pt"
    tokenizer_device: str = "auto"

    seed: int = 42
    device: str = "auto"
    vector_env: str = "sync"
    save_path: str = "ppo_pusht_model.pth"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
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
        env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode or "rgb_array")
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
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return env

    return _thunk


def train_pusht():
    config = TrainConfig()
    policy_device = resolve_device(config.device)
    tokenizer_device = resolve_device(config.tokenizer_device)
    config.device = str(policy_device)
    config.tokenizer_device = str(tokenizer_device)
    wandb.init(
        project="pusht-ppo",
        name=f"ppo_bc_chunk{config.chunk_size}_seed{config.seed}",
        config=config.__dict__,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = policy_device
    print(f"Training starts on policy_device={device}, tokenizer_device={tokenizer_device}")

    env_fns = [make_env(rank, config.seed, config) for rank in range(config.num_envs)]
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
    state_dim = int(obs_shape[-1]) if config.network_type == "bc_latent" else int(np.prod(obs_shape))
    action_dim = int(envs.single_action_space.shape[0])

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
        prior_log_std_init=config.init_log_std,
        device=device,
        network_type=config.network_type,
        actor_hidden_dim=config.actor_hidden_dim,
        actor_dropout=config.actor_dropout,
    )
    print(agent.prior_load_info.message)

    buffer = PPOVectorBuffer(
        buffer_size=config.rollout_steps,
        num_envs=config.num_envs,
        state_shape=obs_shape,
        action_dim=action_dim,
        device=device,
    )

    num_updates = config.total_timesteps // (config.num_envs * config.rollout_steps)
    last_dones_for_gae = np.zeros(config.num_envs, dtype=np.float32)
    warned_missing_timeout_bootstrap = False

    for update in range(num_updates):
        frac = 1.0 - (update / max(1, num_updates))
        lr_now = frac * config.learning_rate
        agent.optimizer.param_groups[0]["lr"] = lr_now
        agent.ent_coef = frac * config.ent_coef

        buffer.clear()
        reward_sum = 0.0
        reward_count = 0
        action_abs_max = 0.0
        rollout_returns = []
        rollout_coverages = []
        rollout_chunks = []

        for _ in range(config.rollout_steps):
            state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions, logprobs, _, values = agent.network.get_action_and_value(state_tensor)

            actions_np = actions.detach().cpu().numpy().astype(np.float32)
            env_actions = np.clip(actions_np, -1.0, 1.0)
            action_abs_max = max(action_abs_max, float(np.max(np.abs(env_actions))))

            next_states, rewards, terminations, truncations, infos = envs.step(env_actions)

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
            elif "final_info" in infos:
                for final_info in infos["final_info"]:
                    if final_info is not None and "episode" in final_info:
                        ep_return = as_float(final_info["episode"].get("r", 0.0))
                        rollout_returns.append(ep_return)
                        if "coverage" in final_info:
                            rollout_coverages.append(float(final_info["coverage"]))

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
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        stats = agent.update(
            buffer,
            advantages.flatten(),
            returns.flatten(),
            batch_size=config.batch_size,
            ppo_epochs=config.ppo_epochs,
            update_idx=update,
        )

        global_step = (update + 1) * config.num_envs * config.rollout_steps
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
            "PPO/Prior_Loss_Coef": stats["prior_loss_coef"],
            "Hyperparameters/Learning_Rate": lr_now,
            "Hyperparameters/Entropy_Coef": agent.ent_coef,
        }

        if rollout_returns:
            wandb_log_dict["Environment/Mean_Episode_Return"] = sum(rollout_returns) / len(rollout_returns)
        if rollout_coverages:
            wandb_log_dict["Environment/Mean_Coverage"] = sum(rollout_coverages) / len(rollout_coverages)
            wandb_log_dict["Environment/Max_Coverage"] = max(rollout_coverages)
        if rollout_chunks:
            wandb_log_dict["Chunking/Mean_Active_Chunks"] = sum(rollout_chunks) / len(rollout_chunks)

        wandb.log(wandb_log_dict, step=global_step)

        if (update + 1) % 50 == 0:
            torch.save(agent.network.state_dict(), config.save_path)
            wandb.save(config.save_path)
            print(
                f"update={update + 1}/{num_updates} "
                f"return={wandb_log_dict.get('Environment/Mean_Episode_Return', float('nan')):.2f} "
                f"coverage={wandb_log_dict.get('Environment/Mean_Coverage', float('nan')):.3f}"
            )

    print("Saving final model weights...")
    torch.save(agent.network.state_dict(), config.save_path)
    wandb.save(config.save_path)
    print("Training finished.")
    envs.close()
    wandb.finish()


if __name__ == "__main__":
    train_pusht()
