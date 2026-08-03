import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "dreamer4-src"))

from dreamer4.train_dynamics import make_tau_schedule, sample_one_timestep_packed


from model import (
    temporal_patchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck
)
#from train_dynamics import make_tau_schedule, sample_one_timestep_packed


class LatentWorldModelEnv:
    """
    An imagination environment that lives entirely within the packed latent space
    of the Tokenizer, using the Dynamics Flow Model for state transitions and the
    Reward Head for dense reward calculation. Ideal for PPO rollouts.
    """

    def __init__(
            self,
            encoder: nn.Module,
            dynamics: nn.Module,
            reward_head: nn.Module,
            k_max: int = 8,
            packing_factor: int = 2,
            eval_schedule: str = "shortcut",
            eval_d: float = 0.25,
            device: torch.device = torch.device("cuda")
    ):
        self.encoder = encoder.eval()
        self.dynamics = dynamics.eval()
        self.reward_head = reward_head.eval()

        self.k_max = k_max
        self.packing_factor = packing_factor
        self.device = device

        # Build the ODE integration schedule for the flow matching dynamics
        self.sched = make_tau_schedule(k_max=self.k_max, schedule=eval_schedule, d=eval_d)

        # Internal trajectory buffers
        self.past_packed_buffer = None  # (B, t, n_spatial, d_spatial)
        self.past_actions_buffer = None  # (B, t, 16)
        self.act_mask = None  # (16,)

    @torch.no_grad()
    def reset(self, initial_frames: torch.Tensor, patch: int = 4) -> torch.Tensor:
        """
        Resets the imagination environment with real context frames.
        Args:
            initial_frames: (B, T_ctx, C, H, W) normalized float [0, 1]
            patch: patch size for the encoder
        Returns:
            latest_latent: (B, n_spatial, d_spatial) - The current packed latent state
        """
        B, T_ctx = initial_frames.shape[:2]
        initial_frames = initial_frames.to(self.device)

        # 1. Map frames through the spatial patchifier and encoder
        patches = temporal_patchify(initial_frames, patch)
        z_btLd, _ = self.encoder(patches)  # (B, T_ctx, n_latents, d_bottleneck)

        # 2. Determine spatial dimensions and pack them down
        n_latents = z_btLd.shape[2]
        assert n_latents % self.packing_factor == 0, "n_latents must be divisible by packing_factor"
        n_spatial = n_latents // self.packing_factor

        # (B, T_ctx, n_spatial, d_spatial)
        self.past_packed_buffer = pack_bottleneck_to_spatial(
            z_btLd, n_spatial=n_spatial, k=self.packing_factor
        )

        # 3. Initialize empty historical actions buffer
        self.past_actions_buffer = torch.zeros((B, 0, 16), device=self.device, dtype=torch.float32)
        self.act_mask = torch.ones(16, device=self.device, dtype=torch.float32)  # Full action padding mask

        # Return the most recent latent state in the batch context
        return self.past_packed_buffer[:, -1]

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advances the imagination environment by one step using the dynamics flow network.
        Args:
            action: (B, action_dim) raw actions from your PPO policy network
        Returns:
            next_latent: (B, n_spatial, d_spatial) - The predicted next environment latent state
            reward: (B, 1) - Extracted dense environment reward bounds [0, 1]
            done: (B, 1) - Boolean indicator tensor (all zeros for pure imagination)
        """
        B = action.shape[0]
        action = action.to(self.device).clamp(-1, 1)

        # Pad standard actions out to the structural vector length of 16
        padded_action = torch.zeros((B, 1, 16), device=self.device, dtype=torch.float32)
        padded_action[..., :action.shape[-1]] = action.unsqueeze(1)

        # Append action to tracking context sequence
        self.past_actions_buffer = torch.cat([self.past_actions_buffer, padded_action], dim=1)

        # 1. Autoregressively sample the next latent state timestep
        # Expects: (B, t, n_spatial, d_spatial) -> Returns: (B, n_spatial, d_spatial)
        next_latent = sample_one_timestep_packed(
            dyn=self.dynamics,
            past_packed=self.past_packed_buffer,
            k_max=self.k_max,
            sched=self.sched,
            actions=self.past_actions_buffer,
            act_mask=self.act_mask
        )

        # Append the new state prediction into the historical track buffer
        self.past_packed_buffer = torch.cat([self.past_packed_buffer, next_latent.unsqueeze(1)], dim=1)

        # 2. Extract Reward: Unpack spatial tokens back to pooled bottleneck latents for RewardHead
        # Shape output from unpack: (B, 1, n_latents, d_bottleneck)
        z_btLd = unpack_spatial_to_bottleneck(next_latent.unsqueeze(1), k=self.packing_factor)

        # Pool latents spatially over the n_latents dimension to match RewardHead requirements
        z_pooled = z_btLd.mean(dim=2)  # Yields (B, 1, d_bottleneck)

        # Feed pooled latents through RewardHead to extract task-success proxy metrics
        reward_logits = self.reward_head(z_pooled).squeeze(1)  # Yields (B, 1) raw logit scalar
        reward = torch.sigmoid(reward_logits)  # Map to a smooth [0, 1] dense space

        # Imagination tracks run until a fixed horizon limit inside the PPO loop
        done = torch.zeros((B, 1), device=self.device, dtype=torch.bool)

        return next_latent, reward, done