from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class VectorActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, actor_output_tanh: bool = True):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)

        actor_layers: list[nn.Module] = [
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        ]
        if actor_output_tanh:
            actor_layers.append(nn.Tanh())
        self.actor = nn.Sequential(*actor_layers)

        self.log_std = nn.Parameter(torch.full((1, action_dim), -0.5))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def actor_mean(self, state_vector: torch.Tensor) -> torch.Tensor:
        flat_state = self.flatten(state_vector)
        return self.actor(flat_state)

    def action_dist(self, state_vector: torch.Tensor) -> Normal:
        mean = self.actor_mean(state_vector)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(self, state_vector: torch.Tensor, action: torch.Tensor | None = None):
        distribution = self.action_dist(state_vector)
        if action is None:
            action = distribution.sample()
        value = self.get_value(state_vector)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_value(self, state_vector: torch.Tensor) -> torch.Tensor:
        flat_state = self.flatten(state_vector)
        return self.critic(flat_state)


class BCActionClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, action_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features_btD: torch.Tensor) -> torch.Tensor:
        if features_btD.ndim == 2:
            features_btD = features_btD.unsqueeze(1)
        if features_btD.ndim != 3:
            raise ValueError(f"Expected latent sequence with shape (B, T, D), got {tuple(features_btD.shape)}")
        B, T, D = features_btD.shape
        logits = self.net(features_btD.reshape(B * T, D))
        actions = torch.tanh(logits)
        return actions.view(B, T, -1)


class BCStyleLatentActorCritic(nn.Module):
    """
    PPO actor-critic that matches the BC classifier head while consuming
    precomputed tokenizer latents instead of raw frames.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.classifier = BCActionClassifier(
            in_dim=feature_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.full((1, action_dim), -1.0))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _ensure_sequence(self, state_vector: torch.Tensor) -> torch.Tensor:
        if state_vector.ndim == 2:
            return state_vector.unsqueeze(1)
        if state_vector.ndim == 3:
            return state_vector
        raise ValueError(f"Expected state shape (B, D) or (B, T, D), got {tuple(state_vector.shape)}")

    def _last_features(self, state_vector: torch.Tensor) -> torch.Tensor:
        sequence = self._ensure_sequence(state_vector)
        return sequence[:, -1, :]

    def actor_mean(self, state_vector: torch.Tensor) -> torch.Tensor:
        sequence = self._ensure_sequence(state_vector)
        action_sequence = self.classifier(sequence)
        return action_sequence[:, -1, :]

    def action_dist(self, state_vector: torch.Tensor) -> Normal:
        mean = self.actor_mean(state_vector)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(self, state_vector: torch.Tensor, action: torch.Tensor | None = None):
        distribution = self.action_dist(state_vector)
        if action is None:
            action = distribution.sample()
        value = self.get_value(state_vector)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_value(self, state_vector: torch.Tensor) -> torch.Tensor:
        return self.critic(self._last_features(state_vector))
