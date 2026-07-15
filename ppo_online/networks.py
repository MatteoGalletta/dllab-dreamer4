from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .tokenizer_utils import load_tokenizer_from_ckpt


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
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        action_dim: int,
        dropout: float,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.max_seq_len = int(max_seq_len)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        self.temporal_in = nn.Linear(in_dim, hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_seq_len, hidden_dim))
        self.temporal_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=temporal_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(0, int(temporal_layers)))
            ]
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_out = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.temporal_out.weight)
        nn.init.zeros_(self.temporal_out.bias)

    def _positional_encoding(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.max_seq_len:
            return self.temporal_pos[:, :seq_len, :]
        pos = self.temporal_pos.transpose(1, 2)
        pos = F.interpolate(pos, size=seq_len, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, features_btD: torch.Tensor) -> torch.Tensor:
        if features_btD.ndim == 2:
            features_btD = features_btD.unsqueeze(1)
        if features_btD.ndim != 3:
            raise ValueError(f"Expected latent sequence with shape (B, T, D), got {tuple(features_btD.shape)}")
        B, T, D = features_btD.shape
        base_logits = self.net(features_btD.reshape(B * T, D)).view(B, T, -1)
        temporal_features = self.temporal_in(features_btD) + self._positional_encoding(T)
        for block in self.temporal_blocks:
            temporal_features = block(temporal_features)
        temporal_logits = self.temporal_out(self.temporal_norm(temporal_features))
        logits = base_logits + temporal_logits
        actions = torch.tanh(logits)
        return actions.view(B, T, -1)


class TokenizerBackbone(nn.Module):
    def __init__(self, tokenizer_ckpt: str, device: torch.device | str = "cpu"):
        super().__init__()
        self.device_override = torch.device(device)
        tokenizer, info = load_tokenizer_from_ckpt(tokenizer_ckpt, self.device_override)
        self.encoder = tokenizer.encoder
        self.patch = int(info["patch"])
        self.image_height = int(info["H"])
        self.image_width = int(info["W"])
        self.feature_dim = int(info["latent_dim"])
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def _normalize_frames(self, image_sequence: torch.Tensor) -> torch.Tensor:
        if image_sequence.dtype == torch.uint8:
            return image_sequence.to(torch.float32) / 255.0
        image_sequence = image_sequence.to(torch.float32)
        if image_sequence.numel() > 0 and float(image_sequence.max().detach().cpu()) > 1.5:
            image_sequence = image_sequence / 255.0
        return image_sequence.clamp(0.0, 1.0)

    def _patchify(self, x_btchw: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_btchw.shape
        patch = self.patch
        if H % patch != 0 or W % patch != 0:
            raise ValueError(f"Image shape {(H, W)} is not divisible by patch size {patch}.")
        x = x_btchw.reshape(B * T, C, H, W)
        x = x.unfold(2, patch, patch).unfold(3, patch, patch)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.reshape(B, T, -1, C * patch * patch)
        return x

    def forward(self, image_sequence: torch.Tensor) -> torch.Tensor:
        x = self._normalize_frames(image_sequence)
        x = self._patchify(x)
        with torch.no_grad():
            z_btld, _ = self.encoder(x)
        return z_btld.reshape(z_btld.shape[0], z_btld.shape[1], -1)


class BCPixelActorCritic(nn.Module):
    """
    PPO actor-critic that matches the BC policy structure:
    rendered image sequence -> tokenizer backbone -> BC classifier head.
    """

    def __init__(
        self,
        image_shape: tuple[int, ...],
        action_dim: int,
        tokenizer_ckpt: str,
        hidden_dim: int = 512,
        dropout: float = 0.05,
        backbone_device: torch.device | str = "cpu",
        temporal_layers: int = 2,
        temporal_heads: int = 4,
    ):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.backbone = TokenizerBackbone(tokenizer_ckpt=tokenizer_ckpt, device=backbone_device)
        self.classifier = BCActionClassifier(
            in_dim=self.backbone.feature_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            dropout=dropout,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            max_seq_len=self.image_shape[0] if len(self.image_shape) >= 1 else 64,
        )
        self.log_std = nn.Parameter(torch.full((1, action_dim), -1.0))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.backbone.feature_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _ensure_sequence(self, image_obs: torch.Tensor) -> torch.Tensor:
        if image_obs.ndim == 4:
            image_obs = image_obs.unsqueeze(1)
        if image_obs.ndim != 5:
            raise ValueError(
                f"Expected image observations with shape (B, T, H, W, C) or (B, H, W, C), got {tuple(image_obs.shape)}"
            )
        if image_obs.shape[-1] != 3:
            raise ValueError(f"Expected RGB images in the last dimension, got shape {tuple(image_obs.shape)}")
        return image_obs.permute(0, 1, 4, 2, 3).contiguous()

    def _last_features(self, image_obs: torch.Tensor) -> torch.Tensor:
        sequence = self._ensure_sequence(image_obs)
        features = self.backbone(sequence)
        return features[:, -1, :]

    def actor_mean(self, image_obs: torch.Tensor) -> torch.Tensor:
        sequence = self._ensure_sequence(image_obs)
        features = self.backbone(sequence)
        action_sequence = self.classifier(features)
        return action_sequence[:, -1, :]

    def action_dist(self, image_obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(image_obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(self, image_obs: torch.Tensor, action: torch.Tensor | None = None):
        distribution = self.action_dist(image_obs)
        if action is None:
            action = distribution.sample()
        value = self.get_value(image_obs)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_value(self, image_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._last_features(image_obs))


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
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        max_seq_len: int = 64,
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
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            max_seq_len=max_seq_len,
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
