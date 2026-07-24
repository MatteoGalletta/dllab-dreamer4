from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from behavioural_cloning.train_tokenizer_latent_bc import TokenizerLatentBCPolicy
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
        temporal_context: int = 3,
        output_tanh: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.max_seq_len = int(max_seq_len)
        self.temporal_context = max(1, int(temporal_context))
        self.output_tanh = bool(output_tanh)
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

    def _causal_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        query_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        key_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        future_mask = key_idx > query_idx
        history_mask = key_idx < (query_idx - (self.temporal_context - 1))
        return future_mask | history_mask

    def forward(self, features_btD: torch.Tensor) -> torch.Tensor:
        if features_btD.ndim == 2:
            features_btD = features_btD.unsqueeze(1)
        if features_btD.ndim != 3:
            raise ValueError(f"Expected latent sequence with shape (B, T, D), got {tuple(features_btD.shape)}")
        B, T, D = features_btD.shape
        base_logits = self.net(features_btD.reshape(B * T, D)).view(B, T, -1)
        with torch.autocast(device_type=features_btD.device.type, enabled=False):
            temporal_features = self.temporal_in(features_btD.float())
            temporal_pos = self._positional_encoding(T).to(
                device=temporal_features.device,
                dtype=torch.float32,
            )
            temporal_features = temporal_features + temporal_pos
            attn_mask = self._causal_local_mask(T, temporal_features.device)
            for block in self.temporal_blocks:
                temporal_features = block(temporal_features, src_mask=attn_mask)
            temporal_logits = self.temporal_out(self.temporal_norm(temporal_features))
        temporal_logits = temporal_logits.to(base_logits.dtype)
        logits = base_logits + temporal_logits
        actions = torch.tanh(logits) if self.output_tanh else logits
        return actions.view(B, T, -1)


class DirectChunkPolicyHead(nn.Module):
    def __init__(self, *, in_dim: int, seq_len: int, hidden_dim: int, action_dim: int, dropout: float, output_tanh: bool = True):
        super().__init__()
        self.seq_len = int(seq_len)
        self.action_dim = int(action_dim)
        self.output_tanh = bool(output_tanh)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim) * self.seq_len, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.action_dim),
        )

    def forward(self, features_btD: torch.Tensor) -> torch.Tensor:
        if features_btD.ndim != 3:
            raise ValueError(f"Expected feature sequence with shape (B, T, D), got {tuple(features_btD.shape)}")
        batch, steps, feature_dim = features_btD.shape
        if steps != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {steps}")
        logits = self.net(features_btD.reshape(batch, steps * feature_dim))
        return torch.tanh(logits) if self.output_tanh else logits


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        x_flat = x.view(bsz, channels, height * width) / self.temperature
        weights = F.softmax(x_flat, dim=-1).view(bsz, channels, height, width)

        y_coord = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        x_coord = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        y_grid, x_grid = torch.meshgrid(y_coord, x_coord, indexing="ij")

        expected_y = torch.sum(weights * y_grid, dim=(2, 3))
        expected_x = torch.sum(weights * x_grid, dim=(2, 3))
        return torch.cat([expected_x, expected_y], dim=-1)


class PixelBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 256, backbone_style: str = "avgpool"):
        super().__init__()
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)
        self.backbone_style = str(backbone_style)

        def conv_block(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
            kernel = 5 if stride == 2 else 3
            padding = 2 if kernel == 5 else 1
            groups = min(8, out_ch)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.ReLU(),
            )

        if self.backbone_style == "spatial_softmax":
            self.backbone = nn.Sequential(
                conv_block(self.in_channels, 32, stride=2),
                conv_block(32, 64, stride=2),
                conv_block(64, 128, stride=2),
                conv_block(128, 256, stride=2),
                conv_block(256, 256, stride=2),
                SpatialSoftmax(),
            )
            proj_in_dim = 256 * 2
        else:
            self.backbone = nn.Sequential(
                conv_block(self.in_channels, 32, stride=2),
                conv_block(32, 64, stride=2),
                conv_block(64, 128, stride=2),
                conv_block(128, 256, stride=2),
                conv_block(256, 256, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            proj_in_dim = 256
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(proj_in_dim, self.feature_dim),
            nn.ReLU(),
        )

    def _normalize_frames(self, image_sequence: torch.Tensor) -> torch.Tensor:
        if image_sequence.dtype == torch.uint8:
            return image_sequence.to(torch.float32) / 255.0
        image_sequence = image_sequence.to(torch.float32)
        if image_sequence.numel() > 0 and float(image_sequence.max().detach().cpu()) > 1.5:
            image_sequence = image_sequence / 255.0
        return image_sequence.clamp(0.0, 1.0)

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        x_btchw = self._normalize_frames(x_btchw)
        bsz, seq_len, channels, height, width = x_btchw.shape
        x = x_btchw.reshape(bsz * seq_len, channels, height, width)
        features = self.proj(self.backbone(x))
        return features.view(bsz, seq_len, -1)


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
    rendered image sequence -> CNN backbone -> BC classifier head.
    """

    def __init__(
        self,
        image_shape: tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.05,
        policy_style: str = "sequence_classifier",
        backbone_style: str = "avgpool",
        feature_dim: int = 256,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_context: int = 3,
    ):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.policy_style = str(policy_style)
        self.backbone_style = str(backbone_style)
        self.feature_dim = int(feature_dim)
        self.backbone = PixelBackbone(
            in_channels=3,
            feature_dim=self.feature_dim,
            backbone_style=self.backbone_style,
        )
        if self.policy_style == "direct_chunk_cnn":
            self.classifier = DirectChunkPolicyHead(
                in_dim=self.backbone.feature_dim,
                seq_len=self.image_shape[0] if len(self.image_shape) >= 1 else 1,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                dropout=dropout,
            )
        else:
            self.classifier = BCActionClassifier(
                in_dim=self.backbone.feature_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                dropout=dropout,
                temporal_layers=temporal_layers,
                temporal_heads=temporal_heads,
                max_seq_len=self.image_shape[0] if len(self.image_shape) >= 1 else 64,
                temporal_context=temporal_context,
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
        action_output = self.classifier(features)
        if action_output.ndim == 2:
            return action_output
        return action_output[:, -1, :]

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
        policy_style: str = "sequence_classifier",
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        max_seq_len: int = 64,
        temporal_context: int = 3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.policy_style = str(policy_style)
        if self.policy_style == "direct_chunk_cnn":
            self.classifier = DirectChunkPolicyHead(
                in_dim=feature_dim,
                seq_len=max_seq_len,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                dropout=dropout,
            )
        else:
            self.classifier = BCActionClassifier(
                in_dim=feature_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                dropout=dropout,
                temporal_layers=temporal_layers,
                temporal_heads=temporal_heads,
                max_seq_len=max_seq_len,
                temporal_context=temporal_context,
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
        action_output = self.classifier(sequence)
        if action_output.ndim == 2:
            return action_output
        return action_output[:, -1, :]

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


class TokenizerLatentBCPPOActorCritic(nn.Module):
    """
    Latent PPO actor-critic that stays close to the other group's setup:
    the actor mean is exactly the tokenizer latent BC policy, while PPO learns
    a state-independent Gaussian log-std and a separate critic over the full
    stacked latent history.
    """

    def __init__(
        self,
        feature_dim: int,
        frame_stack: int,
        action_dim: int,
        action_chunk_size: int,
        hidden_dim: int = 512,
        init_log_std: float = -1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.frame_stack = int(frame_stack)
        self.action_dim = int(action_dim)
        self.action_chunk_size = int(action_chunk_size)
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1")
        if self.action_dim % self.action_chunk_size != 0:
            raise ValueError(
                f"Flat action_dim={self.action_dim} must be divisible by action_chunk_size={self.action_chunk_size}"
            )
        self.primitive_action_dim = int(self.action_dim // self.action_chunk_size)

        self.bc_policy = TokenizerLatentBCPolicy(
            latent_dim=self.feature_dim,
            frame_stack=self.frame_stack,
            action_dim=self.primitive_action_dim,
            hidden_dim=int(hidden_dim),
            action_chunk_size=self.action_chunk_size,
        )
        self.log_std = nn.Parameter(torch.full((1, self.action_dim), float(init_log_std)))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim * self.frame_stack, int(hidden_dim))),
            nn.ReLU(),
            layer_init(nn.Linear(int(hidden_dim), int(hidden_dim))),
            nn.ReLU(),
            layer_init(nn.Linear(int(hidden_dim), 1), std=1.0),
        )

    def _ensure_sequence(self, state_vector: torch.Tensor) -> torch.Tensor:
        if state_vector.ndim == 2:
            state_vector = state_vector.unsqueeze(1)
        if state_vector.ndim != 3:
            raise ValueError(f"Expected state shape (B, D) or (B, T, D), got {tuple(state_vector.shape)}")
        if state_vector.shape[1] != self.frame_stack:
            raise ValueError(f"Expected frame_stack={self.frame_stack}, got {state_vector.shape[1]}")
        if state_vector.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {state_vector.shape[2]}")
        return state_vector

    def actor_mean(self, state_vector: torch.Tensor) -> torch.Tensor:
        sequence = self._ensure_sequence(state_vector)
        return self.bc_policy(sequence)

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
        sequence = self._ensure_sequence(state_vector)
        flat = sequence.reshape(sequence.shape[0], -1)
        return self.critic(flat)
