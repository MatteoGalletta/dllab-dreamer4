from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .networks import BCPixelActorCritic, BCStyleLatentActorCritic, TokenizerLatentBCPPOActorCritic, VectorActorCritic


@dataclass
class PriorLoadInfo:
    loaded: bool
    message: str


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(payload, dict):
        if payload and all(isinstance(k, str) for k in payload.keys()):
            first_value = next(iter(payload.values()))
            if torch.is_tensor(first_value):
                return payload
        for key in ("state_dict", "model_state_dict", "network", "model", "actor"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                state_dict = _extract_state_dict(nested)
                if state_dict is not None:
                    return state_dict
    return None


def _normalize_prior_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "_orig_mod.", "network."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]

        if clean_key.startswith(("actor.", "critic.", "log_std")):
            normalized[clean_key] = value
            continue

        # Accept actor-only checkpoints saved from the Sequential directly.
        if clean_key[0].isdigit():
            normalized[f"actor.{clean_key}"] = value
            continue

        normalized[clean_key] = value
    return normalized


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _extract_checkpoint_args(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    args = payload.get("args", {}) or {}
    if isinstance(args, dict):
        return args
    try:
        return vars(args)
    except TypeError:
        return {}


def _load_bc_prior_payload(prior_path: Path, device: torch.device) -> Any:
    try:
        return torch.load(prior_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(prior_path, map_location=device)


def _infer_prior_format(state_dict: dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(key.startswith("classifier.") for key in keys) or any(key.startswith("backbone.") for key in keys):
        return "backbone_classifier"
    if any(key.startswith("net.") for key in keys):
        return "latent_mlp"
    if any(key.startswith("actor.") for key in keys):
        return "actor_critic"
    return "unknown"


def _infer_bc_latent_architecture(
    bc_prior_path: str | None,
    *,
    obs_shape: tuple[int, ...] | None,
    state_dim: int,
    action_dim: int,
    actor_hidden_dim: int,
) -> dict[str, int]:
    frame_stack = obs_shape[0] if obs_shape is not None and len(obs_shape) >= 1 else 1
    action_chunk_size = max(1, action_dim // 2)
    hidden_dim = int(actor_hidden_dim)
    source = "ppo_defaults"

    if bc_prior_path is not None:
        prior_path = Path(bc_prior_path)
        if prior_path.exists():
            try:
                payload = _load_bc_prior_payload(prior_path, torch.device("cpu"))
                checkpoint_args = _extract_checkpoint_args(payload)
            except Exception:
                checkpoint_args = {}
            if checkpoint_args.get("seq_len") is not None:
                frame_stack = int(checkpoint_args["seq_len"])
                source = "bc_prior_args"
            if checkpoint_args.get("action_chunk_size") is not None:
                action_chunk_size = int(checkpoint_args["action_chunk_size"])
                source = "bc_prior_args"
            if checkpoint_args.get("hidden_dim") is not None:
                hidden_dim = int(checkpoint_args["hidden_dim"])
                source = "bc_prior_args"

    if action_chunk_size < 1:
        raise ValueError("action_chunk_size must be at least 1")
    if action_dim != action_chunk_size * 2:
        raise ValueError(
            f"PushT bc_latent PPO expects flattened action_dim=2*chunk_size, got action_dim={action_dim} "
            f"and action_chunk_size={action_chunk_size}"
        )
    if obs_shape is not None and len(obs_shape) >= 1 and frame_stack != int(obs_shape[0]):
        raise ValueError(
            f"BC prior expects frame_stack={frame_stack}, but PPO env emits obs_stack={obs_shape[0]}. "
            "Match PPO obs_stack_size/chunk config to the BC prior before training."
        )

    return {
        "feature_dim": int(state_dim),
        "frame_stack": int(frame_stack),
        "action_dim": int(action_dim),
        "action_chunk_size": int(action_chunk_size),
        "hidden_dim": int(hidden_dim),
        "source": source,
    }


def _infer_bc_pixel_architecture(
    bc_prior_path: str | None,
    *,
    obs_shape: tuple[int, ...] | None,
    action_dim: int,
    actor_hidden_dim: int,
) -> dict[str, Any]:
    frame_stack = obs_shape[0] if obs_shape is not None and len(obs_shape) >= 1 else 1
    feature_dim = 256
    hidden_dim = int(actor_hidden_dim)
    policy_style = "sequence_classifier"
    backbone_style = "avgpool"
    source = "ppo_defaults"

    if bc_prior_path is not None:
        prior_path = Path(bc_prior_path)
        if prior_path.exists():
            try:
                payload = _load_bc_prior_payload(prior_path, torch.device("cpu"))
                checkpoint_args = _extract_checkpoint_args(payload)
                state_dict = _extract_state_dict(payload) or {}
                state_dict = _normalize_prior_keys(state_dict)
            except Exception:
                checkpoint_args = {}
                state_dict = {}

            if checkpoint_args.get("seq_len") is not None:
                frame_stack = int(checkpoint_args["seq_len"])
                source = "bc_prior_args"
            if checkpoint_args.get("hidden_dim") is not None:
                hidden_dim = int(checkpoint_args["hidden_dim"])
                source = "bc_prior_args"

            proj_weight = state_dict.get("backbone.proj.1.weight")
            if torch.is_tensor(proj_weight) and proj_weight.ndim == 2:
                feature_dim = int(proj_weight.shape[0])
                proj_in_dim = int(proj_weight.shape[1])
                if proj_in_dim == 512:
                    backbone_style = "spatial_softmax"
                    source = "bc_prior_state_dict"
                elif proj_in_dim == 256:
                    backbone_style = "avgpool"
                    source = "bc_prior_state_dict"

            classifier_first = state_dict.get("classifier.net.0.weight")
            if torch.is_tensor(classifier_first) and classifier_first.ndim == 2:
                hidden_dim = int(classifier_first.shape[0])
                classifier_in_dim = int(classifier_first.shape[1])
                if feature_dim > 0 and classifier_in_dim == feature_dim * frame_stack:
                    policy_style = "direct_chunk_cnn"
                    source = "bc_prior_state_dict"
                elif feature_dim > 0 and classifier_in_dim == feature_dim:
                    policy_style = "sequence_classifier"
                    source = "bc_prior_state_dict"

    if obs_shape is not None and len(obs_shape) >= 1 and frame_stack != int(obs_shape[0]):
        raise ValueError(
            f"BC prior expects frame_stack={frame_stack}, but PPO env emits obs_stack={obs_shape[0]}. "
            "Match PPO obs_stack_size/chunk config to the BC prior before training."
        )

    return {
        "frame_stack": int(frame_stack),
        "feature_dim": int(feature_dim),
        "hidden_dim": int(hidden_dim),
        "action_dim": int(action_dim),
        "policy_style": str(policy_style),
        "backbone_style": str(backbone_style),
        "source": source,
    }


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        clip_coef: float = 0.2,
        ent_coef: float = 0.003,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        actor_output_tanh: bool = True,
        bc_prior_path: str | None = None,
        prior_loss_coef: float = 0.0,
        prior_loss_decay: float = 1.0,
        bc_kl_penalty: bool = False,
        bc_kl_penalty_coef: float = 0.0,
        prior_log_std_init: float | None = None,
        device: torch.device | str = "cpu",
        network_type: str = "mlp",
        actor_hidden_dim: int = 512,
        actor_dropout: float = 0.05,
        obs_shape: tuple[int, ...] | None = None,
        tokenizer_path: str | None = None,
        backbone_device: torch.device | str = "cpu",
    ):
        self.device_override = torch.device(device)
        self.obs_shape = obs_shape
        self.tokenizer_path = tokenizer_path
        self.backbone_device = torch.device(backbone_device)
        self.bc_latent_architecture: dict[str, Any] | None = None
        self.bc_pixel_architecture: dict[str, Any] | None = None
        if network_type == "bc_pixels":
            if obs_shape is None:
                raise ValueError("obs_shape is required for bc_pixels PPO.")
            self.bc_pixel_architecture = _infer_bc_pixel_architecture(
                bc_prior_path,
                obs_shape=obs_shape,
                action_dim=action_dim,
                actor_hidden_dim=actor_hidden_dim,
            )
            self.network = BCPixelActorCritic(
                image_shape=obs_shape,
                action_dim=action_dim,
                hidden_dim=self.bc_pixel_architecture["hidden_dim"],
                dropout=actor_dropout,
                policy_style=self.bc_pixel_architecture["policy_style"],
                backbone_style=self.bc_pixel_architecture["backbone_style"],
                feature_dim=self.bc_pixel_architecture["feature_dim"],
            ).to(self.device_override)
        elif network_type == "bc_latent":
            self.bc_latent_architecture = _infer_bc_latent_architecture(
                bc_prior_path,
                obs_shape=obs_shape,
                state_dim=state_dim,
                action_dim=action_dim,
                actor_hidden_dim=actor_hidden_dim,
            )
            self.network = TokenizerLatentBCPPOActorCritic(
                feature_dim=self.bc_latent_architecture["feature_dim"],
                frame_stack=self.bc_latent_architecture["frame_stack"],
                action_dim=self.bc_latent_architecture["action_dim"],
                action_chunk_size=self.bc_latent_architecture["action_chunk_size"],
                hidden_dim=self.bc_latent_architecture["hidden_dim"],
                init_log_std=prior_log_std_init if prior_log_std_init is not None else -1.0,
            ).to(self.device_override)
        else:
            self.network = VectorActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                actor_output_tanh=actor_output_tanh,
            ).to(self.device_override)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            eps=1e-5,
        )

        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.base_prior_loss_coef = prior_loss_coef
        self.prior_loss_decay = prior_loss_decay
        self.bc_kl_penalty = bool(bc_kl_penalty)
        self.bc_kl_penalty_coef = float(bc_kl_penalty_coef)

        self._prior_network: nn.Module | None = None
        self.network_type = network_type
        self.prior_load_info = PriorLoadInfo(False, "No BC prior requested.")

        if prior_log_std_init is not None:
            with torch.no_grad():
                self.network.log_std.fill_(prior_log_std_init)

        if bc_prior_path is not None:
            self.prior_load_info = self.load_bc_prior(
                bc_prior_path=bc_prior_path,
                state_dim=state_dim,
                action_dim=action_dim,
                actor_output_tanh=actor_output_tanh,
                prior_log_std_init=prior_log_std_init,
            )

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    def _get_buffer_attr(self, buffer: Any, names: tuple[str, ...]):
        for name in names:
            if hasattr(buffer, name):
                return getattr(buffer, name)
        raise AttributeError("Buffer is missing attributes: " + ", ".join(names))

    def _to_tensor(self, x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype)
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def _flatten_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim >= 3:
            return states.reshape(-1, *states.shape[2:])
        if states.ndim == 2:
            return states
        raise ValueError(f"Unexpected state shape: {tuple(states.shape)}")

    def _flatten_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3:
            return actions.reshape(-1, actions.shape[-1])
        if actions.ndim == 2:
            return actions
        raise ValueError(f"Unexpected action shape: {tuple(actions.shape)}")

    def _flatten_logprobs(self, logprobs: torch.Tensor) -> torch.Tensor:
        if logprobs.ndim == 2:
            return logprobs.reshape(-1)
        if logprobs.ndim == 1:
            return logprobs
        if logprobs.ndim == 3:
            return logprobs.reshape(logprobs.shape[0] * logprobs.shape[1], -1).sum(dim=-1)
        raise ValueError(f"Unexpected logprob shape: {tuple(logprobs.shape)}")

    def current_prior_loss_coef(self, update_idx: int) -> float:
        return self.base_prior_loss_coef * (self.prior_loss_decay ** max(update_idx, 0))

    @torch.no_grad()
    def set_log_std(self, value: float) -> None:
        self.network.log_std.data.fill_(float(value))

    def load_bc_prior(
        self,
        bc_prior_path: str,
        state_dim: int,
        action_dim: int,
        actor_output_tanh: bool,
        prior_log_std_init: float | None,
    ) -> PriorLoadInfo:
        prior_path = Path(bc_prior_path)
        if not prior_path.exists():
            return PriorLoadInfo(False, f"BC prior not found at {prior_path}.")

        try:
            payload = _load_bc_prior_payload(prior_path, self.device)
        except Exception as error:
            return PriorLoadInfo(False, f"Failed to load BC prior: {error}")

        state_dict = _extract_state_dict(payload)
        if state_dict is None:
            return PriorLoadInfo(False, "BC prior format is unsupported.")

        checkpoint_args = _extract_checkpoint_args(payload)
        state_dict = _normalize_prior_keys(state_dict)
        prior_format = _infer_prior_format(state_dict)
        if self.network_type == "bc_pixels":
            load_target = {
                key: value
                for key, value in state_dict.items()
                if key.startswith("backbone.") or key.startswith("classifier.")
            }
            if not load_target:
                return PriorLoadInfo(False, "BC prior does not contain backbone/classifier weights for bc_pixels PPO.")
            try:
                incompatible = self.network.load_state_dict(load_target, strict=False)
            except RuntimeError as error:
                return PriorLoadInfo(
                    False,
                    f"BC prior at {prior_path} is incompatible with bc_pixels PPO ({prior_format}): {error}",
                )
        elif self.network_type == "bc_latent":
            policy_state = state_dict
            if any(key.startswith("classifier.") for key in state_dict):
                policy_state = _strip_prefix(state_dict, "classifier.")
            try:
                incompatible = self.network.bc_policy.load_state_dict(policy_state, strict=False)
            except RuntimeError as error:
                return PriorLoadInfo(
                    False,
                    "BC prior at "
                    f"{prior_path} is incompatible with tokenizer latent PPO "
                    f"({prior_format}). This usually means the checkpoint comes from the older "
                    "CNN/backbone BC pipeline rather than the newer latent-MLP policy. "
                    f"Original error: {error}"
                )
        else:
            actor_only = {k: v for k, v in state_dict.items() if k.startswith("actor.") or k == "log_std"}
            if not actor_only:
                actor_only = state_dict
            try:
                incompatible = self.network.load_state_dict(actor_only, strict=False)
            except RuntimeError as error:
                return PriorLoadInfo(
                    False,
                    f"BC prior at {prior_path} is incompatible with PPO actor loading ({prior_format}): {error}",
                )

        if self.network_type == "bc_pixels":
            prior_network = BCPixelActorCritic(
                image_shape=self.obs_shape or getattr(self.network, "image_shape"),
                action_dim=action_dim,
                hidden_dim=getattr(self.network, "hidden_dim", 512),
                dropout=0.0,
                policy_style=getattr(self.network, "policy_style", "sequence_classifier"),
                backbone_style=getattr(self.network, "backbone_style", "avgpool"),
                feature_dim=getattr(self.network, "feature_dim", 256),
            ).to(self.device)
        elif self.network_type == "bc_latent":
            prior_network = TokenizerLatentBCPPOActorCritic(
                feature_dim=int(self.network.feature_dim),
                frame_stack=int(self.network.frame_stack),
                action_dim=int(self.network.action_dim),
                action_chunk_size=int(self.network.action_chunk_size),
                hidden_dim=int(self.network.bc_policy.net[0].out_features),
                init_log_std=prior_log_std_init if prior_log_std_init is not None else -1.0,
            ).to(self.device)
        else:
            prior_network = VectorActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                actor_output_tanh=actor_output_tanh,
            ).to(self.device)
        prior_network.load_state_dict(self.network.state_dict(), strict=False)
        if prior_log_std_init is not None:
            with torch.no_grad():
                prior_network.log_std.fill_(prior_log_std_init)
        prior_network.eval()
        for parameter in prior_network.parameters():
            parameter.requires_grad_(False)
        self._prior_network = prior_network

        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        parts = [f"Loaded BC prior from {prior_path}."]
        if isinstance(checkpoint_args, dict):
            for key in ("seq_len", "action_chunk_size", "hidden_dim", "tokenizer_ckpt_name"):
                if key in checkpoint_args:
                    parts.append(f"{key}={checkpoint_args[key]}")
        if missing:
            parts.append(f"Missing keys: {missing[:6]}")
        if unexpected:
            parts.append(f"Unexpected keys: {unexpected[:6]}")
        if self.network_type == "bc_latent" and self.bc_latent_architecture is not None:
            parts.append(
                "bc_latent_architecture="
                f"(frame_stack={self.bc_latent_architecture['frame_stack']}, "
                f"chunk={self.bc_latent_architecture['action_chunk_size']}, "
                f"hidden={self.bc_latent_architecture['hidden_dim']}, "
                f"source={self.bc_latent_architecture['source']})"
            )
        if self.network_type == "bc_pixels" and self.bc_pixel_architecture is not None:
            parts.append(
                "bc_pixel_architecture="
                f"(frame_stack={self.bc_pixel_architecture['frame_stack']}, "
                f"feature_dim={self.bc_pixel_architecture['feature_dim']}, "
                f"hidden={self.bc_pixel_architecture['hidden_dim']}, "
                f"policy={self.bc_pixel_architecture['policy_style']}, "
                f"backbone={self.bc_pixel_architecture['backbone_style']}, "
                f"source={self.bc_pixel_architecture['source']})"
            )
        return PriorLoadInfo(True, " ".join(parts))

    def update(
        self,
        buffer: Any,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        batch_size: int,
        ppo_epochs: int,
        update_idx: int = 0,
        clip_vloss: bool = True,
    ) -> dict[str, float]:
        states = self._to_tensor(self._get_buffer_attr(buffer, ("states", "observations", "obs")))
        actions = self._to_tensor(self._get_buffer_attr(buffer, ("actions", "acts")))
        old_logprobs = self._to_tensor(self._get_buffer_attr(buffer, ("logprobs", "log_probs", "old_logprobs")))

        b_states = self._flatten_states(states)
        b_actions = self._flatten_actions(actions)
        b_old_logprobs = self._flatten_logprobs(old_logprobs)
        old_values = self._to_tensor(self._get_buffer_attr(buffer, ("values", "vals"))).reshape(-1)

        b_advantages = self._to_tensor(advantages).reshape(-1)
        b_returns = self._to_tensor(returns).reshape(-1)

        num_samples = b_states.shape[0]
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1e-8)
        indices = np.arange(num_samples)

        prior_loss_coef = self.current_prior_loss_coef(update_idx)

        pg_loss_value = 0.0
        v_loss_value = 0.0
        entropy_value = 0.0
        approx_kl_value = 0.0
        clipfrac_value = 0.0
        prior_loss_value = 0.0
        bc_kl_loss_value = 0.0
        num_minibatches = 0

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_inds = torch.as_tensor(indices[start:end], device=self.device, dtype=torch.long)

                _, new_logprobs, entropy, new_values = self.network.get_action_and_value(
                    b_states[mb_inds], b_actions[mb_inds]
                )
                new_values = new_values.squeeze(-1)

                logratio = new_logprobs - b_old_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]

                pg_loss_unclipped = -mb_advantages * ratio
                pg_loss_clipped = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()

                if clip_vloss:
                    value_target = b_returns[mb_inds]
                    old_value_batch = old_values[mb_inds]
                    v_loss_unclipped = (new_values - value_target).pow(2)
                    v_clipped = old_value_batch + torch.clamp(
                        new_values - old_value_batch, -self.clip_coef, self.clip_coef
                    )
                    v_loss_clipped = (v_clipped - value_target).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_inds]).pow(2)).mean()
                entropy_loss = entropy.mean()

                prior_loss = torch.zeros((), device=self.device)
                if self._prior_network is not None and prior_loss_coef > 0.0:
                    with torch.no_grad():
                        prior_mean = self._prior_network.actor_mean(b_states[mb_inds])
                    policy_mean = self.network.actor_mean(b_states[mb_inds])
                    prior_loss = torch.mean((policy_mean - prior_mean).pow(2))

                bc_kl_loss = torch.zeros((), device=self.device)
                if self._prior_network is not None and self.bc_kl_penalty and self.bc_kl_penalty_coef > 0.0:
                    with torch.no_grad():
                        bc_action_mean = self._prior_network.actor_mean(b_states[mb_inds])
                    current_action_mean = self.network.actor_mean(b_states[mb_inds])
                    bc_kl_loss = torch.mean((current_action_mean - bc_action_mean).pow(2))

                loss = (
                    pg_loss
                    + self.vf_coef * v_loss
                    - self.ent_coef * entropy_loss
                    + prior_loss_coef * prior_loss
                    + self.bc_kl_penalty_coef * bc_kl_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

                pg_loss_value += float(pg_loss.detach().cpu())
                v_loss_value += float(v_loss.detach().cpu())
                entropy_value += float(entropy_loss.detach().cpu())
                approx_kl_value += float(approx_kl.detach().cpu())
                clipfrac_value += float(clipfrac.detach().cpu())
                prior_loss_value += float(prior_loss.detach().cpu())
                bc_kl_loss_value += float(bc_kl_loss.detach().cpu())
                num_minibatches += 1

            if self.target_kl is not None:
                mean_kl = approx_kl_value / max(1, num_minibatches)
                if mean_kl > self.target_kl:
                    break

        denom = max(1, num_minibatches)
        return {
            "policy_loss": pg_loss_value / denom,
            "value_loss": v_loss_value / denom,
            "entropy": entropy_value / denom,
            "approx_kl": approx_kl_value / denom,
            "clipfrac": clipfrac_value / denom,
            "prior_loss": prior_loss_value / denom,
            "bc_kl_loss": bc_kl_loss_value / denom,
            "prior_loss_coef": prior_loss_coef,
            "bc_kl_penalty_coef": self.bc_kl_penalty_coef,
        }
