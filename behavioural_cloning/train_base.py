# train_base.py
import os
import time
import random
import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_ROOT = PROJECT_ROOT / "logs"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.PushTDataLoader import PushTSequenceDataset
from ppo_online.env_config import DEFAULT_PUSHT_ENV_ID, make_pusht_env
from ppo_online.model_paths import resolve_tokenizer_path
from ppo_online.tokenizer_utils import load_tokenizer_from_ckpt

DREAMER4_MODEL_PATH = PROJECT_ROOT / "dreamer4-src" / "dreamer4" / "model.py"


def load_dreamer4_model_module():
    spec = importlib.util.spec_from_file_location("dreamer4_model_bc", DREAMER4_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Dreamer4 model module from {DREAMER4_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dreamer4_model = load_dreamer4_model_module()
Dreamer4Encoder = dreamer4_model.Encoder
temporal_patchify = dreamer4_model.temporal_patchify

if importlib.util.find_spec("wandb") is not None:
    wandb = importlib.import_module("wandb")
else:
    class _WandbStub:
        class Image:
            def __init__(self, data, caption=None):
                self.data = data
                self.caption = caption

        class Video:
            def __init__(self, data_or_path, caption=None, fps=None, format=None):
                self.data_or_path = data_or_path
                self.caption = caption
                self.fps = fps
                self.format = format

        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    wandb = _WandbStub()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def get_wandb_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "wandb_mode", "disabled") or "disabled"
    # if mode == "online" and not os.environ.get("WANDB_API_KEY") and not (Path.home() / ".netrc").exists():
        # return "disabled"
    return mode

def get_runtime_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "xpu"
    return torch.device("cpu"), "cpu"

def seed_everything(seed: int):
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.set_device(local_rank)
    return ddp, rank, world_size, local_rank


class CNNBackbone(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)

        def conv_block(in_ch: int, out_ch: int, *, stride: int = 2) -> nn.Sequential:
            kernel = 5 if stride == 2 else 3
            padding = 2 if kernel == 5 else 1
            groups = min(8, out_ch)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.ReLU(),
            )

        self.backbone = nn.Sequential(
            conv_block(self.in_channels, 32, stride=2),
            conv_block(32, 64, stride=2),
            conv_block(64, 128, stride=2),
            conv_block(128, 256, stride=2),
            conv_block(256, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_btchw.shape
        x = x_btchw.reshape(B * T, C, H, W)
        features = self.proj(self.backbone(x))
        return features.view(B, T, -1)


class TokenizerBackbone(nn.Module):
    def __init__(self, encoder: nn.Module, patch: int):
        super().__init__()
        self.encoder = encoder
        self.patch = int(patch)
        latent_dim = int(self.encoder.n_latents) * int(self.encoder.bottleneck_proj.out_features)
        self.feature_dim = latent_dim

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        patches = temporal_patchify(x_btchw, self.patch)
        with torch.no_grad():
            z, _ = self.encoder(patches)
        return z.reshape(z.shape[0], z.shape[1], -1)


class ActionClassifier(nn.Module):
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
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.max_seq_len = int(max_seq_len)
        self.temporal_context = max(1, int(temporal_context))
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
        B, T, D = features_btD.shape
        base_logits = self.net(features_btD.reshape(B * T, D)).view(B, T, -1)
        temporal_features = self.temporal_in(features_btD) + self._positional_encoding(T)
        attn_mask = self._causal_local_mask(T, temporal_features.device)
        for block in self.temporal_blocks:
            temporal_features = block(temporal_features, src_mask=attn_mask)
        temporal_logits = self.temporal_out(self.temporal_norm(temporal_features))
        logits = base_logits + temporal_logits
        actions = torch.tanh(logits)
        return actions.view(B, T, -1)


class Policy(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: ActionClassifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x_btchw)
        return self.classifier(features)

    def train(self, mode: bool = True):
        super().train(mode)
        if isinstance(self.backbone, TokenizerBackbone):
            self.backbone.eval()
            self.backbone.requires_grad_(False)
        return self


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    cleaned = state_dict
    for prefix in prefixes:
        cleaned = _strip_prefix(cleaned, prefix)
    return cleaned


def load_tokenizer_encoder(tokenizer_ckpt_name: str) -> nn.Module:
    ckpt_path = Path(tokenizer_ckpt_name)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        log_candidates = [
            LOCAL_MODEL_ROOT / "tokenizer_ckpts" / tokenizer_ckpt_name,
            LOCAL_MODEL_ROOT / "tokenizer_ckpts" / "tokenizer.pt",
            LOCAL_MODEL_ROOT / "tokenizer_ckpts" / "latest.pt",
        ]
        ckpt_path = next((path for path in log_candidates if path.exists()), Path(resolve_tokenizer_path(tokenizer_ckpt_name)))

    tokenizer, info = load_tokenizer_from_ckpt(str(ckpt_path), torch.device("cpu"))
    encoder = tokenizer.encoder
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.patch = int(info["patch"])
    return encoder


def save_ckpt(path: Path, *, step: int, epoch: int, model, opt, scaler, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, *, model, opt, scaler) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"]
    (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def get_base_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def normalize_image_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        return x.to(torch.float32) / 255.0
    x = x.to(torch.float32)
    if x.numel() > 0 and float(x.max().detach().cpu()) > 1.5:
        x = x / 255.0
    return x


def split_indices(num_items: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if num_items <= 1 or val_frac <= 0.0:
        return list(range(num_items)), []
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(num_items).tolist()
    val_size = min(num_items - 1, max(1, int(round(num_items * float(val_frac)))))
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    return train_indices, val_indices


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader | None,
    *,
    device: torch.device,
    device_type: str,
    action_dim: int,
    use_amp: bool,
    max_batches: int,
) -> dict[str, float]:
    if loader is None:
        return {}

    base_model = get_base_model(model)
    was_training = base_model.training
    base_model.eval()

    loss_sum = 0.0
    mae_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            x = normalize_image_batch(batch["image"].to(device, non_blocking=True))
            target_actions = batch["action"].to(device, non_blocking=True).to(torch.float32)
            if target_actions.ndim != 3 or target_actions.shape[-1] != action_dim:
                raise RuntimeError(f"Expected actions shape (B,T,{action_dim}), got {tuple(target_actions.shape)}")

            with autocast(device_type=device_type, enabled=use_amp):
                pred = base_model(x)
                loss = F.mse_loss(pred, target_actions)
                mae = torch.mean(torch.abs(pred - target_actions))

            loss_sum += float(loss.item())
            mae_sum += float(mae.item())
            num_batches += 1

    if was_training:
        base_model.train()

    if num_batches == 0:
        return {}

    return {
        "validation/loss": loss_sum / num_batches,
        "validation/mae": mae_sum / num_batches,
        "validation/num_batches": float(num_batches),
    }


def _success_from_eval_info(info: dict, terminated: bool) -> float:
    for key in ("block_success", "success", "is_success", "task_success"):
        if key in info:
            try:
                return float(np.asarray(info[key]).squeeze()) > 0.5
            except Exception:
                pass
    return float(bool(terminated))


def _map_primitive_to_env_action(
    primitive: np.ndarray,
    *,
    mode: str,
    current_eef: np.ndarray,
    max_step_pixels: float,
) -> np.ndarray:
    primitive = np.asarray(primitive, dtype=np.float32)
    if mode == "relative":
        return primitive
    if mode == "delta":
        return np.clip(current_eef + primitive * float(max_step_pixels), 0.0, 512.0)
    if mode == "absolute":
        return np.clip((primitive + 1.0) * 256.0, 0.0, 512.0)
    raise ValueError(f"Unsupported eval action mode: {mode}")


def _pad_history(frames: list[np.ndarray], seq_len: int) -> np.ndarray:
    if not frames:
        raise ValueError("frame history is empty")
    history = list(frames[-seq_len:])
    if len(history) < seq_len:
        history = [history[0]] * (seq_len - len(history)) + history
    return np.stack(history, axis=0)


def _ensemble_primitive(
    pending_chunks: list[dict[str, object]],
    *,
    chunk_size: int,
    decay: float,
) -> np.ndarray:
    candidates = []
    weights = []
    for age, entry in enumerate(reversed(pending_chunks)):
        offset = int(entry["offset"])
        if offset >= chunk_size:
            continue
        candidates.append(np.asarray(entry["chunk"], dtype=np.float32)[offset])
        weights.append(np.exp(-float(decay) * age))
    if not candidates:
        return np.zeros(2, dtype=np.float32)
    weights_np = np.asarray(weights, dtype=np.float32)
    weights_np /= max(weights_np.sum(), 1e-8)
    return np.sum(np.asarray(candidates, dtype=np.float32) * weights_np[:, None], axis=0)


def evaluate_rollouts(
    model: nn.Module,
    *,
    device: torch.device,
    image_hw: tuple[int, int],
    seq_len: int,
    chunk_size: int,
    episodes: int,
    max_steps: int,
    seed: int,
    action_mode: str,
    max_step_pixels: float,
    temporal_ensemble: bool,
    temporal_ensemble_decay: float,
    video_fps: int,
) -> dict[str, object]:
    base_model = get_base_model(model)
    was_training = base_model.training
    base_model.eval()

    env = make_pusht_env(
        env_id=DEFAULT_PUSHT_ENV_ID,
        render_mode="rgb_array",
        image_height=int(image_hw[0]),
        image_width=int(image_hw[1]),
        relative=(action_mode == "relative"),
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=True,
        render_obs=False,
    )

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    coverages: list[float] = []
    video_frames: list[np.ndarray] = []

    with torch.no_grad():
        for episode_idx in range(int(episodes)):
            env.reset(seed=int(seed) + episode_idx)
            frame_history: list[np.ndarray] = []
            action_buffer: list[np.ndarray] = []
            pending_chunks: list[dict[str, object]] = []
            total_reward = 0.0
            step_count = 0
            done = False
            final_info: dict = {}
            current_eef = np.asarray(env.unwrapped._get_obs(), dtype=np.float32)[0:2].copy()

            while not done and step_count < int(max_steps):
                frame = np.asarray(env.render(), dtype=np.uint8)
                frame_history.append(frame)
                if episode_idx == 0:
                    video_frames.append(frame.copy())

                if temporal_ensemble or not action_buffer:
                    stacked_frames = _pad_history(frame_history, seq_len)
                    input_tensor = torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=device)
                    action_seq = base_model(input_tensor.permute(0, 1, 4, 2, 3).to(torch.float32) / 255.0)
                    action_chunk = action_seq[:, -1, :].view(1, chunk_size, 2).squeeze(0).cpu().numpy()
                    action_chunk = np.clip(action_chunk, -1.0, 1.0)
                    if temporal_ensemble:
                        pending_chunks.append({"chunk": action_chunk, "offset": 0})
                    else:
                        action_buffer.extend([np.asarray(a, dtype=np.float32) for a in action_chunk])

                if temporal_ensemble:
                    primitive = _ensemble_primitive(
                        pending_chunks,
                        chunk_size=chunk_size,
                        decay=temporal_ensemble_decay,
                    )
                else:
                    primitive = np.asarray(action_buffer.pop(0), dtype=np.float32)

                env_action = _map_primitive_to_env_action(
                    primitive,
                    mode=action_mode,
                    current_eef=current_eef,
                    max_step_pixels=max_step_pixels,
                )
                obs, reward, terminated, truncated, info = env.step(env_action)
                current_eef = np.asarray(obs["state"] if isinstance(obs, dict) and "state" in obs else obs, dtype=np.float32)[
                    0:2
                ].copy()
                total_reward += float(reward)
                step_count += 1
                final_info = dict(info)
                done = bool(terminated or truncated)

                if temporal_ensemble:
                    for entry in pending_chunks:
                        entry["offset"] = int(entry["offset"]) + 1
                    pending_chunks = [entry for entry in pending_chunks if int(entry["offset"]) < chunk_size]

            returns.append(total_reward)
            lengths.append(step_count)
            successes.append(float(_success_from_eval_info(final_info, done)))
            if "coverage" in final_info:
                coverages.append(float(final_info["coverage"]))

    env.close()
    if was_training:
        base_model.train()

    metrics: dict[str, object] = {
        "rollout/success_rate": float(np.mean(successes)) if successes else 0.0,
        "rollout/mean_return": float(np.mean(returns)) if returns else 0.0,
        "rollout/mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    if coverages:
        metrics["rollout/mean_coverage"] = float(np.mean(coverages))
    if video_frames:
        video_array = np.stack(video_frames, axis=0).transpose(0, 3, 1, 2)
        metrics["rollout/video"] = wandb.Video(video_array, fps=int(video_fps), format="mp4")
    return metrics


def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device, device_type = get_runtime_device()
    if device.type in {"cuda", "xpu"}:
        device = torch.device(f"{device.type}:{local_rank}") if ddp else device

    seed_everything(args.seed + rank)

    # ---- data ----
    full_dataset = PushTSequenceDataset(
        h5_path=args.dataset,
        seq_len=args.seq_len,
        action_chunk_size=args.action_chunk_size,
    )
    train_indices, val_indices = split_indices(len(full_dataset), args.val_frac, args.seed)
    dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices) if val_indices else None

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )
    val_loader = None
    if is_rank0() and val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=worker_init_fn,
        )

    # ---- model ----
    action_dim = args.action_chunk_size * 2
    if args.tokenizer_ckpt_name and is_rank0():
        print("Ignoring --tokenizer_ckpt_name. This BC run uses raw image pixels only.")
    args.tokenizer_ckpt_name = None
    backbone = CNNBackbone(in_channels=args.C)
    backbone_dim = backbone.feature_dim

    classifier = ActionClassifier(
        in_dim=backbone_dim,
        hidden_dim=args.hidden_dim,
        action_dim=action_dim,
        dropout=args.dropout,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        max_seq_len=args.seq_len,
        temporal_context=args.temporal_context,
    )
    model = Policy(backbone=backbone, classifier=classifier).to(device)

    if is_rank0():
        print(f"Dataset split: train={len(dataset)} val={0 if val_dataset is None else len(val_dataset)}")
        print(model)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Learnable parameters: {param_count:,}")

    if args.compile:
        model = torch.compile(model)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    # ---- optim ----
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device_type in {"cuda", "xpu"}
    scaler = GradScaler(device=device_type, enabled=use_amp)

    # ---- wandb ----
    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode=get_wandb_mode(args),
            config=vars(args),
        )

    # ---- resume ----
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    best_rollout_success = float("-inf")
    best_val_loss = float("inf")
    if args.resume is not None:
        step, start_epoch = load_ckpt(Path(args.resume), model=model, opt=opt, scaler=scaler)
        if is_rank0():
            print(f"[rank0] Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # ---- train ----
    model.train()
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))

    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in loader:
                if step >= args.max_steps:
                    break

                x = normalize_image_batch(batch["image"].to(device, non_blocking=True))  # (B,T,C,H,W)

                with autocast(device_type=device_type, enabled=use_amp):
                    pred = model(x)

                target_actions = batch["action"].to(device, non_blocking=True).to(torch.float32)
                if target_actions.ndim != 3 or target_actions.shape[-1] != action_dim:
                    raise RuntimeError(f"Expected actions shape (B,T,{action_dim}), got {tuple(target_actions.shape)}")

                loss = F.mse_loss(pred, target_actions)
                mae = torch.mean(torch.abs(pred - target_actions))

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: loss={loss}")

                loss_to_backprop = loss / grad_accum

                scaler.scale(loss_to_backprop).backward()

                do_step = ((step + 1) % grad_accum == 0)
                if do_step:
                    if use_amp:
                        scaler.step(opt)

                        if is_rank0() and step % args.log_every == 0:
                            wandb.log({"amp/scale": float(scaler.get_scale())}, step=step)

                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                # ---- logging ----
                if is_rank0() and (step % args.log_every == 0):
                    wandb.log(
                        {
                            "loss/total": float(loss.item()),
                            "loss/action_mse": float(loss.item()),
                            "loss/mae": float(mae.item()),
                            "stats/pred_mean": float(pred.mean().item()),
                            "stats/target_mean": float(target_actions.mean().item()),
                            "lr": float(opt.param_groups[0]["lr"]),
                            "time/hrs": (time.time() - t0) / 3600.0,
                        },
                        step=step,
                    )

                if is_rank0() and (step % args.print_every == 0):
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| mae={mae.item():.6f} | pred_mean={pred.mean().item():.4f}"
                    )

                if (
                    is_rank0()
                    and args.eval_every > 0
                    and step > 0
                    and (step % args.eval_every == 0)
                    and do_step
                ):
                    eval_log_dict: dict[str, object] = {}
                    eval_log_dict.update(
                        evaluate_validation(
                            model,
                            val_loader,
                            device=device,
                            device_type=device_type,
                            action_dim=action_dim,
                            use_amp=use_amp,
                            max_batches=args.val_max_batches,
                        )
                    )
                    eval_log_dict.update(
                        evaluate_rollouts(
                            model,
                            device=device,
                            image_hw=(args.H, args.W),
                            seq_len=args.seq_len,
                            chunk_size=args.action_chunk_size,
                            episodes=args.eval_episodes,
                            max_steps=args.eval_max_steps,
                            seed=args.seed + step,
                            action_mode=args.eval_action_mode,
                            max_step_pixels=args.eval_max_step_pixels,
                            temporal_ensemble=args.eval_temporal_ensemble,
                            temporal_ensemble_decay=args.eval_temporal_ensemble_decay,
                            video_fps=args.eval_video_fps,
                        )
                    )
                    if eval_log_dict:
                        wandb.log(eval_log_dict, step=step)
                        val_loss = eval_log_dict.get("validation/loss")
                        success_rate = eval_log_dict.get("rollout/success_rate")
                        val_text = "n/a" if val_loss is None else f"{float(val_loss):.6f}"
                        success_text = "n/a" if success_rate is None else f"{float(success_rate):.3f}"
                        print(f"eval step {step:07d} | val_loss={val_text} | rollout_success={success_text}")
                        if success_rate is not None:
                            success_value = float(success_rate)
                            val_value = float(val_loss) if val_loss is not None else float("inf")
                            improved = success_value > best_rollout_success
                            tied_better_val = success_value == best_rollout_success and val_value < best_val_loss
                            if improved or tied_better_val:
                                best_rollout_success = success_value
                                best_val_loss = val_value
                                best_path = ckpt_dir / "best_rollout.pt"
                                save_ckpt(
                                    best_path,
                                    step=step,
                                    epoch=epoch,
                                    model=model,
                                    opt=opt,
                                    scaler=scaler,
                                    args=args,
                                )
                                wandb.log(
                                    {
                                        "best_rollout/success_rate": best_rollout_success,
                                        "best_rollout/validation_loss": best_val_loss,
                                        "best_rollout/step": float(step),
                                    },
                                    step=step,
                                )
                                print(
                                    f"saved best rollout checkpoint at step {step:07d} "
                                    f"| success={best_rollout_success:.3f} | val_loss={best_val_loss:.6f}"
                                )

                # ---- ckpt ----
                if is_rank0() and args.save_every > 0 and (step % args.save_every == 0) and do_step:
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(ckpt_path, step=step, epoch=epoch, model=model, opt=opt, scaler=scaler, args=args)
                    # also update a "latest" pointer
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(latest, step=step, epoch=epoch, model=model, opt=opt, scaler=scaler, args=args)

                step += 1

            start_epoch = epoch + 1

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data
    p.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
    )
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--action_chunk_size", type=int, default=5)

    # image
    p.add_argument("--H", type=int, default=224)
    p.add_argument("--W", type=int, default=224)
    p.add_argument("--C", type=int, default=3)

    # model
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--temporal_layers", type=int, default=2)
    p.add_argument("--temporal_heads", type=int, default=4)
    p.add_argument("--temporal_context", type=int, default=3)
    p.add_argument(
        "--tokenizer_ckpt_name",
        type=str,
        default=None,
        help="ignored for this setup; BC now trains on raw image pixels only",
    )

    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10_000_000)
    p.add_argument("--grad_accum", type=int, default=1)

    # lpips
    p.add_argument("--lpips_weight", type=float, default=0.2)
    p.add_argument("--lpips_frac", type=float, default=0.5)
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    # logging / viz
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--viz_every", type=int, default=1000)
    p.add_argument("--viz_max_items", type=int, default=4)
    p.add_argument("--viz_max_T", type=int, default=8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--val_max_batches", type=int, default=0, help="0 evaluates the full validation loader")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_batch_size", type=int, default=0)
    p.add_argument("--eval_episodes", type=int, default=3)
    p.add_argument("--eval_max_steps", type=int, default=300)
    p.add_argument("--eval_action_mode", type=str, default="relative", choices=["relative", "delta", "absolute"])
    p.add_argument("--eval_max_step_pixels", type=float, default=15.0)
    p.add_argument("--eval_temporal_ensemble", action="store_true")
    p.add_argument("--eval_temporal_ensemble_decay", type=float, default=0.35)
    p.add_argument("--eval_video_fps", type=int, default=15)

    # wandb
    p.add_argument("--wandb_project", type=str, default="pusht-behavior-cloning")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"], help="wandb logging mode")

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./local_models/behavior_cloning")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    train(p.parse_args())
# torchrun --nproc_per_node=8 train_base.py --dataset /data2/ws1/lagandua-MySpace/pusht_expert_train.h5 --wandb_mode online
