from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODELS_ROOT = PROJECT_ROOT / "local_models"
LOGS_ROOT = PROJECT_ROOT / "logs"
BC_MODEL_DIR = LOCAL_MODELS_ROOT / "behavior_cloning"
TOKENIZER_MODEL_DIR = LOGS_ROOT / "tokenizer_ckpts"
TOKENIZER_FALLBACK_DIR = LOCAL_MODELS_ROOT / "tokenizer"
PPO_MODEL_DIR = LOCAL_MODELS_ROOT / "ppo_online"
DYNAMICS_MODEL_DIR = LOGS_ROOT / "dynamics_ckpts"


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_explicit_or_directory_candidate(path: str | None, directory: Path) -> str | None:
    if not path:
        return None
    explicit = Path(path)
    if explicit.is_absolute() or explicit.exists():
        return str(explicit)
    directory_candidate = directory / explicit
    if directory_candidate.exists():
        return str(directory_candidate)
    return None


def resolve_bc_prior_path(path: str | None = None) -> str:
    resolved = _resolve_explicit_or_directory_candidate(path, BC_MODEL_DIR)
    if resolved is not None:
        return resolved

    candidates = [
        BC_MODEL_DIR / "latest.pt",
        BC_MODEL_DIR / "bc_prior.pt",
        BC_MODEL_DIR / "bc_best.pt",
        BC_MODEL_DIR / "cnn_strided_bc.pt",
        PROJECT_ROOT / "logs" / "behavior_cloning_ckpts" / "latest.pt",
    ]
    return str(_first_existing(candidates))


def resolve_tokenizer_path(path: str | None = None) -> str:
    resolved = _resolve_explicit_or_directory_candidate(path, TOKENIZER_MODEL_DIR)
    if resolved is not None:
        return resolved
    resolved = _resolve_explicit_or_directory_candidate(path, TOKENIZER_FALLBACK_DIR)
    if resolved is not None:
        return resolved

    candidates = [
        TOKENIZER_MODEL_DIR / "latest.pt",
        TOKENIZER_MODEL_DIR / "step_0000000.pt",
        TOKENIZER_FALLBACK_DIR / "tokenizer.pt",
        TOKENIZER_FALLBACK_DIR / "latest.pt",
    ]
    return str(_first_existing(candidates))


def resolve_ppo_checkpoint_path(path: str | None = None) -> str:
    resolved = _resolve_explicit_or_directory_candidate(path, PPO_MODEL_DIR)
    if resolved is not None:
        return resolved

    candidates = [
        PPO_MODEL_DIR / "latest.pth",
        PPO_MODEL_DIR / "ppo_pusht_model.pth",
        PROJECT_ROOT / "logs" / "ppo_online_ckpts" / "latest.pth",
    ]
    return str(_first_existing(candidates))


def resolve_dynamics_checkpoint_path(path: str | None = None) -> str:
    resolved = _resolve_explicit_or_directory_candidate(path, DYNAMICS_MODEL_DIR)
    if resolved is not None:
        return resolved

    candidates = [
        DYNAMICS_MODEL_DIR / "latest.pt",
        DYNAMICS_MODEL_DIR / "step_0000000.pt",
    ]
    return str(_first_existing(candidates))
