from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_bc_prior_path(path: str | None = None) -> str:
    if path:
        explicit = Path(path)
        if explicit.is_absolute() or explicit.exists():
            return str(explicit)

    candidates = [
        PROJECT_ROOT / "local_models" / "behavior_cloning" / "bc_prior.pt",
        PROJECT_ROOT / "local_models" / "behavior_cloning" / "latest.pt",
        PROJECT_ROOT / "logs" / "behavior_cloning_ckpts" / "latest.pt",
    ]
    return str(_first_existing(candidates))


def resolve_tokenizer_path(path: str | None = None) -> str:
    if path:
        explicit = Path(path)
        if explicit.is_absolute() or explicit.exists():
            return str(explicit)

    candidates = [
        PROJECT_ROOT / "logs" / "tokenizer_ckpts" / "latest.pt",
        PROJECT_ROOT / "local_models" / "tokenizer" / "tokenizer.pt",
        PROJECT_ROOT / "local_models" / "tokenizer" / "latest.pt",
    ]
    return str(_first_existing(candidates))


def resolve_ppo_checkpoint_path(path: str | None = None) -> str:
    if path:
        explicit = Path(path)
        if explicit.is_absolute() or explicit.exists():
            return str(explicit)

    candidates = [
        PROJECT_ROOT / "local_models" / "ppo_online" / "ppo_pusht_model.pth",
        PROJECT_ROOT / "logs" / "ppo_online_ckpts" / "latest.pth",
    ]
    return str(_first_existing(candidates))
