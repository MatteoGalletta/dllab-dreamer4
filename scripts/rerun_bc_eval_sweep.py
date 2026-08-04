#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SNAPSHOT_PREFIXES = ("epoch_", "step_")
SNAPSHOT_FILENAMES = {"latest.pt", "latest.pth", "second_best.pt", "second_best.pth", "model.pt", "model.pth"}


@dataclass
class EvalResult:
    success_rate: float
    mean_return: float
    mean_length: float
    mean_coverage: float | None
    metrics_path: str | None


@dataclass
class SweepRow:
    checkpoint_name: str
    checkpoint_path: str
    evaluator: str
    family: str
    action_mode: str | None
    seq_len: int | None
    frame_stride: int | None
    chunk_size: int | None
    hidden_dim: int | None
    cnn_feature_dim: int | None
    augment: bool | None
    success_rate: float
    mean_return: float
    mean_length: float
    mean_coverage: float | None
    metrics_path: str | None


def _safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value)).strip("-._") or "eval"


def _is_snapshot(path: Path) -> bool:
    return path.name in SNAPSHOT_FILENAMES or any(path.name.startswith(prefix) for prefix in SNAPSHOT_PREFIXES)


def discover_checkpoints(base_dir: Path, *, include_snapshots: bool) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(sorted(path for path in base_dir.rglob("best.pt") if path.is_file()))
    candidates.extend(sorted(path for path in base_dir.rglob("best.pth") if path.is_file()))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if not include_snapshots and _is_snapshot(path) and path.name not in {"best.pt", "best.pth"}:
            continue
        unique.append(path)
    return unique


def _matches_filters(
    path: Path,
    *,
    include_substrings: list[str],
    exclude_substrings: list[str],
) -> bool:
    haystack = path.as_posix().lower()
    if include_substrings and not any(pattern in haystack for pattern in include_substrings):
        return False
    if exclude_substrings and any(pattern in haystack for pattern in exclude_substrings):
        return False
    return True


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{checkpoint_path} is not a dict checkpoint")
    return payload


def _checkpoint_args(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("args")
    return dict(args) if isinstance(args, dict) else {}


def _infer_evaluator(checkpoint_path: Path, ckpt_args: dict[str, Any]) -> str:
    if "cnn_feature_dim" in ckpt_args:
        return "cnn_exact"
    if ckpt_args.get("tokenizer_ckpt_name"):
        return "tokenizer_latent_exact"
    name = checkpoint_path.as_posix().lower()
    if "cnn" in name:
        return "cnn_exact"
    if "tokenizer" in name or "latent" in name:
        return "tokenizer_latent_exact"
    return "unknown"


def _infer_family(ckpt_args: dict[str, Any], evaluator: str) -> str:
    if evaluator == "cnn_exact":
        return "CNN"
    if evaluator == "tokenizer_latent_exact":
        return "Frozen Tokenizer Encoder"
    return "Unknown"


def _find_metrics_for_checkpoint(eval_root: Path, checkpoint_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    target_abs = str(checkpoint_path.resolve()).replace("\\", "/")
    target_rel = str(checkpoint_path).replace("\\", "/")
    matches: list[tuple[float, Path, dict[str, Any]]] = []
    for metrics_path in eval_root.rglob("metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = payload.get("config", {}) or {}
        logged_checkpoint = str(config.get("checkpoint", "")).replace("\\", "/")
        if not logged_checkpoint:
            continue
        if logged_checkpoint == target_abs or logged_checkpoint.endswith(target_rel) or target_abs.endswith(logged_checkpoint):
            matches.append((metrics_path.stat().st_mtime, metrics_path, payload))
    if not matches:
        return None, None
    matches.sort(key=lambda item: item[0], reverse=True)
    _, path, payload = matches[0]
    return path, payload


def _run_command(cmd: list[str]) -> None:
    print("  " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Evaluation command failed with exit code {completed.returncode}")


def _checkpoint_run_name(checkpoint_path: Path) -> str:
    checkpoint_abs = checkpoint_path.resolve()
    bc_root = (PROJECT_ROOT / "local_models" / "behavior_cloning").resolve()
    try:
        suffix = checkpoint_abs.relative_to(bc_root)
        label = str(suffix)
    except ValueError:
        label = checkpoint_abs.stem
    return f"sweep-{_slug(label)}"


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    block_start_radius: float | None,
    out_dir: Path,
) -> tuple[dict[str, Any], str, EvalResult]:
    payload = _load_checkpoint_payload(checkpoint_path)
    ckpt_args = _checkpoint_args(payload)
    evaluator = _infer_evaluator(checkpoint_path, ckpt_args)
    eval_root = out_dir / "_eval_runs"
    eval_root.mkdir(parents=True, exist_ok=True)
    run_name = _checkpoint_run_name(checkpoint_path)

    if evaluator == "cnn_exact":
        cmd = [
            sys.executable,
            "behavioural_cloning/eval_cnn_bc_exact.py",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(int(episodes)),
            "--max-steps",
            str(int(max_steps)),
            "--seed",
            str(int(seed)),
            "--output-root",
            str(eval_root),
            "--run-name",
            run_name,
            "--video-path",
            "",
            "--wandb-mode",
            "disabled",
        ]
        if block_start_radius is not None:
            cmd.extend(["--block-start-radius", str(float(block_start_radius))])
    elif evaluator == "tokenizer_latent_exact":
        cmd = [
            sys.executable,
            "behavioural_cloning/eval_tokenizer_latent_bc_exact.py",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(int(episodes)),
            "--max-steps",
            str(int(max_steps)),
            "--eval_seed",
            str(int(seed)),
            "--output-root",
            str(eval_root),
            "--run-name",
            run_name,
            "--video-path",
            "",
            "--wandb-mode",
            "disabled",
        ]
        if block_start_radius is not None:
            cmd.extend(["--block-start-radius", str(float(block_start_radius))])
    else:
        raise RuntimeError(
            f"Unsupported checkpoint format for automated sweep: {checkpoint_path}. "
            "Only CNN exact and tokenizer latent exact checkpoints are supported."
        )

    _run_command(cmd)
    metrics_path, metrics_payload = _find_metrics_for_checkpoint(eval_root, checkpoint_path)
    if metrics_path is None or metrics_payload is None:
        raise RuntimeError(f"Could not find fresh metrics.json for {checkpoint_path} under {eval_root}")

    summary = metrics_payload.get("summary", {}) or {}
    return ckpt_args, evaluator, EvalResult(
        success_rate=float(summary.get("success_rate", 0.0)),
        mean_return=float(summary.get("mean_return", 0.0)),
        mean_length=float(summary.get("mean_length", 0.0)),
        mean_coverage=_safe_float(summary.get("mean_coverage")),
        metrics_path=str(metrics_path),
    )


def write_markdown(rows: list[SweepRow], path: Path, *, block_start_radius: float | None, seed: int, episodes: int) -> None:
    lines = [
        "# BC Eval Sweep",
        "",
        f"- `episodes={episodes}`",
        f"- `seed={seed}`",
        f"- `block_start_radius={block_start_radius}`",
        "",
        "| Checkpoint | Family | Evaluator | Action | Chunk | Hidden | Success | Return | Length | Coverage |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.checkpoint_name} | "
            f"{row.family} | "
            f"{row.evaluator} | "
            f"{row.action_mode or '-'} | "
            f"{row.chunk_size if row.chunk_size is not None else '-'} | "
            f"{row.hidden_dim if row.hidden_dim is not None else '-'} | "
            f"{_format_float(row.success_rate)} | "
            f"{_format_float(row.mean_return)} | "
            f"{_format_float(row.mean_length, 1)} | "
            f"{_format_float(row.mean_coverage)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[SweepRow], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_summary(rows: list[SweepRow]) -> None:
    print("\nBC comparison summary\n")
    print(
        f"{'checkpoint':36} {'family':24} {'chunk':>5} {'hidden':>6} "
        f"{'success':>8} {'return':>9}"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row.checkpoint_name[:36]:36} "
            f"{row.family[:24]:24} "
            f"{str(row.chunk_size or '-'):>5} "
            f"{str(row.hidden_dim or '-'):>6} "
            f"{_format_float(row.success_rate):>8} "
            f"{_format_float(row.mean_return):>9}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run the current BC checkpoint sweep and save a fresh comparison table.")
    parser.add_argument("--base-dir", default="local_models/behavior_cloning")
    parser.add_argument("--out-dir", default="runs/bc_comparison_current")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-start-radius", type=float, default=200.0)
    parser.add_argument("--include-snapshots", action="store_true")
    parser.add_argument(
        "--include-substring",
        action="append",
        default=[],
        help="Only evaluate checkpoints whose path contains this substring. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=[],
        help="Skip checkpoints whose path contains this substring. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(base_dir, include_snapshots=bool(args.include_snapshots))
    include_substrings = [value.lower() for value in args.include_substring]
    exclude_substrings = [value.lower() for value in args.exclude_substring]
    checkpoints = [
        path
        for path in checkpoints
        if _matches_filters(
            path,
            include_substrings=include_substrings,
            exclude_substrings=exclude_substrings,
        )
    ]
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {base_dir}")

    rows: list[SweepRow] = []
    raw_results: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        print(f"Evaluating: {checkpoint_path}")
        ckpt_args, evaluator, result = evaluate_checkpoint(
            checkpoint_path,
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            block_start_radius=float(args.block_start_radius) if args.block_start_radius is not None else None,
            out_dir=out_dir,
        )
        try:
            checkpoint_name = str(checkpoint_path.relative_to(base_dir))
        except ValueError:
            checkpoint_name = checkpoint_path.name
        row = SweepRow(
            checkpoint_name=checkpoint_name,
            checkpoint_path=str(checkpoint_path.resolve()),
            evaluator=evaluator,
            family=_infer_family(ckpt_args, evaluator),
            action_mode=ckpt_args.get("action_mode"),
            seq_len=_safe_int(ckpt_args.get("seq_len")),
            frame_stride=_safe_int(ckpt_args.get("frame_stride")),
            chunk_size=_safe_int(ckpt_args.get("action_chunk_size")),
            hidden_dim=_safe_int(ckpt_args.get("hidden_dim")),
            cnn_feature_dim=_safe_int(ckpt_args.get("cnn_feature_dim")),
            augment=(None if "augment" not in ckpt_args else bool(ckpt_args.get("augment"))),
            success_rate=result.success_rate,
            mean_return=result.mean_return,
            mean_length=result.mean_length,
            mean_coverage=result.mean_coverage,
            metrics_path=result.metrics_path,
        )
        rows.append(row)
        raw_results.append(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "args": ckpt_args,
                "evaluator": evaluator,
                "summary": asdict(result),
            }
        )

    rows.sort(key=lambda row: (-row.success_rate, -row.mean_return, row.checkpoint_name))

    write_csv(rows, out_dir / "bc_runs.csv")
    write_markdown(
        rows,
        out_dir / "bc_runs.md",
        block_start_radius=args.block_start_radius,
        seed=args.seed,
        episodes=args.episodes,
    )
    (out_dir / "bc_runs.json").write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8")
    (out_dir / "raw_eval_results.json").write_text(json.dumps(raw_results, indent=2) + "\n", encoding="utf-8")

    print_summary(rows)
    print(f"\nSaved CSV: {out_dir / 'bc_runs.csv'}")
    print(f"Saved Markdown: {out_dir / 'bc_runs.md'}")
    print(f"Saved JSON: {out_dir / 'bc_runs.json'}")
    print(f"Saved raw evals: {out_dir / 'raw_eval_results.json'}")


if __name__ == "__main__":
    main()
