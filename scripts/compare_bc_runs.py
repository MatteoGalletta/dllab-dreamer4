from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalSummary:
    success_rate: float | None = None
    mean_return: float | None = None
    mean_length: float | None = None
    mean_coverage: float | None = None
    metrics_path: str | None = None


@dataclass
class BCRunRow:
    run_dir: str
    checkpoint: str
    family: str
    variant: str
    dataset: str | None
    action_mode: str | None
    seq_len: int | None
    frame_stride: int | None
    chunk_size: int | None
    hidden_dim: int | None
    cnn_feature_dim: int | None
    augment: bool | None
    epochs: int | None
    batch_size: int | None
    lr: float | None
    train_step: int | None
    train_epoch: int | None
    eval_success_rate: float | None
    eval_mean_return: float | None
    eval_mean_length: float | None
    eval_mean_coverage: float | None
    eval_metrics_path: str | None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")


def _infer_family(args: dict[str, Any]) -> tuple[str, str]:
    has_tokenizer = bool(args.get("tokenizer_ckpt_name"))
    has_cnn = "cnn_feature_dim" in args
    augment = bool(args.get("augment", False))

    if has_tokenizer:
        family = "Frozen Tokenizer Encoder"
        variant = "Tokenizer+BC"
    elif has_cnn:
        family = "CNN"
        variant = "CNN+DataAug" if augment else "CNN"
    else:
        family = "Other"
        variant = "Other"
    return family, variant


def _match_checkpoint(eval_checkpoint: str | None, checkpoint_path: Path) -> bool:
    if not eval_checkpoint:
        return False
    eval_path = str(eval_checkpoint).replace("\\", "/")
    ckpt_abs = str(checkpoint_path.resolve()).replace("\\", "/")
    ckpt_rel = str(checkpoint_path).replace("\\", "/")
    return eval_path == ckpt_abs or eval_path.endswith(ckpt_rel) or ckpt_abs.endswith(eval_path)


def _matches_block_start_radius(
    config: dict[str, Any],
    required_block_start_radius: float | None,
) -> bool:
    if required_block_start_radius is None:
        return True
    value = config.get("block_start_radius")
    actual = _safe_float(value)
    if actual is None:
        return False
    return abs(actual - float(required_block_start_radius)) <= 1e-6


def _best_eval_for_checkpoint(
    eval_root: Path,
    checkpoint_path: Path,
    *,
    required_block_start_radius: float | None = None,
) -> EvalSummary:
    if not eval_root.exists():
        return EvalSummary()

    best: EvalSummary | None = None
    for metrics_path in sorted(eval_root.rglob("metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        config = payload.get("config", {}) or {}
        checkpoint = config.get("checkpoint")
        if not _match_checkpoint(checkpoint, checkpoint_path):
            continue
        if not _matches_block_start_radius(config, required_block_start_radius):
            continue

        summary = payload.get("summary", {}) or {}
        candidate = EvalSummary(
            success_rate=_safe_float(summary.get("success_rate")),
            mean_return=_safe_float(summary.get("mean_return")),
            mean_length=_safe_float(summary.get("mean_length")),
            mean_coverage=_safe_float(summary.get("mean_coverage")),
            metrics_path=str(metrics_path),
        )
        if best is None:
            best = candidate
            continue

        cand_key = (-1.0 if candidate.success_rate is None else candidate.success_rate, -1.0 if candidate.mean_return is None else candidate.mean_return)
        best_key = (-1.0 if best.success_rate is None else best.success_rate, -1.0 if best.mean_return is None else best.mean_return)
        if cand_key > best_key:
            best = candidate

    return best or EvalSummary()


def collect_rows(
    base_dir: Path,
    eval_root: Path,
    *,
    required_block_start_radius: float | None = None,
) -> list[BCRunRow]:
    rows: list[BCRunRow] = []
    for checkpoint_path in sorted(base_dir.glob("*/best.pt")):
        try:
            ckpt = _load_torch_checkpoint(checkpoint_path)
        except Exception as exc:
            rows.append(
                BCRunRow(
                    run_dir=checkpoint_path.parent.name,
                    checkpoint=str(checkpoint_path),
                    family="Unreadable",
                    variant=f"error: {exc}",
                    dataset=None,
                    action_mode=None,
                    seq_len=None,
                    frame_stride=None,
                    chunk_size=None,
                    hidden_dim=None,
                    cnn_feature_dim=None,
                    augment=None,
                    epochs=None,
                    batch_size=None,
                    lr=None,
                    train_step=None,
                    train_epoch=None,
                    eval_success_rate=None,
                    eval_mean_return=None,
                    eval_mean_length=None,
                    eval_mean_coverage=None,
                    eval_metrics_path=None,
                )
            )
            continue

        args = dict(ckpt.get("args") or {})
        family, variant = _infer_family(args)
        eval_summary = _best_eval_for_checkpoint(
            eval_root,
            checkpoint_path,
            required_block_start_radius=required_block_start_radius,
        )
        rows.append(
            BCRunRow(
                run_dir=checkpoint_path.parent.name,
                checkpoint=str(checkpoint_path),
                family=family,
                variant=variant,
                dataset=args.get("dataset"),
                action_mode=args.get("action_mode"),
                seq_len=_safe_int(args.get("seq_len")),
                frame_stride=_safe_int(args.get("frame_stride")),
                chunk_size=_safe_int(args.get("action_chunk_size")),
                hidden_dim=_safe_int(args.get("hidden_dim")),
                cnn_feature_dim=_safe_int(args.get("cnn_feature_dim")),
                augment=(None if "augment" not in args else bool(args.get("augment"))),
                epochs=_safe_int(args.get("epochs")),
                batch_size=_safe_int(args.get("batch_size")),
                lr=_safe_float(args.get("lr")),
                train_step=_safe_int(ckpt.get("step")),
                train_epoch=_safe_int(ckpt.get("epoch")),
                eval_success_rate=eval_summary.success_rate,
                eval_mean_return=eval_summary.mean_return,
                eval_mean_length=eval_summary.mean_length,
                eval_mean_coverage=eval_summary.mean_coverage,
                eval_metrics_path=eval_summary.metrics_path,
            )
        )
    return rows


def write_csv(rows: list[BCRunRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(BCRunRow("", "", "", "", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(rows: list[BCRunRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BC Run Comparison",
        "",
        "| Run | Family | Variant | Chunk | Hidden | Aug | Action | Best Eval Success | Best Eval Return | Metrics |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        metrics_ref = row.eval_metrics_path or "-"
        lines.append(
            "| "
            f"{row.run_dir} | "
            f"{row.family} | "
            f"{row.variant} | "
            f"{row.chunk_size if row.chunk_size is not None else '-'} | "
            f"{row.hidden_dim if row.hidden_dim is not None else '-'} | "
            f"{'-' if row.augment is None else ('yes' if row.augment else 'no')} | "
            f"{row.action_mode or '-'} | "
            f"{_format_float(row.eval_success_rate)} | "
            f"{_format_float(row.eval_mean_return)} | "
            f"{metrics_ref} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Family=CNN` are direct pixel-based BC models.",
            "- `Variant=CNN+DataAug` highlights the augmented pixel BC runs.",
            "- `Family=Frozen Tokenizer Encoder` are latent BC runs using the pretrained tokenizer as encoder.",
            "- The strongest claim for the poster should use the real eval metrics (`success_rate`, `mean_return`) rather than training loss alone.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(rows: list[BCRunRow]) -> None:
    print("\nBC comparison summary\n")
    print(
        f"{'run':38} {'family':24} {'chunk':>5} {'hidden':>6} {'aug':>5} "
        f"{'success':>8} {'return':>9}"
    )
    print("-" * 108)
    for row in rows:
        aug = "-"
        if row.augment is not None:
            aug = "yes" if row.augment else "no"
        print(
            f"{row.run_dir[:38]:38} "
            f"{row.family[:24]:24} "
            f"{str(row.chunk_size or '-'):>5} "
            f"{str(row.hidden_dim or '-'):>6} "
            f"{aug:>5} "
            f"{_format_float(row.eval_success_rate):>8} "
            f"{_format_float(row.eval_mean_return):>9}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BC checkpoint families and merge canonical eval metrics.")
    parser.add_argument(
        "--base-dir",
        default="local_models/behavior_cloning",
        help="Directory containing BC run subdirectories with best.pt checkpoints.",
    )
    parser.add_argument(
        "--eval-root",
        default="runs/evaluations",
        help="Directory containing eval metrics.json files.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/bc_comparison",
        help="Directory where comparison CSV/Markdown/JSON files will be written.",
    )
    parser.add_argument(
        "--family-filter",
        nargs="*",
        default=None,
        help="Optional family filter, e.g. --family-filter CNN 'Frozen Tokenizer Encoder'",
    )
    parser.add_argument(
        "--block-start-radius",
        type=float,
        default=None,
        help="Only use eval metrics whose config.block_start_radius matches this value exactly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)

    rows = collect_rows(
        base_dir,
        eval_root,
        required_block_start_radius=args.block_start_radius,
    )
    if args.family_filter:
        allowed = set(args.family_filter)
        rows = [row for row in rows if row.family in allowed]

    rows.sort(
        key=lambda row: (
            row.family,
            -(row.eval_success_rate if row.eval_success_rate is not None else -1.0),
            -(row.eval_mean_return if row.eval_mean_return is not None else -1.0),
            row.run_dir,
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "bc_runs.csv")
    write_markdown(rows, out_dir / "bc_runs.md")
    (out_dir / "bc_runs.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    print_summary(rows)
    print(f"\nSaved CSV: {out_dir / 'bc_runs.csv'}")
    print(f"Saved Markdown: {out_dir / 'bc_runs.md'}")
    print(f"Saved JSON: {out_dir / 'bc_runs.json'}")


if __name__ == "__main__":
    main()
