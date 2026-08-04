#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavioural_cloning.eval_bc_exact import (
    BCImagePolicy,
    DEFAULT_PUSHT_ENV_ID,
    PushTDenseRewardWrapper,
    clean_state_dict_keys,
    ensemble_primitive,
    extract_checkpoint_args,
    extract_checkpoint_state_dict,
    extract_state_array,
    infer_cnn_image_hw,
    load_state_dict_safe,
    load_tokenizer_from_ckpt,
    make_pusht_env,
    map_primitive_to_env_action,
    pad_history,
    resolve_bc_prior_path,
    resolve_device,
    resolve_model_config,
    resolve_tokenizer_path,
)


SNAPSHOT_PREFIXES = ("epoch_", "step_")
SNAPSHOT_FILENAMES = {"latest.pt", "latest.pth", "second_best.pt", "second_best.pth", "model.pt", "model.pth"}


@dataclass
class EvalResult:
    success_rate: float
    mean_return: float
    mean_length: float
    mean_coverage: float | None


@dataclass
class SweepRow:
    checkpoint_name: str
    checkpoint_path: str
    family: str
    action_mode: str | None
    seq_len: int | None
    frame_stride: int | None
    chunk_size: int | None
    hidden_dim: int | None
    tokenizer: str | None
    success_rate: float
    mean_return: float
    mean_length: float
    mean_coverage: float | None


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


def _infer_family(ckpt_args: dict[str, Any]) -> str:
    return "Frozen Tokenizer Encoder" if ckpt_args.get("tokenizer_ckpt_name") else "CNN"


def _is_snapshot(path: Path) -> bool:
    return path.name in SNAPSHOT_FILENAMES or any(path.name.startswith(prefix) for prefix in SNAPSHOT_PREFIXES)


def discover_checkpoints(base_dir: Path, *, include_snapshots: bool) -> list[Path]:
    candidates: list[Path] = []

    nested_best = sorted(path for path in base_dir.rglob("best.pt") if path.is_file())
    candidates.extend(nested_best)

    for pattern in ("*.pt", "*.pth"):
        for path in sorted(base_dir.glob(pattern)):
            if path.is_file():
                candidates.append(path)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if not include_snapshots and _is_snapshot(path) and path.name != "best.pt":
            continue
        unique.append(path)
    return unique


def _success_from_info(info: dict[str, Any], terminated: bool) -> float:
    for key in ("success", "is_success", "task_success", "block_success"):
        if key in info:
            try:
                return float(info[key])
            except Exception:
                pass
    return float(bool(terminated))


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    block_start_radius: float | None,
    fixed_target_eval: bool,
) -> tuple[dict[str, Any], EvalResult]:
    device = resolve_device("auto")
    checkpoint_path = Path(resolve_bc_prior_path(str(checkpoint_path)))
    payload = load_state_dict_safe(checkpoint_path, torch.device("cpu"))
    state_dict = extract_checkpoint_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unsupported BC checkpoint format in {checkpoint_path}")

    cleaned_state = clean_state_dict_keys(state_dict)
    ckpt_args = extract_checkpoint_args(payload)
    model_cfg = resolve_model_config(ckpt_args, cleaned_state)
    effective_action_mode = str(model_cfg["action_mode"])

    tokenizer_path = None
    image_hw = (224, 224)
    if model_cfg["tokenizer_name"] is not None:
        tokenizer_path = resolve_tokenizer_path(str(model_cfg["tokenizer_name"]))
        _, tokenizer_info = load_tokenizer_from_ckpt(tokenizer_path, torch.device("cpu"))
        image_hw = (int(tokenizer_info["H"]), int(tokenizer_info["W"]))
    else:
        image_hw = infer_cnn_image_hw(model_cfg)

    model = BCImagePolicy(
        image_shape=(image_hw[0], image_hw[1], 3),
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        action_chunk_size=model_cfg["action_chunk_size"],
        seq_len=model_cfg["seq_len"],
        tokenizer_ckpt=tokenizer_path,
        tokenizer_feature_dim=model_cfg["tokenizer_feature_dim"],
        policy_style=model_cfg["policy_style"],
        temporal_layers=model_cfg["temporal_layers"],
        temporal_heads=model_cfg["temporal_heads"],
        temporal_context=model_cfg["temporal_context"],
        backbone_device=device,
        action_output_tanh=model_cfg["action_output_tanh"],
    ).to(device)
    model.load_state_dict(cleaned_state, strict=True)
    model.eval()

    env = make_pusht_env(
        env_id=DEFAULT_PUSHT_ENV_ID,
        render_mode="rgb_array",
        image_height=image_hw[0],
        image_width=image_hw[1],
        relative=(effective_action_mode in {"relative", "swm_relative"}),
        sync_goal_pose=True,
        align_sampled_goal_to_fixed_target=fixed_target_eval,
        render_obs=False,
        block_start_near_goal=block_start_radius is not None,
        block_start_radius=float(block_start_radius or 0.0),
    )
    env = PushTDenseRewardWrapper(env, env_id=DEFAULT_PUSHT_ENV_ID)

    all_returns: list[float] = []
    all_lengths: list[int] = []
    all_coverages: list[float] = []
    all_successes: list[float] = []

    for episode_idx in range(episodes):
        env.reset(seed=seed + episode_idx)
        max_history_len = (int(model_cfg["seq_len"]) - 1) * int(model_cfg["frame_stride"]) + 1
        frame_history: deque[np.ndarray] = deque(maxlen=max_history_len)
        action_buffer: deque[np.ndarray] = deque()
        pending_chunks: deque[dict[str, np.ndarray | int]] = deque()
        done = False
        total_reward = 0.0
        step_count = 0
        final_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        current_eef = extract_state_array(env.unwrapped._get_obs())[0:2].astype(np.float32)

        while not done and step_count < max_steps:
            frame = np.asarray(env.render(), dtype=np.uint8)
            frame_history.append(frame)

            if not action_buffer:
                stacked_frames = pad_history(
                    frame_history,
                    model_cfg["seq_len"],
                    model_cfg["frame_stride"],
                )
                input_tensor = torch.as_tensor(stacked_frames[None], dtype=torch.uint8, device=device)
                with torch.no_grad():
                    action_chunk = model.predict_action_chunk(input_tensor).squeeze(0).detach().cpu().numpy()
                action_buffer.extend(action_chunk)

            primitive = np.asarray(action_buffer.popleft(), dtype=np.float32)
            env_action = map_primitive_to_env_action(
                primitive,
                mode=effective_action_mode,
                current_eef=current_eef,
                max_step_pixels=15.0,
                action_output_tanh=bool(model_cfg["action_output_tanh"]),
            )
            obs, reward, terminated, truncated, info = env.step(env_action)
            state = extract_state_array(obs)
            current_eef = state[0:2].astype(np.float32)
            total_reward += float(reward)
            step_count += 1
            final_info = dict(info)
            done = bool(terminated or truncated)

            if pending_chunks:
                for entry in pending_chunks:
                    entry["offset"] = int(entry["offset"]) + 1
                while pending_chunks and int(pending_chunks[0]["offset"]) >= model_cfg["action_chunk_size"]:
                    pending_chunks.popleft()

        all_returns.append(total_reward)
        all_lengths.append(step_count)
        if "coverage" in final_info:
            all_coverages.append(float(final_info["coverage"]))
        all_successes.append(_success_from_info(final_info, bool(terminated)))

    env.close()
    return ckpt_args, EvalResult(
        success_rate=float(np.mean(all_successes)) if all_successes else 0.0,
        mean_return=float(np.mean(all_returns)) if all_returns else 0.0,
        mean_length=float(np.mean(all_lengths)) if all_lengths else 0.0,
        mean_coverage=(float(np.mean(all_coverages)) if all_coverages else None),
    )


def write_markdown(rows: list[SweepRow], path: Path, *, block_start_radius: float | None, seed: int, episodes: int) -> None:
    lines = [
        "# BC Eval Sweep",
        "",
        f"- `episodes={episodes}`",
        f"- `seed={seed}`",
        f"- `block_start_radius={block_start_radius}`",
        "",
        "| Checkpoint | Family | Action | Chunk | Hidden | Success | Return | Length | Coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.checkpoint_name} | "
            f"{row.family} | "
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
        f"{'checkpoint':28} {'family':24} {'chunk':>5} {'hidden':>6} "
        f"{'success':>8} {'return':>9}"
    )
    print("-" * 92)
    for row in rows:
        print(
            f"{row.checkpoint_name[:28]:28} "
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
    parser.add_argument("--fixed-target-eval", action="store_true", default=True)
    parser.add_argument("--include-snapshots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(base_dir, include_snapshots=bool(args.include_snapshots))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {base_dir}")

    rows: list[SweepRow] = []
    raw_results: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        print(f"Evaluating: {checkpoint_path}")
        ckpt_args, result = evaluate_checkpoint(
            checkpoint_path,
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            block_start_radius=float(args.block_start_radius) if args.block_start_radius is not None else None,
            fixed_target_eval=bool(args.fixed_target_eval),
        )
        row = SweepRow(
            checkpoint_name=checkpoint_path.name if checkpoint_path.parent == base_dir else str(checkpoint_path.relative_to(base_dir)),
            checkpoint_path=str(checkpoint_path.resolve()),
            family=_infer_family(ckpt_args),
            action_mode=ckpt_args.get("action_mode"),
            seq_len=_safe_int(ckpt_args.get("seq_len")),
            frame_stride=_safe_int(ckpt_args.get("frame_stride")),
            chunk_size=_safe_int(ckpt_args.get("action_chunk_size")),
            hidden_dim=_safe_int(ckpt_args.get("hidden_dim")),
            tokenizer=ckpt_args.get("tokenizer_ckpt_name"),
            success_rate=result.success_rate,
            mean_return=result.mean_return,
            mean_length=result.mean_length,
            mean_coverage=result.mean_coverage,
        )
        rows.append(row)
        raw_results.append(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "args": ckpt_args,
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
