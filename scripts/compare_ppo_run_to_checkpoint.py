#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import pickle
import re
from pathlib import Path
from typing import Any


BEST_RE = re.compile(
    r"New best PPO checkpoint at update=(?P<update>\d+):\s+success_rate=(?P<success>[0-9.]+)\s+saved=(?P<path>\S+)"
)
SECOND_BEST_RE = re.compile(
    r"New second-best PPO checkpoint at update=(?P<update>\d+):\s+success_rate=(?P<success>[0-9.]+)\s+saved=(?P<path>\S+)"
)
UPDATE_RE = re.compile(
    r"update=(?P<update>\d+)/(?P<num_updates>\d+)\s+step=(?P<step>\d+).*?eval_success=(?P<success>[0-9.]+)"
)
INITIAL_RE = re.compile(
    r"initial_eval\s+step=0\s+success=(?P<success>[0-9.]+)\s+mean_return=(?P<mean_return>[0-9.]+)\s+mean_length=(?P<mean_length>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the best eval success reported in a PPO training log with a saved checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to PPO checkpoint, e.g. local_models/ppo_online/best.pth")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to a console log / pasted output file. If omitted, the script searches wandb/ for matching logs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional W&B run id (e.g. af78by8j). Used to narrow the wandb/ search when --log-file is omitted.",
    )
    parser.add_argument(
        "--wandb-root",
        default="wandb",
        help="Root directory containing local wandb runs. Default: wandb",
    )
    return parser.parse_args()


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def discover_log_files(wandb_root: Path, run_id: str | None) -> list[Path]:
    candidates: list[Path] = []
    if not wandb_root.exists():
        return candidates
    patterns = ["output.log", "debug.log"]
    for path in wandb_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name not in patterns:
            continue
        if run_id and run_id not in str(path):
            continue
        candidates.append(path)
    return sorted(candidates)


def parse_training_log(text: str) -> dict[str, Any]:
    best_events = []
    second_best_events = []
    updates = []
    initial_eval = None

    for line in text.splitlines():
        match = INITIAL_RE.search(line)
        if match:
            initial_eval = {
                "success_rate": _safe_float(match.group("success")),
                "mean_return": _safe_float(match.group("mean_return")),
                "mean_length": _safe_float(match.group("mean_length")),
            }
            continue

        match = BEST_RE.search(line)
        if match:
            best_events.append(
                {
                    "update": _safe_int(match.group("update")),
                    "success_rate": _safe_float(match.group("success")),
                    "path": match.group("path"),
                }
            )
            continue

        match = SECOND_BEST_RE.search(line)
        if match:
            second_best_events.append(
                {
                    "update": _safe_int(match.group("update")),
                    "success_rate": _safe_float(match.group("success")),
                    "path": match.group("path"),
                }
            )
            continue

        match = UPDATE_RE.search(line)
        if match:
            updates.append(
                {
                    "update": _safe_int(match.group("update")),
                    "num_updates": _safe_int(match.group("num_updates")),
                    "global_step": _safe_int(match.group("step")),
                    "success_rate": _safe_float(match.group("success")),
                }
            )

    best_from_updates = None
    if updates:
        best_from_updates = max(updates, key=lambda item: item["success_rate"])

    best_event = None
    if best_events:
        best_event = max(best_events, key=lambda item: item["success_rate"])

    return {
        "initial_eval": initial_eval,
        "best_events": best_events,
        "second_best_events": second_best_events,
        "updates": updates,
        "best_from_updates": best_from_updates,
        "best_event": best_event,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def summarize_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {}) or {}
    return {
        "success_rate": payload.get("success_rate"),
        "global_step": payload.get("global_step"),
        "reward_mode": config.get("reward_mode"),
        "block_start_radius": config.get("block_start_radius"),
        "block_start_near_goal": config.get("block_start_near_goal"),
        "frame_stride": config.get("frame_stride"),
        "obs_stack_size": config.get("obs_stack_size"),
        "chunk_size": config.get("chunk_size"),
        "tokenizer_path": config.get("tokenizer_path"),
        "action_mode": config.get("action_mode"),
    }


def print_report(log_path: Path, parsed: dict[str, Any], checkpoint_path: Path, checkpoint_info: dict[str, Any]) -> None:
    print(f"log_file={log_path}")
    print(f"checkpoint={checkpoint_path}")
    print()

    initial_eval = parsed.get("initial_eval")
    if initial_eval is not None:
        print(
            "initial_eval: "
            f"success_rate={initial_eval['success_rate']:.3f} "
            f"mean_return={initial_eval['mean_return']:.2f} "
            f"mean_length={initial_eval['mean_length']:.1f}"
        )
    else:
        print("initial_eval: not found")

    best_event = parsed.get("best_event")
    best_from_updates = parsed.get("best_from_updates")
    print()
    if best_event is not None:
        print(
            "best_event_from_log: "
            f"update={best_event['update']} "
            f"success_rate={best_event['success_rate']:.3f} "
            f"saved_path={best_event['path']}"
        )
    else:
        print("best_event_from_log: not found")

    if best_from_updates is not None:
        print(
            "best_eval_success_seen: "
            f"update={best_from_updates['update']} "
            f"global_step={best_from_updates['global_step']} "
            f"success_rate={best_from_updates['success_rate']:.3f}"
        )
    else:
        print("best_eval_success_seen: not found")

    print()
    print(
        "checkpoint_metadata: "
        f"success_rate={checkpoint_info['success_rate']} "
        f"global_step={checkpoint_info['global_step']} "
        f"reward_mode={checkpoint_info['reward_mode']} "
        f"block_start_radius={checkpoint_info['block_start_radius']} "
        f"frame_stride={checkpoint_info['frame_stride']} "
        f"chunk_size={checkpoint_info['chunk_size']} "
        f"action_mode={checkpoint_info['action_mode']}"
    )

    print()
    mismatches = []
    if best_event is not None and checkpoint_info["success_rate"] is not None:
        best_success = float(best_event["success_rate"])
        ckpt_success = float(checkpoint_info["success_rate"])
        if not math.isclose(best_success, ckpt_success, rel_tol=0.0, abs_tol=1e-6):
            mismatches.append(
                f"checkpoint success_rate {ckpt_success:.3f} does not match best log success_rate {best_success:.3f}"
            )

    if best_from_updates is not None and checkpoint_info["global_step"] is not None:
        best_step = int(best_from_updates["global_step"])
        ckpt_step = int(checkpoint_info["global_step"])
        if best_step != ckpt_step:
            mismatches.append(
                f"checkpoint global_step {ckpt_step} does not match best-eval global_step {best_step}"
            )

    if mismatches:
        print("comparison_result=MISMATCH")
        for item in mismatches:
            print(f"- {item}")
    else:
        print("comparison_result=MATCH")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log_files: list[Path]
    if args.log_file is not None:
        log_files = [Path(args.log_file).resolve()]
    else:
        log_files = discover_log_files(Path(args.wandb_root), args.run_id)
        if not log_files:
            raise FileNotFoundError(
                f"Could not find any candidate log files under {args.wandb_root!r}"
                + (f" for run id {args.run_id!r}" if args.run_id else "")
            )

    payload = load_checkpoint(checkpoint_path)
    checkpoint_info = summarize_checkpoint(payload)

    best_choice: tuple[Path, dict[str, Any], float] | None = None
    for log_path in log_files:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        parsed = parse_training_log(text)
        score = -1.0
        if parsed.get("best_event") is not None:
            score = float(parsed["best_event"]["success_rate"])
        elif parsed.get("best_from_updates") is not None:
            score = float(parsed["best_from_updates"]["success_rate"])
        if best_choice is None or score > best_choice[2]:
            best_choice = (log_path, parsed, score)

    if best_choice is None:
        raise RuntimeError("Could not parse any usable PPO eval information from the provided logs.")

    print_report(best_choice[0], best_choice[1], checkpoint_path, checkpoint_info)


if __name__ == "__main__":
    main()
