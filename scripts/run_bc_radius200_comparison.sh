#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running BC radius-200 comparison from: $ROOT_DIR"
echo

python scripts/rerun_bc_eval_sweep.py \
  --base-dir local_models/behavior_cloning \
  --out-dir runs/bc_comparison_radius200 \
  --episodes 50 \
  --max-steps 300 \
  --seed 42 \
  --block-start-radius 200

echo
echo "Done."
echo "Comparison outputs:"
  echo "  $ROOT_DIR/runs/bc_comparison_radius200/bc_runs.md"
echo "  $ROOT_DIR/runs/bc_comparison_radius200/bc_runs.csv"
echo "  $ROOT_DIR/runs/bc_comparison_radius200/bc_runs.json"
