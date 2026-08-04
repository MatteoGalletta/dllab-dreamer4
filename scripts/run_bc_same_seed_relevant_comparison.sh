#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-42}"
RADIUS="${2:-200}"
EPISODES="${3:-50}"

echo "Running same-seed BC comparison from: $ROOT_DIR"
echo "  seed=$SEED"
echo "  block_start_radius=$RADIUS"
echo "  episodes=$EPISODES"
echo

python scripts/rerun_bc_eval_sweep.py \
  --base-dir local_models/behavior_cloning \
  --out-dir "runs/bc_comparison_seed${SEED}_r${RADIUS}" \
  --episodes "$EPISODES" \
  --max-steps 300 \
  --seed "$SEED" \
  --block-start-radius "$RADIUS" \
  --include-substring cnn_bc_abs_chunk5_h512_b128 \
  --include-substring cnn_bc_abs_chunk5_h512_b128_aug \
  --include-substring tokenizer_latent_bc_npz224_chunk5 \
  --include-substring tokenizer_latent_bc_npz224_chunk5_2048 \
  --include-substring tokenizer_latent_bc_npz224_chunk8 \
  --include-substring tokenizer_latent_bc_npz224_chunk10

echo
echo "Done."
echo "Comparison outputs:"
echo "  $ROOT_DIR/runs/bc_comparison_seed${SEED}_r${RADIUS}/bc_runs.md"
echo "  $ROOT_DIR/runs/bc_comparison_seed${SEED}_r${RADIUS}/bc_runs.csv"
echo "  $ROOT_DIR/runs/bc_comparison_seed${SEED}_r${RADIUS}/bc_runs.json"
