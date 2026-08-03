#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running BC radius-200 comparison from: $ROOT_DIR"
echo

python behavioural_cloning/eval_cnn_bc_exact.py \
  --checkpoint local_models/behavior_cloning/cnn_bc_abs_chunk5_h512_b128_aug/best.pt \
  --episodes 50 \
  --max-steps 300 \
  --block-start-radius 200 \
  --seed 42

python behavioural_cloning/eval_cnn_bc_exact.py \
  --checkpoint local_models/behavior_cloning/cnn_bc_abs_chunk5_h512_b128/best.pt \
  --episodes 50 \
  --max-steps 300 \
  --block-start-radius 200 \
  --seed 42

python behavioural_cloning/eval_tokenizer_latent_bc_exact.py \
  --checkpoint local_models/behavior_cloning/tokenizer_latent_bc_npz224_chunk5_2048/best.pt \
  --episodes 50 \
  --max-steps 300 \
  --block-start-radius 200 \
  --eval_seed 42

python behavioural_cloning/eval_tokenizer_latent_bc_exact.py \
  --checkpoint local_models/behavior_cloning/tokenizer_latent_bc_npz224_chunk5/best.pt \
  --episodes 50 \
  --max-steps 300 \
  --block-start-radius 200 \
  --eval_seed 42

python scripts/compare_bc_runs.py \
  --base-dir local_models/behavior_cloning \
  --eval-root runs/evaluations \
  --out-dir runs/bc_comparison_radius200 \
  --family-filter CNN "Frozen Tokenizer Encoder" \
  --block-start-radius 200

echo
echo "Done."
echo "Comparison outputs:"
echo "  $ROOT_DIR/runs/bc_comparison_radius200/bc_runs.md"
echo "  $ROOT_DIR/runs/bc_comparison_radius200/bc_runs.csv"
echo "  $ROOT_DIR/runs/bc_comparison_radius200/bc_runs.json"
