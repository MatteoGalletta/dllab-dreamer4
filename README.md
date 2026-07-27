# Tokenizer-Latent Behavioral Cloning on PushT

This repository contains the current behavioral cloning (BC) experiments for PushT using:

- a pretrained tokenizer as the image encoder,
- a small MLP policy on stacked latent features,
- the `pusht_expert.npz` expert dataset,
- and a canonical fixed-target PushT evaluation protocol.

This README is meant as a short handoff for review and reproduction.

## Current Status

As of July 18, 2026:

- The original BC setup with larger action chunks did not reliably solve PushT.
- Switching to the `.npz` expert dataset and training with `action_chunk_size=1` produced clearly better behavior.
- With `chunk_size=1`, the BC model consistently moves toward the T-block and can solve some evaluation episodes.
- `chunk_size=5` is still much harder in the current tokenizer-based setup.
- The current tokenizer was originally trained on the older `.h5` data distribution and a different image resolution.
- Because of that mismatch, `.npz` images are currently resized to match the tokenizer input resolution.
- A clean next step is to retrain the tokenizer on the new `.npz` dataset, then retrain BC and retry larger chunk sizes such as `2`, `3`, and `5`.

## What Changed

The main changes from the older BC baseline are:

1. Dataset change
   - Old dataset: `pusht_expert_train.h5`
   - New dataset: `data/expert_trajectories/pusht_expert.npz`

2. Action semantics change
   - The old `.h5` setup aligned more naturally with relative/delta actions.
   - The new `.npz` dataset uses absolute action targets, which match the current successful BC setup better.

3. BC training script
   - New script: [behavioural_cloning/train_tokenizer_latent_bc.py](/home/dennis/2nd%20semester/dllab-dreamer4/behavioural_cloning/train_tokenizer_latent_bc.py)
   - This script caches tokenizer features, trains a simple latent BC MLP, and saves `best.pt`, `epoch_XXXX.pt`, and `latest.pt`.

4. Evaluation script
   - New script: [behavioural_cloning/eval_tokenizer_latent_bc_exact.py](/home/dennis/2nd%20semester/dllab-dreamer4/behavioural_cloning/eval_tokenizer_latent_bc_exact.py)
   - It now supports a canonical PushT-style eval protocol:
     - fixed target pose,
     - `--block-start-radius`,
     - `--seed`,
     - `--max-episode-steps`,
     - open-loop or temporal-ensemble execution,
     - per-episode success/fail videos,
     - run directories with `metrics.json`.

## Dataset Notes

### Old dataset: `pusht_expert_train.h5`

- many more transitions and episodes,
- images at a larger resolution,
- action values behave like small relative deltas.

### New dataset: `pusht_expert.npz`

- fewer transitions and fewer episodes,
- images at lower resolution,
- action values behave like absolute workspace positions,
- better matched to the current BC rollout semantics.

### Why this matters

The most important difference is not only image resolution, but action meaning:

- old dataset: "move by `(dx, dy)`"
- new dataset: "move to `(x, y)`"

The current successful BC setup interprets model outputs as absolute actions, so the `.npz` dataset is a better fit.

## Current Architecture

The current BC policy is:

- tokenizer encoder -> latent features,
- stack `seq_len` latent vectors,
- flatten them,
- pass them through a 2-layer MLP,
- predict `action_chunk_size * action_dim`.

This is implemented in [behavioural_cloning/train_tokenizer_latent_bc.py](/home/dennis/2nd%20semester/dllab-dreamer4/behavioural_cloning/train_tokenizer_latent_bc.py).

## Recommended Training Setup

The current working baseline is:

- dataset: `.npz`
- action mode: absolute
- `seq_len=3`
- `frame_stride=5`
- `action_chunk_size=1`

Example command:

```bash
python behavioural_cloning/train_tokenizer_latent_bc.py \
  --dataset data/expert_trajectories/pusht_expert.npz \
  --tokenizer_ckpt_name logs/tokenizer_ckpts/tokenizer.pt \
  --seq_len 3 \
  --frame_stride 5 \
  --action_chunk_size 1 \
  --hidden_dim 256 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-3 \
  --num_workers 0 \
  --val_max_batches 50 \
  --ckpt_dir local_models/behavior_cloning/tokenizer_latent_bc_npz_chunk1 \
  --wandb_mode online \
  --wandb_run_name tokenizer-latent-bc-npz-chunk1
```

Note:

- The run only uses `chunk=1` if `--action_chunk_size 1` is set.
- The checkpoint directory name is only a folder label. If the folder is accidentally named `chunk5` but the flag is `1`, the actual run is still `chunk=1`.

## Checkpoints

During training, the script writes:

- `best.pt`: lowest validation loss so far
- `epoch_0010.pt`, `epoch_0020.pt`, ...: periodic checkpoints
- `latest.pt`: written at the end of training

The best checkpoint does not update unless validation improves. It can therefore have an older timestamp than the currently running training job.

## Evaluation

### Quick local eval without video

```bash
python behavioural_cloning/eval_tokenizer_latent_bc_exact.py \
  --checkpoint local_models/behavior_cloning/tokenizer_latent_bc_npz_chunk1/best.pt \
  --episodes 10 \
  --max-steps 300 \
  --video_path "" \
  --eval_seed 42
```

### Canonical-style comparison eval

This is the closest equivalent to the other group's evaluator:

```bash
python behavioural_cloning/eval_tokenizer_latent_bc_exact.py \
  --checkpoint local_models/behavior_cloning/tokenizer_latent_bc_npz_chunk1/best.pt \
  --block-start-radius 200 \
  --episodes 50 \
  --max-episode-steps 300 \
  --video \
  --seed 42 \
  --execution-mode open-loop
```

This produces:

- a timestamped run directory under `runs/evaluations/`
- per-episode videos under `runs/evaluations/.../videos/`
- a `metrics.json` file summarizing the run

### Temporal ensembling

Temporal ensembling can also be tested:

```bash
python behavioural_cloning/eval_tokenizer_latent_bc_exact.py \
  --checkpoint local_models/behavior_cloning/tokenizer_latent_bc_npz_chunk1/best.pt \
  --block-start-radius 200 \
  --episodes 50 \
  --max-episode-steps 300 \
  --video \
  --seed 42 \
  --execution-mode temporal-ensemble \
  --temporal-ensemble-decay 0.01
```

In practice, temporal ensembling is most meaningful for chunk sizes greater than `1`. With `chunk_size=1`, the effect is usually limited.

## Video Output

Evaluation videos are saved one file per episode, like:

- `episode_000_success.mp4`
- `episode_001_fail.mp4`

If `--video` is used, videos go into the evaluation run directory:

- `runs/evaluations/<timestamp>_bc_<checkpoint-stem>/videos/`

## Useful Files

- Training: [behavioural_cloning/train_tokenizer_latent_bc.py](/home/dennis/2nd%20semester/dllab-dreamer4/behavioural_cloning/train_tokenizer_latent_bc.py)
- Evaluation: [behavioural_cloning/eval_tokenizer_latent_bc_exact.py](/home/dennis/2nd%20semester/dllab-dreamer4/behavioural_cloning/eval_tokenizer_latent_bc_exact.py)
- NPZ downloader: [scripts/download_pusht_npz.py](/home/dennis/2nd%20semester/dllab-dreamer4/scripts/download_pusht_npz.py)

