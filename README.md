# Offline Reinforcement Learning with Dreamer4 for PushT

This repository studies whether a Dreamer4-style world model can improve a
visual behavior-cloning policy without additional interactions with the real
PushT simulator. The pipeline trains a masked visual tokenizer, an
action-conditioned latent dynamics model, a reward classifier, and then
fine-tunes a behavior-cloned policy with PPO either in the simulator or in
imagined latent rollouts.

The project was developed as a University of Freiburg Deep Learning Lab course
project. It is research code, not a production robotics system. Results are
sensitive to the evaluation distribution and random seed; use the provided
fixed protocol when comparing variants.

## Highlights

- PushT expert trajectories are converted from the public Diffusion Policy
  replay dataset and can be re-rendered at the tokenizer resolution.
- The visual tokenizer is trained with masked autoencoding and produces a
  `16 x 32 = 512` dimensional per-frame latent representation.
- The dynamics model predicts future tokenizer latents from previous latents,
  actions, and shortcut-flow conditioning.
- Two image-based BC variants are included: a direct CNN policy and a frozen
  tokenizer plus MLP policy.
- PPO supports both the real PushT simulator and offline imagined rollouts
  using the learned dynamics model and reward classifier.

## Poster Results

The following values are the final course-project poster results. All entries
use the same fixed PushT evaluation protocol: evaluation seed `7`, a block
start radius of `200 px` around the target, and a maximum of 300 environment
steps. They are single-protocol measurements, not confidence intervals.

| Stage | Variant | Success rate | Notes |
| --- | --- | ---: | --- |
| BC | CNN with augmentation | **26%** | Direct pixel encoder, chunk size 5, hidden size 512 |
| BC | CNN baseline | 14% | Same architecture without augmentation |
| BC | Frozen-tokenizer latent policy | 20% | Frozen Dreamer4 tokenizer, chunk size 5, hidden size 512 |
| PPO | Simulator fine-tuning | **20% -> 32%** | 537,039 real environment steps in the reported run |
| PPO | Dreamer4 imagination fine-tuning | **20% -> 32%** | No real environment steps during PPO optimization |

The tokenizer reached a reported masked-reconstruction accuracy of 96.85%.
The dynamics evaluation captured 65.5% visual-motion accuracy one step ahead
and 35.7% through step 15. These reconstruction and prediction metrics do not
by themselves guarantee the best control representation: in this setup,
augmentation made the direct CNN BC policy more robust than the frozen
tokenizer policy.

## Repository Layout

```text
behavioural_cloning/       CNN and frozen-tokenizer BC training and evaluation
data_pipeline/             Sequence dataset utilities
dreamer4-src/dreamer4/     Vendored Dreamer4 tokenizer, dynamics, and reward code
ppo_online/                PPO agents, simulator/imagination environments, evaluation
scripts/                   Data preparation, sweeps, diagnostics, and poster utilities
local_models/              Ignored local model checkpoints
logs/                      Ignored tokenizer, dynamics, and reward checkpoints
data/                      Ignored datasets
runs/                      Ignored evaluation outputs and figures
```

## Installation

Python 3.10 and a CUDA-capable PyTorch installation are recommended. Install
the PyTorch build that matches the target CUDA driver first, then install the
remaining dependencies:

```bash
git clone https://github.com/eiseled/dllab-dreamer4.git
cd dllab-dreamer4

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Weights & Biases tracking, authenticate separately and pass
`--wandb_mode online`. Every training command defaults to disabled or can be
run with `--wandb_mode offline`; no API key is stored in this repository.

## Data Preparation

Download and convert the public PushT expert replay dataset:

```bash
python scripts/download_pusht_npz.py
```

`download_data.py` remains available only for the older, much larger LeWM HDF5
dataset; it is not required for the reported results.

This creates `data/expert_trajectories/pusht_expert.npz`. To match the
tokenizer's `224 x 224` input, faithfully re-render the stored PushT states:

```bash
python scripts/regenerate_pusht_expert.py \
  --dataset data/expert_trajectories/pusht_expert.npz \
  --output_dataset data/expert_trajectories/pusht_expert_224.npz \
  --image_height 224 \
  --image_width 224 \
  --mode render
```

The re-rendered dataset is several gigabytes. Datasets, checkpoints, W&B
directories, videos, and run outputs are intentionally ignored by Git.

## Method

### 1. Causal visual tokenizer

The tokenizer consumes RGB image sequences `(B, T, 3, H, W)`. It patchifies
each frame and learns to reconstruct masked patches using block-causal
attention over prior frames. Its encoder emits `(B, T, 16, 32)` latent tokens,
which are flattened to a 512-dimensional vector per frame for BC and PPO.

```text
RGB frames -> masked patch encoder -> 16 x 32 latent tokens -> patch decoder
```

Train a tokenizer on the re-rendered data:

```bash
python train_tokenizer.py \
  --dataset data/expert_trajectories/pusht_expert_224.npz \
  --seq_len 8 \
  --H 224 --W 224 \
  --batch_size 16 \
  --action_chunk_size 5 \
  --max_steps 100000 \
  --ckpt_dir logs/tokenizer_ckpts \
  --wandb_mode disabled
```

### 2. Action-conditioned dynamics model

The frozen tokenizer encodes observation sequences. The dynamics model receives
packed latent tokens, padded action tokens, and shortcut signal/step tokens. It
predicts the next clean latent representation with a shortcut flow-matching
objective.

```text
(latent context, actions, shortcut conditioning) -> next latent prediction
```

```bash
python train_dynamics.py \
  --dataset data/expert_trajectories/pusht_expert_224.npz \
  --tokenizer_ckpt logs/tokenizer_ckpts/latest.pt \
  --use_actions \
  --seq_len 32 \
  --batch_size 16 \
  --max_steps 100000 \
  --ckpt_dir logs/dynamics_ckpts \
  --wandb_mode disabled
```

### 3. Reward classifier

For imagined PPO, a reward head maps imagined tokenizer latents to a success
probability. It is trained against the fixed PushT target pose.

```bash
python dreamer4-src/dreamer4/train_reward.py \
  --dataset data/expert_trajectories/pusht_expert_224.npz \
  --tokenizer_ckpt logs/tokenizer_ckpts/latest.pt \
  --batch_size 16 \
  --max_steps 50000 \
  --ckpt_dir logs/reward_ckpts
```

### 4. Behavior cloning policies

Both BC variants consume three frames separated by five environment steps and
predict a five-action PushT chunk. The `.npz` expert action targets are used in
the `absolute` action convention.

| Policy | Encoder | Policy head |
| --- | --- | --- |
| CNN BC | Trainable spatial-softmax CNN | MLP directly predicts 5 x 2 actions |
| Latent BC | Frozen `16 x 32` tokenizer representation | MLP over three stacked 512-D latents |

Reproduce the reported augmented CNN BC configuration:

```bash
python behavioural_cloning/train_cnn_bc.py \
  --dataset data/expert_trajectories/pusht_expert_224.npz \
  --action_mode absolute \
  --seq_len 3 --frame_stride 5 --action_chunk_size 5 \
  --cnn_feature_dim 256 --hidden_dim 512 \
  --batch_size 64 --eval_batch_size 32 \
  --epochs 100 --lr 1e-4 \
  --augment \
  --ckpt_dir local_models/behavior_cloning/cnn_bc_abs_chunk5_h512_aug \
  --wandb_mode disabled
```

Reproduce the frozen-tokenizer latent BC baseline:

```bash
python behavioural_cloning/train_tokenizer_latent_bc.py \
  --dataset data/expert_trajectories/pusht_expert_224.npz \
  --tokenizer_ckpt_name logs/tokenizer_ckpts/latest.pt \
  --action_mode absolute \
  --seq_len 3 --frame_stride 5 --action_chunk_size 5 \
  --hidden_dim 512 \
  --batch_size 64 --epochs 100 --lr 1e-3 \
  --ckpt_dir local_models/behavior_cloning/tokenizer_latent_bc_npz224_chunk5 \
  --wandb_mode disabled
```

## Evaluation

Use the same fixed protocol for fair comparisons. `open-loop` is the reported
execution mode: the policy predicts a chunk and executes all five actions
before predicting the next chunk.

```bash
# CNN policy
python behavioural_cloning/eval_cnn_bc_exact.py \
  --checkpoint local_models/behavior_cloning/cnn_bc_abs_chunk5_h512_aug/best.pt \
  --episodes 50 --max-steps 300 \
  --block-start-radius 200 --seed 7 \
  --execution-mode open-loop

# Frozen-tokenizer latent policy
python behavioural_cloning/eval_tokenizer_latent_bc_exact.py \
  --checkpoint local_models/behavior_cloning/tokenizer_latent_bc_npz224_chunk5/best.pt \
  --episodes 50 --max-steps 300 \
  --block-start-radius 200 --seed 7 \
  --execution-mode open-loop
```

Each evaluation creates `runs/evaluations/<timestamp>_*` containing
`metrics.json` and, with `--video`, per-episode videos. Temporal ensembling is
available through `--execution-mode temporal-ensemble`; it was not used for the
reported headline numbers.

Re-evaluate the available BC checkpoints under one protocol and write a table:

```bash
bash scripts/run_bc_same_seed_relevant_comparison.sh
```

## PPO Fine-Tuning

PPO starts from a BC policy. It samples an action chunk from a diagonal
Gaussian policy, executes it through either the real environment or the learned
world model, stores transitions, computes generalized advantages, and applies
clipped PPO updates. The optional prior loss keeps the actor near the BC policy
early in fine-tuning.

### Simulator PPO

This is the conservative real-environment configuration used for the final
comparison. It uses seed 7 for rollout sampling and evaluates every five PPO
updates with 25 episodes under evaluation seed 42. Change `--eval-seed 7` when
you need an exact match to the poster protocol.

```bash
python -m ppo_online.train \
  --env-source real --network-type bc_latent --vector-env manual \
  --bc-prior-path local_models/behavior_cloning/tokenizer_latent_bc_npz224_chunk5_2048/best.pt \
  --save-path local_models/ppo_online/ppo_simulator/model.pth \
  --hidden-dim 2048 --frame-stack 3 --frame-stride 5 --action-chunk-size 5 \
  --fixed-target --reward-mode sparse --block-start-radius 200 \
  --num-envs 8 --num-chunks 128 --total-timesteps 1000000 \
  --max-episode-steps 300 --seed 7 \
  --log-interval 5 --save-interval 10 --eval-interval 5 \
  --eval-episodes 25 --eval-seed 42 \
  --learning-rate 5e-6 --init-log-std -1.5 --no-anneal-log-std \
  --critic-warmup-ratio 0.1 --prior-loss-coef 0.02 \
  --no-bc-kl-penalty --wandb-mode disabled
```

### Dreamer4 imagination PPO

Imagined PPO samples a valid offline context from the dataset, encodes it with
the frozen tokenizer, predicts the next latent state with the dynamics model,
obtains a reward from the reward classifier, and repeats this for the imagined
horizon. A horizon of 15 chunk transitions bounds compounding world-model
error. The command below takes no real simulator steps during PPO rollout
collection; simulator episodes are only used for evaluation.

```bash
python -m ppo_online.train \
  --env-source imagination --network-type bc_latent --vector-env manual \
  --bc-prior-path local_models/behavior_cloning/tokenizer_latent_bc_npz224_chunk5_2048/best.pt \
  --tokenizer-path logs/tokenizer_ckpts/latest.pt \
  --imagination-dataset data/expert_trajectories/pusht_expert_224.npz \
  --dynamics-ckpt logs/dynamics_ckpts/latest.pt \
  --reward-ckpt logs/reward_ckpts/latest.pt \
  --imagination-context-len 5 --imagination-horizon 15 \
  --imagination-min-goal-dist 0 --imagination-max-goal-dist 200 \
  --save-path local_models/ppo_online/ppo_imagination/model.pth \
  --hidden-dim 2048 --frame-stack 3 --frame-stride 5 --action-chunk-size 5 \
  --reward-mode sparse --block-start-radius 200 \
  --num-envs 4 --num-chunks 32 --total-timesteps 1000000 \
  --seed 7 --learning-rate 5e-6 --init-log-std -1.5 --no-anneal-log-std \
  --critic-warmup-ratio 0.1 --prior-loss-coef 0.02 --no-bc-kl-penalty \
  --log-interval 5 --save-interval 10 --eval-interval 5 \
  --eval-episodes 25 --eval-seed 42 --wandb-mode disabled
```

Evaluate any PPO checkpoint with the shared fixed-target protocol:

```bash
python -m ppo_online.eval \
  --checkpoint local_models/ppo_online/ppo_imagination/best.pth \
  --episodes 50 --max-episode-steps 300 \
  --block-start-radius 200 --seed 7
```

## Reproducibility Notes

- Keep the dataset, tokenizer, BC checkpoint, and evaluation image resolution
  consistent. The reported latent experiments use the re-rendered 224 x 224
  dataset and a 224 x 224 tokenizer.
- Success rate and mean return are different metrics. Success is the fixed
  target-pose criterion; dense returns can increase without completing the
  final alignment.
- PPO evaluation does not affect gradient updates. It only selects and reports
  checkpoints, so small evaluation sets can make the best checkpoint noisy.
- The image model, reward classifier, and PPO are experimental. Do not treat
  the single-seed comparison as evidence that imagination universally matches
  simulator training.

## Artifact Policy and Security

No datasets, trained weights, W&B runs, SSH keys, API keys, or cloud
credentials are tracked. Put private values in shell environment variables
(for example `WANDB_API_KEY` or `HF_TOKEN`) rather than source files. Before
pushing a fork, run:

```bash
git status --ignored
git grep -nEI 'ghp_|github_pat_|api[_-]?key|secret|password'
```

## Attribution and License

`dreamer4-src/` is a vendored and adapted copy of the Dreamer4 PyTorch project
by Nicklas Hansen and remains covered by its included MIT license. Please cite
both the original Dreamer4 paper and this repository when reusing the PushT
adaptation. The project-level citation metadata is in `CITATION.cff`.

## Citation

```bibtex
@software{eisele2026offline_dreamer4_pusht,
  title   = {Offline Reinforcement Learning with Dreamer4 for PushT},
  author  = {Eisele, Dennis and Galletta, Matteo and Porta, Luca},
  year    = {2026},
  url     = {https://github.com/eiseled/dllab-dreamer4}
}
```
