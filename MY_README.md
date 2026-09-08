# Seven-experiment quick reproduction

Run every command from the repository root. Defaults are SIDReasoner's official split, 50 history items, Qwen3-0.6B, and seed 42. All seven experiments reuse the same released `.index.json`; they do not train a new SID codebook.

## 1. Install the environment

Purpose: create an isolated Python 3.11 environment, install the PyTorch 2.6.0 build matching the host CUDA stack, then install the remaining packages. Do not install this file into an Open WebUI or other shared service environment: an unrelated package's `pip check` conflicts do not describe the standalone DIPRec stack.

```bash
conda create -n diprec python=3.11 -y
conda activate diprec
# Install the CUDA-matched torch==2.6.0 wheel using the official PyTorch command first.
python -m pip install --upgrade pip
python -m pip install -r requirements-diprec.txt
python -c "import torch, trl; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'TRL', trl.__version__, 'GPU', torch.cuda.is_available())"
```

The `torch` line in `requirements-diprec.txt` stays commented. All four RL entries use pinned `trl==0.24.0` and `transformers==4.57.1`; they do not require VeRL, vLLM, FlashAttention, PEFT, or W&B. DIPRec adds a two-stage hierarchical override on top of TRL rather than using a separate optimizer loop.

These seven entries do not install VeRL, vLLM, or FlashAttention. If the retained original SIDReasoner/VeRL scripts are needed later, build that environment separately from the official VeRL documentation.

MiniOneRec here is a comparable shared-contract reproduction, not a line-for-line rerun of its upstream scripts. The enabled SFT task families (history SID→SID, SID→title, title→SID, and history SID→title), enabled RL task families (history SID→SID, title→SID, description→SID, and up to 10,000 title-history→SID rows), `G=16`, ranking-reward structure, and catalog constraint follow the official implementation. For SID reward comparison, DIPRec ignores internal whitespace to tolerate tokenizer-inserted spaces, whereas upstream MiniOneRec compares internal whitespace literally. All seven methods instead share this repository's Qwen chat prompt, reconstructed long-history data, AdamW-based schedules, checkpoint protocol, and evaluator. Each RL baseline freezes its own matching parent checkpoint—Direct-SFT for Direct-RL and MiniOneRec-SFT for MiniOneRec-RL—as the reference policy; this intentionally differs from upstream MiniOneRec's `sync_ref_model=True` recipe.

RL training uses MiniOneRec's constrained beam-sampling behavior (`do_sample=True`). The SID-ranking stage of all seven evaluations uses deterministic constrained beams (`do_sample=False`) and the same budget of 80 raw SID candidates. Candidates are ranked, deduplicated by SID, and truncated to **at most** Top-10; no duplicate is inserted to fill a short list. DIPRec divides the 80 candidates across plans (default `8 × 10`) and ranks trajectories jointly by `log p(plan) + log p(SID | plan)`. Its interest plans remain sampled under the fixed seed.

The recommended single-46-GB-GPU MiniOneRec-SFT recipe uses at most 6 epochs, micro-batch 8, accumulation 4 (effective batch 32), learning rate `1e-4`, and 3% cosine warmup. In the Video Games sweep, `1e-4` outperformed `5e-5` and `2e-4` on validation/test NDCG. Each completed epoch is persisted to `outputs/.../sft_training_metrics.json`; the lowest-validation-loss weights are saved to `output_dir/.../best_checkpoint`, while the last epoch remains in `output_dir/.../final_checkpoint`. SFT evaluation and downstream RL automatically use `best_checkpoint`. Direct/MiniOneRec-RL uses micro-batch 32/accumulation 1 (effective batch 32) on a single 46 GB GPU, so each batch contains two complete 16-candidate GRPO groups; DIPRec-RL remains at micro-batch 1/accumulation 8 (effective batch 8). Both RL trainers automatically set the global generation batch to `per_device_batch_size × world_size × gradient_accumulation_steps`. Both DIPRec-RL methods now use a frozen DIPRec-SFT reference (`beta=1e-3`), cache old-policy log-probabilities, and reuse each rollout twice (`num_iterations=2`), so PPO clipping becomes active after the first update. `diprec_traj_rl` applies each trajectory advantage to both stages; `diprec_plan_rl` keeps plan-across-G and SID-within-B advantages separate.

## 2. Place the official data

Purpose: provide the official split, fixed SIDs, and MiniOneRec title/description alignment data.

```text
data/Amazon/train/{dataset}_5_2016-10-2018-11.csv
data/Amazon/valid/{dataset}_5_2016-10-2018-11.csv
data/Amazon/test/{dataset}_5_2016-10-2018-11.csv
data/Amazon/index/{dataset}.index.json
data/Amazon/index/{dataset}.item.json
```

`{dataset}` is `Video_Games`, `Office_Products`, or `Industrial_and_Scientific`.

## 3. Select and build long-history data

Purpose: select the two longest-history datasets, preserve official train/valid/test targets, and expand each prefix to at most 50 items.

```bash
python scripts/select_long_history_datasets.py \
  --datasets Office_Products,Video_Games,Industrial_and_Scientific \
  --top_n 2 \
  --stats_output outputs/history_length_stats.csv \
  --selection_output configs/selected_long_history_datasets.txt

while read -r DATASET; do
  python scripts/build_long_history_data.py --dataset "$DATASET"
done < configs/selected_long_history_datasets.txt
```

Output: `data/processed/$DATASET/history_50/`.

## 4. Check all seven commands

Purpose: validate data, task counts, checkpoint dependencies, and evaluation paths without GPU training.

```bash
bash scripts/run_all_comparisons.sh --dry_run
```

## 5. Run all seven experiments

Purpose: execute this dependency graph in order.

| Method | Parent checkpoint | Training objective / upstream boundary |
|---|---|---|
| `direct_sft` | Qwen | History SID→next SID supervised baseline |
| `direct_rl` | `direct_sft` | TRL GRPO with MiniOneRec-style ranking reward and constrained beam sampling |
| `minionerec_sft` | Qwen | Four enabled MiniOneRec SFT task families under the shared protocol |
| `minionerec_rl` | `minionerec_sft` | Four enabled MiniOneRec RL task families, `G=16`, frozen reference |
| `diprec_sft` | `minionerec_sft` | Interest-plan SFT plus plan-conditioned SID SFT |
| `diprec_traj_rl` | `diprec_sft` | Trajectory-level hierarchical TRL objective |
| `diprec_plan_rl` | `diprec_sft` | Plan-level plus within-plan SID advantages |

```text
Qwen
├─ direct_sft ───────────────→ direct_rl          (TRL)
└─ minionerec_sft ─┬────────→ minionerec_rl      (TRL)
                    └────────→ diprec_sft
                                  ├─ diprec_traj_rl
                                  └─ diprec_plan_rl
```

```bash
bash scripts/run_all_comparisons.sh
```

Both DIPRec-RL runs start independently from the same `diprec_sft` checkpoint. Outputs are written to:

```text
outputs/$DATASET/history_50/Qwen_Qwen3-0.6B/$METHOD/seed_42/
```

For replicated multi-GPU training, prefix the same commands with the launcher controls below. The wrappers apply `accelerate launch` to SFT and TRL-RL training while keeping preprocessing and evaluation single-process:

```bash
DIPREC_DDP=1 DIPREC_NUM_PROCESSES=4 bash scripts/run_all_comparisons.sh
```

Use an Accelerate configuration with ordinary multi-GPU DDP. Direct-RL and
MiniOneRec-RL follow upstream MiniOneRec's default non-vLLM behavior: every
rank generates its own local prompt groups, after which TRL gathers rewards for
global GRPO normalization. DIPRec's two-stage RL rollout is still centralized
on rank zero. Both custom paths require a complete model replica on every rank,
so trainer initialization rejects DeepSpeed, FSDP, and tensor parallelism.

Before a non-dry-run dependency or evaluation loads model weights, it checks the canonical method, processed-data fingerprint (including the SID index), and (where applicable) item-metadata checksum. DIPRec parent reuse and evaluation additionally check the interest label strategy/top-k/time-decay, conditioning, and parameterization; RL evaluation also checks the training plan/beam shape. A mismatch fails fast instead of silently reusing a stale checkpoint. Result JSON keeps immutable `training_config` separate from the current `evaluation_config`.

## 6. Run one experiment

Run the single-GPU MiniOneRec workflow in this order. Use the same `--run_tag` in both commands so RL uses the SFT checkpoint from this run, rather than an older default checkpoint.

### 1. SFT

The command below uses the recommended configuration from the Video Games sweep: at most 6 epochs,
learning rate `1e-4`, micro-batch 8, and accumulation 4 (effective batch 32).

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh \
  --method minionerec_sft \
  --dataset Video_Games \
  --run_tag sft6e_lr1e-4_best \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 1e-4
```

The terminal continues to print step-level loss for live monitoring. Its concise on-disk
log is updated once per completed epoch; inspect these files before starting RL:

```text
outputs/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e_lr1e-4_best/sft_training_metrics.json
outputs/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e_lr1e-4_best/valid_metrics.json
outputs/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e_lr1e-4_best/metrics.json
```

After the server run, download only the complete
`outputs/.../seed_42_sft6e_lr1e-4_best/` directory for debugging. Large model
weights live separately under `output_dir/.../` and normally stay on the server.

### 2. DIPRec-SFT

DIPRec-SFT continues from the best MiniOneRec-SFT checkpoint with the same run
tag and trains the interest plan plus plan-conditioned SID objectives. If that
parent checkpoint already exists and passes validation, the runner reuses it:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh \
  --method diprec_sft \
  --dataset Office_Products \
  --run_tag sft6e_lr1e-4_best \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 1e-4
```

DIPRec-SFT also validates after every epoch and saves the lowest-validation-loss
weights to:

```text
output_dir/Office_Products/history_50/Qwen_Qwen3-0.6B/diprec_sft/seed_42_sft6e_lr1e-4_best/best_checkpoint
```

The last epoch remains in the adjacent `final_checkpoint`; per-epoch losses are
written to:

```text
outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/diprec_sft/seed_42_sft6e_lr1e-4_best/sft_training_metrics.json
```

Both `diprec_traj_rl` and `diprec_plan_rl` automatically use that
`best_checkpoint`.

#### DIPRec plan-diversity ablation (Office Products)

`single` preserves the original one-label frequency Top-K supervision. `diverse`
deterministically builds up to eight content-distinct plans from level-1 interests
observed in the history prefix only; it never reads the target or counts token
permutations as different plans.
Only the plan-generation task is expanded. SID supervision remains paired with
the legacy frequency Top-K primary plan, avoiding the degenerate signal that
every alternative plan should predict the same target.

```bash
# Single-plan SFT
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh --method diprec_sft --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best --run_tag plan_single \
  --sft_plan_mode single --sft_num_plans 8 --sft_num_epochs 6 \
  --sft_micro_batch_size 8 --sft_gradient_accumulation_steps 4 --sft_learning_rate 1e-4

# Diverse-plan SFT
CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiment.sh --method diprec_sft --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best --run_tag plan_diverse \
  --sft_plan_mode diverse --sft_num_plans 8 --sft_num_epochs 6 \
  --sft_micro_batch_size 8 --sft_gradient_accumulation_steps 4 --sft_learning_rate 1e-4

# RL initialized from single-plan SFT
CUDA_VISIBLE_DEVICES=2 bash scripts/run_experiment.sh --method diprec_plan_rl --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best --diprec_sft_run_tag plan_single \
  --run_tag plan_single_rl --sft_plan_mode single --sft_num_plans 8

# RL initialized from diverse-plan SFT
CUDA_VISIBLE_DEVICES=3 bash scripts/run_experiment.sh --method diprec_plan_rl --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best --diprec_sft_run_tag plan_diverse \
  --run_tag plan_diverse_rl --sft_plan_mode diverse --sft_num_plans 8
```

Both RL runs use the same algorithm and hyperparameters; only their parent SFT
plan supervision differs. Evaluation deduplicates generated plans and reallocates
the fixed 80-candidate SID budget across the unique plans actually returned.

### 3. RL

The 1,000-update **official rewards vs prefix assistance on all-miss recommendation
groups** A/B is complete (2026-09-08). Valid Recall@10/NDCG@10 are
`0.234484/0.189556` for official rewards, `0.234690/0.189771` for prefix λ=0.1,
and `0.235923/0.190004` for SFT. Prefix assistance gains one net Top10 hit but loses
six Top5 hits versus official rewards; paired intervals cross zero. No recommendation
gain is established. Both arms fix the existing SFT, seed42, Qwen3, history50, all
four tasks, sampling, optimizer, and reference.

- `official` (default): exact + rank rewards with the existing group normalization.
- `main_miss_prefix`: add assistance only when a `history_sid_to_sid` group has
  no exact hit. Hit groups and the other three tasks retain their original advantages.
- Define `h = 0.5 × first-level match + 0.5 × first-two-level prefix match`.
  Add `λ × (h - group_mean(h))`, initially `λ=0.1`. Do not divide the auxiliary
  term by group std: its coefficient must control actual strength. Uniform
  prefix scores add no signal. The KL term is unchanged.

The auxiliary signal activates in 17.24% of main-task training groups, raising
nonzero task-advantage coverage across all four tasks from 37.01% to 49.17%.
Its total absolute scalar advantage is only about 0.9% of the official term
(this is not a gradient contribution estimate). Mean KL is about 0.0024 in both
arms, without the earlier large spikes. Signal strength and ranking utility remain
open questions. The earlier SFT Top10 coverage proxy, 23.59%→41.02%, uses different
candidates from actual training G16 rollouts.

The commands below reproduce the completed pilots; use fresh run tags for reruns.
Both arms start from the same existing SFT. The stop callback
preserves the full one-epoch cosine schedule; it does not shorten the scheduler to
1,000 updates. Use `0` or omit the stop option for a full epoch. Both arms evaluate
the checkpoint at the same stopping step, without selecting by RL eval_loss.
`--eval_split valid` evaluates only validation recommendations and defers test evaluation.

```bash
for reward_mode in official main_miss_prefix; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DIPREC_DDP=1 DIPREC_NUM_PROCESSES=4 \
  bash scripts/run_experiment.sh \
    --method minionerec_rl \
    --dataset Office_Products \
    --model Qwen/Qwen3-0.6B --max_history_len 50 --seed 42 \
    --sft_run_tag sft6e_lr1e-4_best \
    --run_tag "rl_prefix_ab_${reward_mode}_s1000_lr2e-6_beta1e-2_4gpu_eb256" \
    --baseline_rl_task_scope official_mixed \
    --baseline_rl_reference_mode fixed \
    --baseline_rl_per_device_batch_size 16 \
    --baseline_rl_gradient_accumulation_steps 4 \
    --baseline_rl_generation_batch_size 256 \
    --baseline_rl_learning_rate 2e-6 \
    --baseline_rl_beta 1e-2 \
    --baseline_rl_num_epochs 1 \
    --baseline_rl_reward_mode "$reward_mode" \
    --baseline_rl_prefix_reward_strength 0.1 \
    --baseline_rl_stop_after_steps 1000 \
    --baseline_rl_diagnostics \
    --eval_split valid || break
done
```

The four-GPU configuration gives `4 × 16 × 4 = 256` candidates/update, or 16 G16
prompt groups. Each rank generates 64 candidates (4 groups). Prefix advantages are
computed before TRL shuffles and splits microbatches. If batch settings need changing,
change both arms and use fresh run tags.

SFT parent:

```text
output_dir/Office_Products/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e_lr1e-4_best/best_checkpoint
```

The two completed run tags start with `rl_prefix_ab_official_...` and
`rl_prefix_ab_main_miss_prefix_...`. Saved training metadata includes reward mode,
strength, completed updates and scheduler total steps. Historical defaults remain
official rewards, LR1e-5, beta1e-3, and two epochs; the command explicitly selects
the conservative configuration. Direct `train_baseline_grpo.py` controls are
`--reward_mode`, `--prefix_reward_strength`, `--stop_after_steps`, and
`--diagnostics_file`; both public experiment runners forward these options.

`--baseline_rl_diagnostics` additionally writes train/eval entries to
`rl_diagnostics.jsonl`. Per-task `prefix_aux/<task>/exact_hit`, `eligible`,
`prefix_informative`, and `aux_active` report fractions of all groups of that task:
exact hits, eligible all-miss recommendation groups, eligible groups with distinct
prefix scores, and groups receiving nonzero auxiliary advantages. `prefix_mean`,
`aux_abs_mean`, and `aux_abs_max` report signal magnitudes. The baseline records
prefix diagnostics too, with `aux_active=0`. Existing `reward` and
`frac_reward_zero_std` still describe only exact+rank. Logging defaults to every
step; with larger log intervals, aggregated diagnostics average batch statistics.

The serial sweep is complete: **official, λ=0.3, λ=0.5, λ=1.0**, each trained for
1,000 updates from the same SFT with all other settings fixed. Valid R@10/NDCG@10
are `0.234484/0.189556`, `0.234073/0.189279`, `0.234279/0.189619`, and
`0.234895/0.190032`. At λ=1.0, NDCG@10 exceeds SFT by only 0.000027, while Top10
and Top5 lose five and thirteen hits. Paired intervals cross zero: no reliable gain
is established. The rerun official training records and validation predictions match
the previous official run exactly; λ=0.1 remains a historical reference.

The auxiliary advantage at λ=1.0 is about ten times its λ=0.1 value, with stable KL,
but recommendation coverage does not clearly improve. Stop scanning prefix strength;
the next five-arm sweep below tests reference mode, KL strength and learning rate
with official exact+rank rewards and no vector-similarity experiments. Commands below remain available for
reproduction; use a new tag for new training, and the completed tag for summaries.

```bash
# Optional independent rerun from DIPRec root; needs training environment, four GPUs and existing SFT
python3 scripts/run_prefix_sweep.py --sweep_tag prefix_strength_rerun --gpus 0,1,2,3

# Print commands without writing files or loading models
python3 scripts/run_prefix_sweep.py --sweep_tag prefix_strength_v1 --dry_run

# Skip verified complete arms; retry incomplete arms from SFT under fresh attempt tags
python3 scripts/run_prefix_sweep.py --sweep_tag prefix_strength_v1 --gpus 0,1,2,3 --resume

# Rebuild the summary only
python3 scripts/run_prefix_sweep.py --sweep_tag prefix_strength_v1 --summarize_only
```

For three arms, add `--strengths 0.5 1.0`; official rewards are always included.
Pass the same strengths when resuming. Omitting `--sweep_tag` generates a timestamp;
use a new tag for independent reruns. Each arm finishes training and Valid evaluation
before the next starts. Failure stops the sweep, preserving previous outputs.
The runner's `--require_existing_sft` flag rejects missing/incompatible parents
instead of triggering SFT training.

Console logs, `state.json`, `summary.csv`, and `summary.json` are saved under
`outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/prefix_sweeps/<sweep_tag>/`.
Individual results keep the standard `minionerec_rl/seed_42_rl_<sweep_tag>_...` paths.
Summaries include Valid R/NDCG@5/10, prefix hits, Top10 metric deltas versus SFT and
the new official control, count-weighted auxiliary coverage/magnitude and KL.
Compatible previous A/B and SFT results are labeled reference; unavailable or
incompatible references are reported and never silently substituted or retrained.

Use Valid NDCG@10 first, then Recall@10, Top5 and subsequent paired-hit analysis.
The summary contains point estimates only; it does not automatically select a winner
or extend training based on tiny differences. Prefix-only gains do not establish
success. Do not use test results for tuning. See
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) for the experiment ledger.

### 4. Five RL optimization runs (implemented, not yet run)

Use the existing `sft6e_lr1e-4_best/best_checkpoint`, four GPUs, and the activated
training environment from the DIPRec root. Every arm starts independently from SFT.

| Arm | Reference | Beta | Learning rate | Comparison |
|---|---|---:|---:|---|
| A | fixed | .01 | 2e-6 | Full-epoch control |
| B | sync | .01 | 2e-6 | B−A: reference sync |
| C | fixed | .001 | 2e-6 | C−A: weaker KL |
| D | sync | .001 | 2e-6 | D−C / D−B: sync and KL |
| E | fixed | .01 | 5e-6 | E−A: higher LR |

All use Office Products, Qwen3-0.6B, history50, seed42, four tasks, official
exact+rank rewards, G16, four GPUs with micro16 × accumulation4 (global batch256).
No prefix shaping or vector similarity is used. Sync runs update the reference every
512 optimizer steps with `ref = .4 * ref + .6 * policy`. Each run is one epoch;
the expected 3,455 optimizer steps are checked before the first update.

```bash
python3 scripts/run_rl_optimization_sweep.py --sweep_tag rl_opt_v1 --dry_run
python3 scripts/run_rl_optimization_sweep.py --sweep_tag rl_opt_v1 --gpus 0,1,2,3
python3 scripts/run_rl_optimization_sweep.py --sweep_tag rl_opt_v1 --gpus 0,1,2,3 --resume
python3 scripts/run_rl_optimization_sweep.py --sweep_tag rl_opt_v1 --summarize_only
```

The runner evaluates the initial SFT afresh, then serially trains and evaluates A–E.
Model snapshots at 1,000/2,000 and the final 3,455 checkpoint receive full Valid
80-candidate deterministic beam evaluation after training finishes. The existing
in-training RL validation cadence remains unchanged. The primary endpoint is 3,455;
compare intermediate checkpoints at matching steps, without per-arm cherry-picking.
Snapshots store model/tokenizer only, adding two model copies per arm; they cannot
resume the optimizer or synchronized reference.

Frozen probes use 128 train and 128 valid main-task records by default. Each keeps the
initial SFT's five highest-ranked distinct wrong items (beam80) plus the target,
using the recommendation prompt. Measurements include full-vocabulary SID+EOS logp,
target probability, target-minus-best-negative margin, fixed-candidate rank and
drift from SFT. `candidate_set_kl_sft_to_policy` normalizes only these six candidates:
it is **not full-policy KL**, nor the training KL against a moving sync reference.
Change sample count with `--probe_samples`; use the same value on resume/summary.

Sweep state, frozen probe, baseline and summaries live under
`outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/rl_optimization_sweeps/<tag>/`.
Individual runs use `minionerec_rl/seed_42_rl_<tag>_<A-E>_a<N>/`, with final metrics at
the root and milestone metrics/predictions/probes in `step_1000/step_2000`.
Summaries include 15 arm/stage rows, deltas versus SFT, five matched arm comparisons,
and per-task exact-hit group fractions and KL across three training windows.
They contain point estimates, not significance tests; test data is not evaluated.

A failed evaluation resumes without retraining completed checkpoints. Interrupted
training restarts from SFT in a fresh attempt directory. Missing/incompatible SFT
never triggers SFT training. Input/weight hashes and completion markers guard reuse.
`--summarize_only` needs the relevant lightweight outputs, not model weights.

All RL methods default to `eval_steps=0.1`, running RL validation at roughly
every 10% of total training steps (about ten times over the full run). These
measurements expose validation-loss trends; full Recall/NDCG evaluation still
runs after training. The workflow saves and evaluates `final_checkpoint` rather than selecting a best
checkpoint from noisy RL validation metrics. Override the interval with
`--baseline_rl_eval_steps` or `--diprec_rl_eval_steps`.

After each periodic validation, the main process atomically refreshes a
lightweight file containing only evaluation results such as `eval_loss`,
`eval_runtime`, step, and epoch. Per-step training logs are not persisted there:

```text
outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/<RL-method>/<run_id>/rl_training_metrics.json
```

This file is safe to download while training is still running. Large checkpoints
remain under `output_dir/.../final_checkpoint`; the optimization sweep also saves sibling `step_1000/step_2000` snapshots.

To trim files produced by the older all-events logger, clean the whole output
tree with:

```bash
python3 scripts/clean_rl_training_metrics.py outputs
```

Only files named `rl_training_metrics.json` are rewritten, and the original is
kept beside each file as `rl_training_metrics.json.bak` by default. Add
`--dry-run` to preview or `--no-backup` to skip backups. A single JSON file path
can be passed instead of `outputs`.

Supported methods:

```text
direct_sft
direct_rl
minionerec_sft
minionerec_rl
diprec_sft
diprec_traj_rl
diprec_plan_rl
```

## 7. Summarize results

Purpose: combine every `metrics.json` into one CSV.

```bash
python scripts/summarize_results.py \
  --input outputs/ \
  --output outputs/comparison.csv
```

## 8. Multiple seeds and ablations

Purpose: add seeds or explicitly run leave-last-two-out.

```bash
bash scripts/run_all_comparisons.sh --seeds 42,43,44

bash scripts/run_all_comparisons.sh \
  --split_strategy leave_last_two_out \
  --seeds 42
```

Common options: `--max_history_len 10|20|50`, `--model Qwen/Qwen3-1.7B`, and `--conditioning history_visible|interest_bottleneck`. Use `--run_tag sft6e_lr1e-4_best` to keep a retrain separate from existing checkpoints; SFT refuses to overwrite either `best_checkpoint` or `final_checkpoint`. SFT controls are `--sft_num_epochs`, `--sft_micro_batch_size`, `--sft_gradient_accumulation_steps`, `--sft_learning_rate`, `--sft_weight_decay`, and `--sft_warmup_ratio`.

RL batch controls are `--baseline_rl_per_device_batch_size`, `--baseline_rl_generation_batch_size`, and `--baseline_rl_gradient_accumulation_steps`; optimization controls are `--baseline_rl_learning_rate`, `--baseline_rl_beta`, and `--baseline_rl_num_epochs`, plus the analogous `--diprec_rl_*` options. Leave generation batch unset to derive it safely. If set explicitly, it must contain complete `num_generations`/`num_plans` groups and equal the global effective update batch. For Direct/MiniOneRec-RL, both the global generation batch and its per-rank share must be divisible by `num_generations`; this ensures every rank owns complete GRPO groups. TRL consequently derives `steps_per_generation = gradient_accumulation_steps`. Internally, its sampler uses `repeat_count = num_iterations × steps_per_generation`: the `steps_per_generation` factor feeds all micro-step slices, while `num_iterations` determines how many optimizer updates reuse the rollout. Keep DIPRec `num_iterations >= 2`, so the same rollout drives two optimizer updates and PPO clipping is active on the reused update.

## 9. Run checks

Purpose: run lightweight regressions and syntax checks.

```bash
python -m unittest discover -s tests -v
python -m compileall -q diprec scripts tests
```

The fixed TRL 0.24.0 compatibility stack passes all 79 discovered tests. Separately, both complete two-process CPU DDP lifecycle tests pass, as do shell syntax, Python compilation, whitespace checks, and the `Video_Games`/`Office_Products` × seven-method dry-run. CUDA kernels, GPU memory limits, and full real-data optimization remain machine-dependent and must be validated on the remote training host.
