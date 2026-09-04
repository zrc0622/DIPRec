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

Run experiment 3 first: a four-GPU MiniOneRec-RL fixed-reference job on Office
Products using GPUs 0--3. The trainer uses rank-local generation, matching
upstream MiniOneRec's default non-vLLM path. Use micro-batch 32, accumulation 4,
and a global generation/effective batch of 512:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DIPREC_DDP=1 \
DIPREC_NUM_PROCESSES=4 \
bash scripts/run_experiment.sh \
  --method minionerec_rl \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag rl_fixed_ref_4gpu_eb512 \
  --baseline_rl_reference_mode fixed \
  --baseline_rl_per_device_batch_size 32 \
  --baseline_rl_gradient_accumulation_steps 4 \
  --baseline_rl_generation_batch_size 512
```

This gives:

```text
4 GPUs x 32 candidates/GPU x 4 accumulation steps = 512 candidates/update
512 / 16 generations = 32 complete GRPO prompt groups/update
Each rank generates 512 / 4 = 128 candidates = 8 complete groups/rollout
```

Each rank generates only its local 128 candidates (8 prompts). If CUDA reports
allocator fragmentation despite sufficient free memory, optionally prefix the
command with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; it is not
required for rank-local generation.

The job initializes from:

```text
output_dir/Office_Products/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e_lr1e-4_best/best_checkpoint
```

Its checkpoint and metrics use the
`seed_42_rl_fixed_ref_4gpu_eb512` directory and do not overwrite the earlier
single-GPU run.

Only after experiment 3 establishes the large-batch fixed-reference result
should the otherwise matched periodic-sync job be run. Do not launch a second
job concurrently on these four GPUs.

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
remain only under `output_dir/.../final_checkpoint`.

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

RL batch controls: `--baseline_rl_per_device_batch_size`, `--baseline_rl_generation_batch_size`, `--baseline_rl_gradient_accumulation_steps`, plus the analogous `--diprec_rl_*` options. Leave generation batch unset to derive it safely. If set explicitly, it must contain complete `num_generations`/`num_plans` groups and equal the global effective update batch. For Direct/MiniOneRec-RL, both the global generation batch and its per-rank share must be divisible by `num_generations`; this ensures every rank owns complete GRPO groups. TRL consequently derives `steps_per_generation = gradient_accumulation_steps`. Internally, its sampler uses `repeat_count = num_iterations × steps_per_generation`: the `steps_per_generation` factor feeds all micro-step slices, while `num_iterations` determines how many optimizer updates reuse the rollout. Keep DIPRec `num_iterations >= 2`, so the same rollout drives two optimizer updates and PPO clipping is active on the reused update.

## 9. Run checks

Purpose: run lightweight regressions and syntax checks.

```bash
python -m unittest discover -s tests -v
python -m compileall -q diprec scripts tests
```

The fixed TRL 0.24.0 compatibility stack passes all 79 discovered tests. Separately, both complete two-process CPU DDP lifecycle tests pass, as do shell syntax, Python compilation, whitespace checks, and the `Video_Games`/`Office_Products` × seven-method dry-run. CUDA kernels, GPU memory limits, and full real-data optimization remain machine-dependent and must be validated on the remote training host.
