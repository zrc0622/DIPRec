# DIPRec remote reproduction

This add-on keeps the original SIDReasoner files intact. Run commands from the repository root. Use Linux, Python 3.10, and a CUDA host for training; Qwen3/VeRL/vLLM must support the installed CUDA/PyTorch versions.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-diprec.txt
```

For multi-GPU SIDReasoner GRPO, use a VeRL-compatible CUDA image as in the original README and set `NUM_GPUS`. Supply local data yourself; Hugging Face resolves the configured model from its cache or downloads it on the remote host.
The vendored VeRL tree was built around the Torch 2.4–2.6 / vLLM 0.8 stack; install FlashAttention/FlashInfer versions matching the CUDA image before RL if the image does not already provide them.

## Data and long-history selection

Place event-level, untruncated Amazon interactions at `data/Amazon/raw/{Office_Products,Video_Games,Industrial_and_Scientific}.jsonl[.gz]`. Each row needs a user (`user_id`/`reviewerID`), item (`item_id`/`asin`) and time (`timestamp`/`unixReviewTime`). Put existing SID maps at `data/Amazon/index/{dataset}.index.json`. Legacy CSVs containing `history_item_*` are rejected as raw input. SIDReasoner's paper/scripts sometimes shorten `Video_Games` to `Games`; the new launchers accept either spelling and canonicalize both to `Video_Games` (not `Toys_and_Games`).

```bash
python scripts/select_long_history_datasets.py \
  --datasets Office_Products,Video_Games,Industrial_and_Scientific \
  --top_n 2 \
  --stats_output outputs/history_length_stats.csv \
  --selection_output configs/selected_long_history_datasets.txt

DATASET="$(sed -n '1p' configs/selected_long_history_datasets.txt)"

python scripts/build_long_history_data.py \
  --dataset "$DATASET" \
  --max_history_len 50
```

The builder uses per-user chronological leave-last-two-out, forms every target from its preceding prefix, then retains the latest 50 items. It writes `train/valid/test.jsonl`, `manifest.json` with source checksums, and before/after distributions under `data/processed/$DATASET/history_50/`. Configs for 10/20/50 ablations are in `configs/`; rebuild data for each cap.

When launching a method repeatedly, preprocessing reuses an existing manifest rather than overwriting it. Dependency checkpoints are reused only when model weights, `config.json`, and `training_config.json` are all present; an incomplete directory fails early. Remove or relocate intentionally stale data/checkpoint directories yourself before rebuilding or retraining.

## Single-dataset comparison (seed 42)

Each command preprocesses if needed, trains, evaluates with the same catalog trie, the same raw search budget of 80 SID candidates, and the same final top-10 cutoff. DIPRec divides that fixed budget across its interest plans and globally reranks the resulting trajectories; this avoids giving it `num_plans` times more candidate exploration than the baselines. Outputs are written below `outputs/$DATASET/history_50/Qwen_Qwen3-0.6B/$METHOD/seed_42/`. Override the common budget with `--eval_candidate_budget`; it must be at least `--num_plans` for DIPRec.

```bash
# 1. Direct-SID
bash scripts/run_experiment.sh --method direct_sid --model Qwen/Qwen3-0.6B --dataset "$DATASET" --max_history_len 50 --max_seq_len 2048 --seed 42

# 2. SIDReasoner: shared-data natural-language reasoning SFT + original VeRL GRPO
bash scripts/run_experiment.sh --method sidreasoner --model Qwen/Qwen3-0.6B --dataset "$DATASET" --max_history_len 50 --max_seq_len 2048 --seed 42

# 3. DIPRec SFT
bash scripts/run_experiment.sh --method diprec_sft --model Qwen/Qwen3-0.6B --dataset "$DATASET" --interest_topk 3 --max_history_len 50 --max_seq_len 2048 --seed 42

# 4. DIPRec trajectory-level GRPO
bash scripts/run_experiment.sh --method diprec_trajectory_grpo --model Qwen/Qwen3-0.6B --dataset "$DATASET" --interest_topk 3 --num_plans 8 --sid_beams 8 --max_history_len 50 --max_seq_len 2048 --seed 42

# 5. DIPRec plan-level GRPO
bash scripts/run_experiment.sh --method diprec_plan_grpo --model Qwen/Qwen3-0.6B --dataset "$DATASET" --interest_topk 3 --num_plans 8 --sid_beams 8 --max_history_len 50 --max_seq_len 2048 --seed 42
```

`independent_head` is the default: interest inputs/logits use a separate embedding and output head. `--interest_parameterization disjoint_rows` uses non-overlapping rows in the shared Qwen embedding/head and is the vLLM-compatible fallback. `--conditioning interest_bottleneck` starts a fresh SID pass containing only the plan; `history_visible` is the ablation.

For split fairness, the unified `sidreasoner` branch derives deterministic natural-language rationales from each prefix instead of mixing in the released 10-item narrative corpus. It preserves SIDReasoner's reasoning→SID SFT/VeRL-GRPO shape, but it is not a bit-for-bit reproduction of the paper checkpoint; the untouched original scripts remain available for that purpose.

## Batch comparison and result summary

The dataset file must contain only the selected one or two datasets.

```bash
bash scripts/run_all_comparisons.sh --model Qwen/Qwen3-0.6B --dataset_file configs/selected_long_history_datasets.txt --seeds 42 --max_history_len 50 --max_seq_len 2048 --eval_beams 10 --eval_candidate_budget 80
python scripts/summarize_results.py --input outputs/ --output outputs/comparison.csv

# Run only after seed-42 DIPRec improves over baselines:
bash scripts/run_all_comparisons.sh --model Qwen/Qwen3-0.6B --dataset_file configs/selected_long_history_datasets.txt --seeds 42,43,44 --max_history_len 50 --max_seq_len 2048 --eval_beams 10 --eval_candidate_budget 80
```

`metrics.json` and `comparison.csv` contain Recall@5/10, NDCG@5/10, SID valid rate, interest diversity, hierarchical hits, evaluation budget, config, and data hashes. Re-evaluate a checkpoint with `bash scripts/eval_diprec.sh --help` options. Use `Qwen/Qwen3-1.7B` only after the small-model result is positive.

Validation and test use the same evaluator; substitute the actual checkpoint and method:

```bash
bash scripts/eval_diprec.sh --method diprec_plan_grpo --model output_dir/$DATASET/history_50/Qwen_Qwen3-0.6B/diprec_plan_grpo/seed_42/final_checkpoint --test_file data/processed/$DATASET/history_50/valid.jsonl --sid_index data/Amazon/index/$DATASET.index.json --output outputs/$DATASET/history_50/Qwen_Qwen3-0.6B/diprec_plan_grpo/seed_42/valid_metrics.json --split valid --max_history_len 50 --max_seq_len 2048 --interest_topk 3 --num_plans 8 --sid_beams 8 --eval_beams 10 --eval_candidate_budget 80 --seed 42
bash scripts/eval_diprec.sh --method diprec_plan_grpo --model output_dir/$DATASET/history_50/Qwen_Qwen3-0.6B/diprec_plan_grpo/seed_42/final_checkpoint --test_file data/processed/$DATASET/history_50/test.jsonl --sid_index data/Amazon/index/$DATASET.index.json --output outputs/$DATASET/history_50/Qwen_Qwen3-0.6B/diprec_plan_grpo/seed_42/metrics.json --split test --max_history_len 50 --max_seq_len 2048 --interest_topk 3 --num_plans 8 --sid_beams 8 --eval_beams 10 --eval_candidate_budget 80 --seed 42
```

## Checks and unverified items

```bash
python -m unittest discover -s tests -v
python -m compileall -q diprec scripts
```

Locally verified: pure-Python contracts, raw-data selection/building, syntax, and shell parsing. Not locally verified: model/tokenizer download, CUDA kernels, vLLM/VeRL distributed execution, GPU memory/batch tuning, checkpoint merge, or real-data metrics. `independent_head` uses the included HF/Accelerate rollout; use `disjoint_rows` where a stock vLLM engine must load the model directly.
