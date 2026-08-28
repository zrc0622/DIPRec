#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="Qwen/Qwen3-0.6B"
DATASET_FILE="configs/selected_long_history_datasets.txt"
SEEDS="42"
MAX_HISTORY_LEN=50
MAX_SEQ_LEN=2048
EVAL_BEAMS=10
EVAL_CANDIDATE_BUDGET=80
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --dataset_file) DATASET_FILE="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --max_history_len) MAX_HISTORY_LEN="$2"; shift 2 ;;
    --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --eval_beams) EVAL_BEAMS="$2"; shift 2 ;;
    --eval_candidate_budget) EVAL_CANDIDATE_BUDGET="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

DATASETS=()
while IFS= read -r dataset; do
  [[ -z "$dataset" || "$dataset" =~ ^[[:space:]]*# ]] && continue
  DATASETS+=("$dataset")
done < "$DATASET_FILE"
if [[ ${#DATASETS[@]} -lt 1 || ${#DATASETS[@]} -gt 2 ]]; then
  echo "$DATASET_FILE must contain one or two datasets, found ${#DATASETS[@]}" >&2
  exit 1
fi
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
METHODS=(direct_sid sidreasoner diprec_sft diprec_trajectory_grpo diprec_plan_grpo)
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for method in "${METHODS[@]}"; do
      args=(bash scripts/run_experiment.sh
        --method "$method"
        --model "$MODEL"
        --dataset "$dataset"
        --interest_topk 3
        --num_plans 8
        --sid_beams 8
        --eval_beams "$EVAL_BEAMS"
        --eval_candidate_budget "$EVAL_CANDIDATE_BUDGET"
        --max_history_len "$MAX_HISTORY_LEN"
        --max_seq_len "$MAX_SEQ_LEN"
        --seed "$seed")
      if [[ "$DRY_RUN" -eq 1 ]]; then args+=(--dry_run); fi
      "${args[@]}"
    done
  done
done
