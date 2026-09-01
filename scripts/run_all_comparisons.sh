#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="Qwen/Qwen3-0.6B"
DATASET_FILE="configs/selected_long_history_datasets.txt"
SEEDS="42"
RUN_TAG=""
MAX_HISTORY_LEN=50
MAX_SEQ_LEN=2048
EVAL_BEAMS=10
EVAL_CANDIDATE_BUDGET=80
DATA_SOURCE="official"
SPLIT_STRATEGY="official_temporal"
INTEREST_TOPK=3
NUM_PLANS=8
SID_BEAMS=8
CONDITIONING="interest_bottleneck"
INTEREST_PARAMETERIZATION="independent_head"
INTEREST_STRATEGY="frequency"
TIME_DECAY=0.1
SFT_NUM_EPOCHS=10
SFT_MICRO_BATCH_SIZE=4
SFT_GRADIENT_ACCUMULATION_STEPS=8
SFT_LEARNING_RATE=5e-5
SFT_WEIGHT_DECAY=0.01
SFT_WARMUP_RATIO=0.03
BASELINE_RL_PER_DEVICE_BATCH_SIZE=1
BASELINE_RL_GENERATION_BATCH_SIZE=""
BASELINE_RL_GRADIENT_ACCUMULATION_STEPS=16
DIPREC_RL_PER_DEVICE_BATCH_SIZE=1
DIPREC_RL_GENERATION_BATCH_SIZE=""
DIPREC_RL_GRADIENT_ACCUMULATION_STEPS=8
DIPREC_RL_NUM_ITERATIONS=2
DIPREC_RL_BETA=0.001
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --dataset_file) DATASET_FILE="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --run_tag) RUN_TAG="$2"; shift 2 ;;
    --max_history_len) MAX_HISTORY_LEN="$2"; shift 2 ;;
    --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --eval_beams) EVAL_BEAMS="$2"; shift 2 ;;
    --eval_candidate_budget) EVAL_CANDIDATE_BUDGET="$2"; shift 2 ;;
    --data_source) DATA_SOURCE="$2"; shift 2 ;;
    --split_strategy) SPLIT_STRATEGY="$2"; shift 2 ;;
    --interest_topk) INTEREST_TOPK="$2"; shift 2 ;;
    --num_plans) NUM_PLANS="$2"; shift 2 ;;
    --sid_beams) SID_BEAMS="$2"; shift 2 ;;
    --conditioning) CONDITIONING="$2"; shift 2 ;;
    --interest_parameterization) INTEREST_PARAMETERIZATION="$2"; shift 2 ;;
    --interest_strategy) INTEREST_STRATEGY="$2"; shift 2 ;;
    --time_decay) TIME_DECAY="$2"; shift 2 ;;
    --sft_num_epochs) SFT_NUM_EPOCHS="$2"; shift 2 ;;
    --sft_micro_batch_size) SFT_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --sft_gradient_accumulation_steps) SFT_GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --sft_learning_rate) SFT_LEARNING_RATE="$2"; shift 2 ;;
    --sft_weight_decay) SFT_WEIGHT_DECAY="$2"; shift 2 ;;
    --sft_warmup_ratio) SFT_WARMUP_RATIO="$2"; shift 2 ;;
    --baseline_rl_per_device_batch_size) BASELINE_RL_PER_DEVICE_BATCH_SIZE="$2"; shift 2 ;;
    --baseline_rl_generation_batch_size) BASELINE_RL_GENERATION_BATCH_SIZE="$2"; shift 2 ;;
    --baseline_rl_gradient_accumulation_steps) BASELINE_RL_GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --diprec_rl_per_device_batch_size|--diprec_rl_train_batch_size) DIPREC_RL_PER_DEVICE_BATCH_SIZE="$2"; shift 2 ;;
    --diprec_rl_generation_batch_size) DIPREC_RL_GENERATION_BATCH_SIZE="$2"; shift 2 ;;
    --diprec_rl_gradient_accumulation_steps) DIPREC_RL_GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --diprec_rl_num_iterations) DIPREC_RL_NUM_ITERATIONS="$2"; shift 2 ;;
    --diprec_rl_beta) DIPREC_RL_BETA="$2"; shift 2 ;;
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
METHODS=(direct_sft direct_rl minionerec_sft minionerec_rl diprec_sft diprec_traj_rl diprec_plan_rl)
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for method in "${METHODS[@]}"; do
      args=(bash scripts/run_experiment.sh
        --method "$method"
        --model "$MODEL"
        --dataset "$dataset"
        --data_source "$DATA_SOURCE"
        --split_strategy "$SPLIT_STRATEGY"
        --interest_topk "$INTEREST_TOPK"
        --num_plans "$NUM_PLANS"
        --sid_beams "$SID_BEAMS"
        --conditioning "$CONDITIONING"
        --interest_parameterization "$INTEREST_PARAMETERIZATION"
        --interest_strategy "$INTEREST_STRATEGY"
        --time_decay "$TIME_DECAY"
        --eval_beams "$EVAL_BEAMS"
        --eval_candidate_budget "$EVAL_CANDIDATE_BUDGET"
        --max_history_len "$MAX_HISTORY_LEN"
        --max_seq_len "$MAX_SEQ_LEN"
        --run_tag "$RUN_TAG"
        --sft_num_epochs "$SFT_NUM_EPOCHS"
        --sft_micro_batch_size "$SFT_MICRO_BATCH_SIZE"
        --sft_gradient_accumulation_steps "$SFT_GRADIENT_ACCUMULATION_STEPS"
        --sft_learning_rate "$SFT_LEARNING_RATE"
        --sft_weight_decay "$SFT_WEIGHT_DECAY"
        --sft_warmup_ratio "$SFT_WARMUP_RATIO"
        --baseline_rl_per_device_batch_size "$BASELINE_RL_PER_DEVICE_BATCH_SIZE"
        --baseline_rl_gradient_accumulation_steps "$BASELINE_RL_GRADIENT_ACCUMULATION_STEPS"
        --diprec_rl_per_device_batch_size "$DIPREC_RL_PER_DEVICE_BATCH_SIZE"
        --diprec_rl_gradient_accumulation_steps "$DIPREC_RL_GRADIENT_ACCUMULATION_STEPS"
        --diprec_rl_num_iterations "$DIPREC_RL_NUM_ITERATIONS"
        --diprec_rl_beta "$DIPREC_RL_BETA"
        --seed "$seed")
      if [[ -n "$DIPREC_RL_GENERATION_BATCH_SIZE" ]]; then
        args+=(--diprec_rl_generation_batch_size "$DIPREC_RL_GENERATION_BATCH_SIZE")
      fi
      if [[ -n "$BASELINE_RL_GENERATION_BATCH_SIZE" ]]; then
        args+=(--baseline_rl_generation_batch_size "$BASELINE_RL_GENERATION_BATCH_SIZE")
      fi
      if [[ "$DRY_RUN" -eq 1 ]]; then args+=(--dry_run); fi
      "${args[@]}"
    done
  done
done
