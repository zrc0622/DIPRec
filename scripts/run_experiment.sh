#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

METHOD=""
MODEL="Qwen/Qwen3-0.6B"
DATASET=""
RAW_PATH=""
SID_INDEX=""
INTEREST_TOPK=3
NUM_PLANS=8
SID_BEAMS=8
EVAL_BEAMS=10
EVAL_CANDIDATE_BUDGET=80
MAX_HISTORY_LEN=50
MAX_SEQ_LEN=2048
SEED=42
CONDITIONING="interest_bottleneck"
INTEREST_PARAMETERIZATION="independent_head"
INTEREST_STRATEGY="frequency"
TIME_DECAY=0.1
SKIP_PREPROCESS=0
DRY_RUN=0

usage() {
  echo "Usage: $0 --method METHOD --dataset DATASET [--model MODEL] [options]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --raw_path) RAW_PATH="$2"; shift 2 ;;
    --sid_index) SID_INDEX="$2"; shift 2 ;;
    --interest_topk) INTEREST_TOPK="$2"; shift 2 ;;
    --num_plans) NUM_PLANS="$2"; shift 2 ;;
    --sid_beams) SID_BEAMS="$2"; shift 2 ;;
    --eval_beams) EVAL_BEAMS="$2"; shift 2 ;;
    --eval_candidate_budget) EVAL_CANDIDATE_BUDGET="$2"; shift 2 ;;
    --max_history_len) MAX_HISTORY_LEN="$2"; shift 2 ;;
    --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --conditioning) CONDITIONING="$2"; shift 2 ;;
    --interest_parameterization) INTEREST_PARAMETERIZATION="$2"; shift 2 ;;
    --interest_strategy) INTEREST_STRATEGY="$2"; shift 2 ;;
    --time_decay) TIME_DECAY="$2"; shift 2 ;;
    --skip_preprocess) SKIP_PREPROCESS=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

: "${METHOD:?--method is required}"
: "${DATASET:?--dataset is required}"

case "$DATASET" in
  Games) DATASET="Video_Games" ;;
  Office) DATASET="Office_Products" ;;
  Industrial) DATASET="Industrial_and_Scientific" ;;
esac
case "$METHOD" in
  direct_sid|sidreasoner|diprec_sft|diprec_trajectory_grpo|diprec_plan_grpo) ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 2 ;;
esac
case "$MAX_HISTORY_LEN" in 10|20|50) ;; *) echo "--max_history_len must be 10, 20, or 50" >&2; exit 2 ;; esac
for value in "$NUM_PLANS" "$SID_BEAMS" "$EVAL_BEAMS" "$EVAL_CANDIDATE_BUDGET"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Plan/beam counts must be positive integers, got: $value" >&2
    exit 2
  fi
done
if (( EVAL_CANDIDATE_BUDGET < EVAL_BEAMS )); then
  echo "--eval_candidate_budget must be at least --eval_beams" >&2
  exit 2
fi
case "$METHOD" in
  diprec_sft|diprec_trajectory_grpo|diprec_plan_grpo)
    if (( EVAL_CANDIDATE_BUDGET < NUM_PLANS )); then
      echo "--eval_candidate_budget must be at least --num_plans for DIPRec" >&2
      exit 2
    fi
    ;;
esac

DATA_DIR="data/processed/$DATASET/history_$MAX_HISTORY_LEN"
MODEL_SLUG="${MODEL//\//_}"
MODEL_SLUG="${MODEL_SLUG// /_}"
RUN_DIR="outputs/$DATASET/history_$MAX_HISTORY_LEN/$MODEL_SLUG/$METHOD/seed_$SEED"
MODEL_DIR="output_dir/$DATASET/history_$MAX_HISTORY_LEN/$MODEL_SLUG/$METHOD/seed_$SEED"
mkdir -p "$RUN_DIR" "$MODEL_DIR"

if [[ -z "$SID_INDEX" ]]; then
  for candidate in \
    "data/Amazon/index/$DATASET.index.json" \
    "data/Amazon/$DATASET/$DATASET.index.json" \
    "data/Amazon_Games/$DATASET/$DATASET.index.json" \
    "data/Amazon_Office/$DATASET/$DATASET.index.json" \
    "data/Amazon_Industrial/$DATASET/$DATASET.index.json"; do
    if [[ -f "$candidate" ]]; then SID_INDEX="$candidate"; break; fi
  done
fi
if [[ -z "$SID_INDEX" && "$DRY_RUN" -eq 1 ]]; then
  SID_INDEX="data/Amazon/index/$DATASET.index.json"
fi
: "${SID_INDEX:?Cannot resolve SID index; pass --sid_index PATH}"

if [[ "$SKIP_PREPROCESS" -eq 0 ]]; then
  BUILD=(python3 scripts/build_long_history_data.py
    --dataset "$DATASET"
    --sid_index "$SID_INDEX"
    --output_dir "$DATA_DIR"
    --max_history_len "$MAX_HISTORY_LEN")
  if [[ -n "$RAW_PATH" ]]; then BUILD+=(--raw_path "$RAW_PATH"); fi
  if [[ -f "$DATA_DIR/manifest.json" ]]; then
    python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); expected=int(sys.argv[2]); assert m.get("source_kind")=="raw_event_interactions" and int(m["max_history_len"])==expected, m' "$DATA_DIR/manifest.json" "$MAX_HISTORY_LEN"
    echo "Using existing validated long-history data: $DATA_DIR"
  elif [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] '; printf '%q ' "${BUILD[@]}"; printf '\n'
  else
    "${BUILD[@]}"
  fi
fi

if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] data validation deferred until $DATA_DIR exists"
  else
    echo "Missing $DATA_DIR/manifest.json after preprocessing" >&2
    exit 1
  fi
fi

BASE_SFT="output_dir/$DATASET/history_$MAX_HISTORY_LEN/$MODEL_SLUG/direct_sid/seed_$SEED/final_checkpoint"
SIDREASONER_SFT="output_dir/$DATASET/history_$MAX_HISTORY_LEN/$MODEL_SLUG/sidreasoner_sft/seed_$SEED/final_checkpoint"
DIPREC_SFT="output_dir/$DATASET/history_$MAX_HISTORY_LEN/$MODEL_SLUG/diprec_sft/seed_$SEED/final_checkpoint"

checkpoint_ready() {
  [[ -f "$1/config.json" && -f "$1/training_config.json" ]] && \
    find "$1" -maxdepth 1 -type f \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .
}

run_sft() {
  local sft_method="$1" source_model="$2" destination="$3"
  local cmd=(bash scripts/train_diprec_sft.sh
    --method "$sft_method"
    --model "$source_model"
    --train_file "$DATA_DIR/train.jsonl"
    --valid_file "$DATA_DIR/valid.jsonl"
    --sid_index "$SID_INDEX"
    --output_dir "$destination"
    --interest_topk "$INTEREST_TOPK"
    --interest_strategy "$INTEREST_STRATEGY"
    --time_decay "$TIME_DECAY"
    --conditioning "$CONDITIONING"
    --interest_parameterization "$INTEREST_PARAMETERIZATION"
    --max_history_len "$MAX_HISTORY_LEN"
    --max_seq_len "$MAX_SEQ_LEN"
    --seed "$SEED")
  if [[ "$DRY_RUN" -eq 1 ]]; then cmd+=(--dry_run); fi
  if [[ "$DRY_RUN" -eq 1 && ! -f "$DATA_DIR/train.jsonl" ]]; then
    printf '[dry-run] '; printf '%q ' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
}

TRAINED_MODEL=""
case "$METHOD" in
  direct_sid)
    run_sft direct_sid "$MODEL" "$MODEL_DIR/final_checkpoint"
    TRAINED_MODEL="$MODEL_DIR/final_checkpoint"
    ;;
  sidreasoner)
    if ! checkpoint_ready "$BASE_SFT"; then
      if [[ -e "$BASE_SFT" && "$DRY_RUN" -eq 0 ]]; then
        echo "Incomplete Direct-SID checkpoint at $BASE_SFT; remove or relocate it before retraining" >&2
        exit 1
      fi
      run_sft direct_sid "$MODEL" "$BASE_SFT"
    fi
    run_sft sidreasoner_sft "$BASE_SFT" "$SIDREASONER_SFT"
    SIDREASONER_CMD=(bash scripts/train_sidreasoner_grpo.sh \
      --model "$SIDREASONER_SFT" \
      --dataset "$DATASET" \
      --data_dir "$DATA_DIR" \
      --sid_index "$SID_INDEX" \
      --output_dir "$MODEL_DIR/verl" \
      --max_seq_len "$MAX_SEQ_LEN" \
      --num_generations "$NUM_PLANS" \
      --seed "$SEED")
    if [[ "$DRY_RUN" -eq 1 ]]; then SIDREASONER_CMD+=(--dry_run); fi
    "${SIDREASONER_CMD[@]}"
    TRAINED_MODEL="$MODEL_DIR/verl/final_checkpoint"
    ;;
  diprec_sft)
    run_sft diprec_sft "$MODEL" "$MODEL_DIR/final_checkpoint"
    TRAINED_MODEL="$MODEL_DIR/final_checkpoint"
    ;;
  diprec_trajectory_grpo|diprec_plan_grpo)
    if ! checkpoint_ready "$DIPREC_SFT"; then
      if [[ -e "$DIPREC_SFT" && "$DRY_RUN" -eq 0 ]]; then
        echo "Incomplete DIPRec-SFT checkpoint at $DIPREC_SFT; remove or relocate it before retraining" >&2
        exit 1
      fi
      run_sft diprec_sft "$MODEL" "$DIPREC_SFT"
    fi
    MODE="${METHOD#diprec_}"
    GRPO_CMD=(bash scripts/train_diprec_grpo.sh
      --mode "$MODE"
      --model "$DIPREC_SFT"
      --train_file "$DATA_DIR/train.jsonl"
      --sid_index "$SID_INDEX"
      --output_dir "$MODEL_DIR/final_checkpoint"
      --interest_topk "$INTEREST_TOPK"
      --num_plans "$NUM_PLANS"
      --sid_beams "$SID_BEAMS"
      --conditioning "$CONDITIONING"
      --interest_parameterization "$INTEREST_PARAMETERIZATION"
      --max_history_len "$MAX_HISTORY_LEN"
      --max_seq_len "$MAX_SEQ_LEN"
      --seed "$SEED")
    if [[ "$DRY_RUN" -eq 1 ]]; then GRPO_CMD+=(--dry_run); fi
    if [[ "$DRY_RUN" -eq 1 && ! -f "$DATA_DIR/train.jsonl" ]]; then
      printf '[dry-run] '; printf '%q ' "${GRPO_CMD[@]}"; printf '\n'
    else
      "${GRPO_CMD[@]}"
    fi
    TRAINED_MODEL="$MODEL_DIR/final_checkpoint"
    ;;
esac

EVAL_CMD=(bash scripts/eval_diprec.sh
  --method "$METHOD"
  --model "$TRAINED_MODEL"
  --base_model "$MODEL"
  --test_file "$DATA_DIR/test.jsonl"
  --sid_index "$SID_INDEX"
  --output "$RUN_DIR/metrics.json"
  --split test
  --interest_topk "$INTEREST_TOPK"
  --num_plans "$NUM_PLANS"
  --sid_beams "$SID_BEAMS"
  --eval_beams "$EVAL_BEAMS"
  --eval_candidate_budget "$EVAL_CANDIDATE_BUDGET"
  --conditioning "$CONDITIONING"
  --interest_parameterization "$INTEREST_PARAMETERIZATION"
  --max_history_len "$MAX_HISTORY_LEN"
  --max_seq_len "$MAX_SEQ_LEN"
  --seed "$SEED")
if [[ "$DRY_RUN" -eq 1 ]]; then EVAL_CMD+=(--dry_run); fi
if [[ "$DRY_RUN" -eq 1 && ! -f "$DATA_DIR/test.jsonl" ]]; then
  printf '[dry-run] '; printf '%q ' "${EVAL_CMD[@]}"; printf '\n'
else
  VALID_CMD=("${EVAL_CMD[@]}")
  for index in "${!VALID_CMD[@]}"; do
    case "${VALID_CMD[$index]}" in
      "$DATA_DIR/test.jsonl") VALID_CMD[$index]="$DATA_DIR/valid.jsonl" ;;
      "$RUN_DIR/metrics.json") VALID_CMD[$index]="$RUN_DIR/valid_metrics.json" ;;
      test) VALID_CMD[$index]=valid ;;
    esac
  done
  "${VALID_CMD[@]}"
  "${EVAL_CMD[@]}"
fi
