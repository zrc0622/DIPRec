#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

METHOD=""
MODEL="Qwen/Qwen3-0.6B"
DATASET=""
RAW_PATH=""
SID_INDEX=""
ITEM_META=""
DATA_SOURCE="official"
SPLIT_STRATEGY="official_temporal"
INTEREST_TOPK=3
NUM_PLANS=8
SID_BEAMS=8
EVAL_BEAMS=10
EVAL_CANDIDATE_BUDGET=80
MAX_HISTORY_LEN=50
MAX_SEQ_LEN=2048
SEED=42
RUN_TAG=""
SFT_RUN_TAG=""
DIPREC_SFT_RUN_TAG=""
CONDITIONING="interest_bottleneck"
INTEREST_PARAMETERIZATION="independent_head"
INTEREST_STRATEGY="frequency"
TIME_DECAY=0.1
SFT_OBJECTIVE="legacy"
SFT_PLAN_MODE="single"
SFT_NUM_PLANS=8
SFT_NUM_EPOCHS=6
SFT_MICRO_BATCH_SIZE=4
SFT_GRADIENT_ACCUMULATION_STEPS=8
SFT_LEARNING_RATE=5e-5
SFT_WEIGHT_DECAY=0.01
SFT_WARMUP_RATIO=0.03
BASELINE_RL_PER_DEVICE_BATCH_SIZE=32
BASELINE_RL_GENERATION_BATCH_SIZE=""
BASELINE_RL_GRADIENT_ACCUMULATION_STEPS=1
BASELINE_RL_REFERENCE_MODE="fixed"
BASELINE_RL_REF_MODEL_SYNC_STEPS=512
BASELINE_RL_REF_MODEL_MIXUP_ALPHA=0.6
BASELINE_RL_EVAL_STEPS=0.1
DIPREC_RL_PER_DEVICE_BATCH_SIZE=1
DIPREC_RL_GENERATION_BATCH_SIZE=""
DIPREC_RL_GRADIENT_ACCUMULATION_STEPS=8
DIPREC_RL_NUM_ITERATIONS=2
DIPREC_RL_BETA=0.001
DIPREC_RL_EVAL_STEPS=0.1
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
    --item_meta) ITEM_META="$2"; shift 2 ;;
    --data_source) DATA_SOURCE="$2"; shift 2 ;;
    --split_strategy) SPLIT_STRATEGY="$2"; shift 2 ;;
    --interest_topk) INTEREST_TOPK="$2"; shift 2 ;;
    --num_plans) NUM_PLANS="$2"; shift 2 ;;
    --sid_beams) SID_BEAMS="$2"; shift 2 ;;
    --eval_beams) EVAL_BEAMS="$2"; shift 2 ;;
    --eval_candidate_budget) EVAL_CANDIDATE_BUDGET="$2"; shift 2 ;;
    --max_history_len) MAX_HISTORY_LEN="$2"; shift 2 ;;
    --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run_tag) RUN_TAG="$2"; shift 2 ;;
    --sft_run_tag) SFT_RUN_TAG="$2"; shift 2 ;;
    --diprec_sft_run_tag) DIPREC_SFT_RUN_TAG="$2"; shift 2 ;;
    --conditioning) CONDITIONING="$2"; shift 2 ;;
    --interest_parameterization) INTEREST_PARAMETERIZATION="$2"; shift 2 ;;
    --interest_strategy) INTEREST_STRATEGY="$2"; shift 2 ;;
    --time_decay) TIME_DECAY="$2"; shift 2 ;;
    --sft_objective) SFT_OBJECTIVE="$2"; shift 2 ;;
    --sft_plan_mode) SFT_PLAN_MODE="$2"; shift 2 ;;
    --sft_num_plans) SFT_NUM_PLANS="$2"; shift 2 ;;
    --sft_num_epochs) SFT_NUM_EPOCHS="$2"; shift 2 ;;
    --sft_micro_batch_size) SFT_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --sft_gradient_accumulation_steps) SFT_GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --sft_learning_rate) SFT_LEARNING_RATE="$2"; shift 2 ;;
    --sft_weight_decay) SFT_WEIGHT_DECAY="$2"; shift 2 ;;
    --sft_warmup_ratio) SFT_WARMUP_RATIO="$2"; shift 2 ;;
    --baseline_rl_per_device_batch_size) BASELINE_RL_PER_DEVICE_BATCH_SIZE="$2"; shift 2 ;;
    --baseline_rl_generation_batch_size) BASELINE_RL_GENERATION_BATCH_SIZE="$2"; shift 2 ;;
    --baseline_rl_gradient_accumulation_steps) BASELINE_RL_GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --baseline_rl_reference_mode) BASELINE_RL_REFERENCE_MODE="$2"; shift 2 ;;
    --baseline_rl_ref_model_sync_steps) BASELINE_RL_REF_MODEL_SYNC_STEPS="$2"; shift 2 ;;
    --baseline_rl_ref_model_mixup_alpha) BASELINE_RL_REF_MODEL_MIXUP_ALPHA="$2"; shift 2 ;;
    --baseline_rl_eval_steps) BASELINE_RL_EVAL_STEPS="$2"; shift 2 ;;
    --diprec_rl_per_device_batch_size|--diprec_rl_train_batch_size) DIPREC_RL_PER_DEVICE_BATCH_SIZE="$2"; shift 2 ;;
    --diprec_rl_generation_batch_size) DIPREC_RL_GENERATION_BATCH_SIZE="$2"; shift 2 ;;
    --diprec_rl_gradient_accumulation_steps) DIPREC_RL_GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --diprec_rl_num_iterations) DIPREC_RL_NUM_ITERATIONS="$2"; shift 2 ;;
    --diprec_rl_beta) DIPREC_RL_BETA="$2"; shift 2 ;;
    --diprec_rl_eval_steps) DIPREC_RL_EVAL_STEPS="$2"; shift 2 ;;
    --skip_preprocess) SKIP_PREPROCESS=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

: "${METHOD:?--method is required}"
: "${DATASET:?--dataset is required}"
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--run_tag must start with an alphanumeric character and contain only letters, digits, ., _, or -" >&2
  exit 2
fi
if [[ -n "$SFT_RUN_TAG" && ! "$SFT_RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--sft_run_tag must start with an alphanumeric character and contain only letters, digits, ., _, or -" >&2
  exit 2
fi
if [[ -n "$DIPREC_SFT_RUN_TAG" && ! "$DIPREC_SFT_RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--diprec_sft_run_tag must start with an alphanumeric character and contain only letters, digits, ., _, or -" >&2
  exit 2
fi
case "$BASELINE_RL_REFERENCE_MODE" in
  fixed|sync) ;;
  *) echo "--baseline_rl_reference_mode must be fixed or sync" >&2; exit 2 ;;
esac
case "$SFT_PLAN_MODE" in
  single|diverse) ;;
  *) echo "--sft_plan_mode must be single or diverse" >&2; exit 2 ;;
esac
case "$SFT_OBJECTIVE" in
  legacy|interest_activation|joint_interest_activation) ;;
  *) echo "--sft_objective must be legacy, interest_activation, or joint_interest_activation" >&2; exit 2 ;;
esac
if [[ "$SFT_OBJECTIVE" != "legacy" && "$CONDITIONING" != "history_visible" ]]; then
  echo "--sft_objective $SFT_OBJECTIVE requires --conditioning history_visible" >&2
  exit 2
fi
if ! [[ "$SFT_NUM_PLANS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--sft_num_plans must be a positive integer" >&2
  exit 2
fi

case "$DATASET" in
  Games) DATASET="Video_Games" ;;
  Office) DATASET="Office_Products" ;;
  Industrial) DATASET="Industrial_and_Scientific" ;;
esac
case "$METHOD" in
  direct_sid) METHOD="direct_sft" ;;
  diprec_trajectory_grpo) METHOD="diprec_traj_rl" ;;
  diprec_plan_grpo) METHOD="diprec_plan_rl" ;;
esac
case "$METHOD" in
  direct_sft|direct_rl|minionerec_sft|minionerec_rl|diprec_sft|diprec_traj_rl|diprec_plan_rl) ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 2 ;;
esac
if [[ "$SFT_OBJECTIVE" == "joint_interest_activation" && ( "$METHOD" == "diprec_traj_rl" || "$METHOD" == "diprec_plan_rl" ) ]]; then
  echo "--sft_objective joint_interest_activation currently supports diprec_sft only; joint-trajectory RL will be added separately" >&2
  exit 2
fi
case "$MAX_HISTORY_LEN" in 10|20|50) ;; *) echo "--max_history_len must be 10, 20, or 50" >&2; exit 2 ;; esac
case "$DATA_SOURCE" in official|raw) ;; *) echo "--data_source must be official or raw" >&2; exit 2 ;; esac
if [[ -n "$RAW_PATH" ]]; then DATA_SOURCE="raw"; fi
case "$SPLIT_STRATEGY" in
  official_temporal|leave_last_two_out) ;;
  *) echo "--split_strategy must be official_temporal or leave_last_two_out" >&2; exit 2 ;;
esac
if [[ "$DATA_SOURCE" == "raw" && "$SPLIT_STRATEGY" != "leave_last_two_out" ]]; then
  echo "--data_source raw requires explicit --split_strategy leave_last_two_out" >&2
  exit 2
fi
for value in "$NUM_PLANS" "$SID_BEAMS" "$EVAL_BEAMS" "$EVAL_CANDIDATE_BUDGET"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Plan/beam counts must be positive integers, got: $value" >&2
    exit 2
  fi
done
for value in \
  "$SFT_NUM_EPOCHS" \
  "$SFT_MICRO_BATCH_SIZE" \
  "$SFT_GRADIENT_ACCUMULATION_STEPS" \
  "$BASELINE_RL_PER_DEVICE_BATCH_SIZE" \
  "$BASELINE_RL_GRADIENT_ACCUMULATION_STEPS" \
  "$DIPREC_RL_PER_DEVICE_BATCH_SIZE" \
  "$DIPREC_RL_GRADIENT_ACCUMULATION_STEPS" \
  "$DIPREC_RL_NUM_ITERATIONS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch settings must be positive integers, got: $value" >&2
    exit 2
  fi
done
if [[ -n "$BASELINE_RL_GENERATION_BATCH_SIZE" && ! "$BASELINE_RL_GENERATION_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Baseline RL generation batch size must be a positive integer" >&2
  exit 2
fi
if [[ -n "$DIPREC_RL_GENERATION_BATCH_SIZE" && ! "$DIPREC_RL_GENERATION_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Batch settings must be positive integers, got: $DIPREC_RL_GENERATION_BATCH_SIZE" >&2
  exit 2
fi
if (( EVAL_CANDIDATE_BUDGET < EVAL_BEAMS )); then
  echo "--eval_candidate_budget must be at least --eval_beams" >&2
  exit 2
fi
case "$METHOD" in
  diprec_sft|diprec_traj_rl|diprec_plan_rl)
    if (( EVAL_CANDIDATE_BUDGET < NUM_PLANS )); then
      echo "--eval_candidate_budget must be at least --num_plans for DIPRec" >&2
      exit 2
    fi
    ;;
esac

DATA_VARIANT="history_$MAX_HISTORY_LEN"
if [[ "$SPLIT_STRATEGY" != "official_temporal" ]]; then
  DATA_VARIANT="${DATA_VARIANT}_${SPLIT_STRATEGY}"
fi
DATA_DIR="data/processed/$DATASET/$DATA_VARIANT"
MODEL_SLUG="${MODEL//\//_}"
MODEL_SLUG="${MODEL_SLUG// /_}"
RUN_ID="seed_$SEED"
if [[ -n "$RUN_TAG" ]]; then RUN_ID+="_$RUN_TAG"; fi
SFT_RUN_ID="seed_$SEED"
if [[ -n "$SFT_RUN_TAG" ]]; then
  SFT_RUN_ID+="_$SFT_RUN_TAG"
elif [[ -n "$RUN_TAG" ]]; then
  SFT_RUN_ID+="_$RUN_TAG"
fi
DIPREC_SFT_RUN_ID="seed_$SEED"
if [[ -n "$DIPREC_SFT_RUN_TAG" ]]; then
  DIPREC_SFT_RUN_ID+="_$DIPREC_SFT_RUN_TAG"
elif [[ -n "$SFT_RUN_TAG" ]]; then
  DIPREC_SFT_RUN_ID+="_$SFT_RUN_TAG"
elif [[ -n "$RUN_TAG" ]]; then
  DIPREC_SFT_RUN_ID+="_$RUN_TAG"
fi
RUN_DIR="outputs/$DATASET/$DATA_VARIANT/$MODEL_SLUG/$METHOD/$RUN_ID"
MODEL_DIR="output_dir/$DATASET/$DATA_VARIANT/$MODEL_SLUG/$METHOD/$RUN_ID"
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

case "$METHOD" in
  minionerec_sft|minionerec_rl|diprec_sft|diprec_traj_rl|diprec_plan_rl)
    if [[ -z "$ITEM_META" ]]; then
      for candidate in \
        "data/Amazon/index/$DATASET.item.json" \
        "data/Amazon/$DATASET/$DATASET.item.json" \
        "data/Amazon_Games/$DATASET/$DATASET.item.json" \
        "data/Amazon_Office/$DATASET/$DATASET.item.json" \
        "data/Amazon_Industrial/$DATASET/$DATASET.item.json"; do
        if [[ -f "$candidate" ]]; then ITEM_META="$candidate"; break; fi
      done
    fi
    if [[ -z "$ITEM_META" && "$DRY_RUN" -eq 1 ]]; then
      ITEM_META="data/Amazon/index/$DATASET.item.json"
    fi
    : "${ITEM_META:?Cannot resolve item metadata; pass --item_meta PATH}"
    ;;
esac

if [[ "$SKIP_PREPROCESS" -eq 0 ]]; then
  BUILD=(python3 scripts/build_long_history_data.py
    --dataset "$DATASET"
    --source "$DATA_SOURCE"
    --split_strategy "$SPLIT_STRATEGY"
    --sid_index "$SID_INDEX"
    --output_dir "$DATA_DIR"
    --max_history_len "$MAX_HISTORY_LEN")
  if [[ -n "$RAW_PATH" ]]; then BUILD+=(--raw_path "$RAW_PATH"); fi
  if [[ -f "$DATA_DIR/manifest.json" ]]; then
    python3 -c 'import json,sys; from diprec.data import validate_processed_manifest; m=json.load(open(sys.argv[1])); source={"official":"sidreasoner_official_csv_reconstruction","raw":"raw_event_interactions"}[sys.argv[4]]; validate_processed_manifest(m,dataset=sys.argv[2],max_history_len=int(sys.argv[3]),source_kind=source,split_strategy=sys.argv[5],sid_index_path=sys.argv[6])' "$DATA_DIR/manifest.json" "$DATASET" "$MAX_HISTORY_LEN" "$DATA_SOURCE" "$SPLIT_STRATEGY" "$SID_INDEX"
    echo "Using existing validated long-history data: $DATA_DIR (source=$DATA_SOURCE, split_strategy=$SPLIT_STRATEGY)"
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

DIRECT_SFT="output_dir/$DATASET/$DATA_VARIANT/$MODEL_SLUG/direct_sft/$SFT_RUN_ID/best_checkpoint"
MINIONEREC_SFT="output_dir/$DATASET/$DATA_VARIANT/$MODEL_SLUG/minionerec_sft/$SFT_RUN_ID/best_checkpoint"
DIPREC_SFT="output_dir/$DATASET/$DATA_VARIANT/$MODEL_SLUG/diprec_sft/$DIPREC_SFT_RUN_ID/best_checkpoint"

checkpoint_ready() {
  local checkpoint="$1" expected_method="$2" expected_parent="$3" expected_item_meta="${4:-}"
  [[ -f "$checkpoint/config.json" && -f "$checkpoint/training_config.json" ]] || return 1
  find "$checkpoint" -maxdepth 1 -type f \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q . || return 1
  if [[ "$expected_method" == "diprec_sft" && "$INTEREST_PARAMETERIZATION" == "independent_head" ]]; then
    [[ -f "$checkpoint/diprec_adapter_config.json" && -f "$checkpoint/diprec_interest_adapter.pt" ]] || return 1
  fi
  python3 -c 'import json,sys; from diprec.data import processed_data_fingerprint,sha256_file; training=json.load(open(sys.argv[1])); manifest=json.load(open(sys.argv[2])); item=sys.argv[5]; item_ok=training.get("item_meta_sha256")==sha256_file(item) if item else training.get("item_meta_sha256") is None; expected={}; expected.update({"interest_topk":int(sys.argv[6]),"interest_strategy":sys.argv[7],"time_decay":float(sys.argv[8]),"conditioning":sys.argv[9],"interest_parameterization":sys.argv[10],"sft_plan_mode":sys.argv[11],"sft_num_plans":int(sys.argv[12]),"sft_objective":sys.argv[13]}) if sys.argv[3]=="diprec_sft" else None; defaults={"sft_objective":"legacy","sft_plan_mode":"single","sft_num_plans":8}; config_ok=all(training.get(k,defaults.get(k))==v for k,v in expected.items()); ok=training.get("checkpoint_role")=="best_validation" and training.get("data_manifest")==processed_data_fingerprint(manifest) and training.get("method")==sys.argv[3] and training.get("model")==sys.argv[4] and item_ok and config_ok; raise SystemExit(0 if ok else 1)' "$checkpoint/training_config.json" "$DATA_DIR/manifest.json" "$expected_method" "$expected_parent" "$expected_item_meta" "$INTEREST_TOPK" "$INTEREST_STRATEGY" "$TIME_DECAY" "$CONDITIONING" "$INTEREST_PARAMETERIZATION" "$SFT_PLAN_MODE" "$SFT_NUM_PLANS" "$SFT_OBJECTIVE"
}

run_sft() {
  local sft_method="$1" source_model="$2" destination="$3"
  local sft_objective="legacy"
  if [[ "$sft_method" == "diprec_sft" ]]; then
    sft_objective="$SFT_OBJECTIVE"
  fi
  local best_destination="$(dirname "$destination")/best_checkpoint"
  local sft_checkpoint_run_id="$(basename "$(dirname "$destination")")"
  local sft_debug_dir="outputs/$DATASET/$DATA_VARIANT/$MODEL_SLUG/$sft_method/$sft_checkpoint_run_id"
  mkdir -p "$sft_debug_dir"
  if [[ "$DRY_RUN" -eq 0 && ( -e "$destination" || -e "$best_destination" ) ]]; then
    echo "Refusing to overwrite existing SFT checkpoints under $(dirname "$destination"); choose a new --run_tag or relocate them first" >&2
    exit 1
  fi
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
    --sft_objective "$sft_objective"
    --sft_plan_mode "$SFT_PLAN_MODE"
    --sft_num_plans "$SFT_NUM_PLANS"
    --conditioning "$CONDITIONING"
    --interest_parameterization "$INTEREST_PARAMETERIZATION"
    --max_history_len "$MAX_HISTORY_LEN"
    --max_seq_len "$MAX_SEQ_LEN"
    --num_epochs "$SFT_NUM_EPOCHS"
    --micro_batch_size "$SFT_MICRO_BATCH_SIZE"
    --gradient_accumulation_steps "$SFT_GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$SFT_LEARNING_RATE"
    --weight_decay "$SFT_WEIGHT_DECAY"
    --warmup_ratio "$SFT_WARMUP_RATIO"
    --training_metrics_file "$sft_debug_dir/sft_training_metrics.json"
    --best_output_dir "$best_destination"
    --seed "$SEED")
  if [[ "$sft_method" == "minionerec_sft" || "$sft_method" == "diprec_sft" ]]; then
    cmd+=(--item_meta "$ITEM_META")
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then cmd+=(--dry_run); fi
  if [[ "$DRY_RUN" -eq 1 && ! -f "$DATA_DIR/train.jsonl" ]]; then
    printf '[dry-run] '; printf '%q ' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
}

ensure_sft() {
  local sft_method="$1" source_model="$2" destination="$3" expected_item_meta="${4:-}"
  local final_destination="$(dirname "$destination")/final_checkpoint"
  if checkpoint_ready "$destination" "$sft_method" "$source_model" "$expected_item_meta"; then
    echo "Using existing validated $sft_method checkpoint: $destination"
    return
  fi
  if [[ "$DRY_RUN" -eq 0 && ( -e "$destination" || -e "$final_destination" ) ]]; then
    echo "Incomplete, incompatible, or legacy $sft_method checkpoints under $(dirname "$destination"); choose a new --run_tag or relocate them before retraining" >&2
    exit 1
  fi
  run_sft "$sft_method" "$source_model" "$final_destination"
}

run_baseline_rl() {
  local rl_method="$1" source_model="$2" destination="$3"
  local cmd=(bash scripts/train_baseline_grpo.sh
    --method "$rl_method"
    --model "$source_model"
    --train_file "$DATA_DIR/train.jsonl"
    --valid_file "$DATA_DIR/valid.jsonl"
    --sid_index "$SID_INDEX"
    --output_dir "$destination"
    --training_metrics_file "$RUN_DIR/rl_training_metrics.json"
    --num_generations 16
    --per_device_batch_size "$BASELINE_RL_PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$BASELINE_RL_GRADIENT_ACCUMULATION_STEPS"
    --reference_mode "$BASELINE_RL_REFERENCE_MODE"
    --ref_model_sync_steps "$BASELINE_RL_REF_MODEL_SYNC_STEPS"
    --ref_model_mixup_alpha "$BASELINE_RL_REF_MODEL_MIXUP_ALPHA"
    --eval_steps "$BASELINE_RL_EVAL_STEPS"
    --max_history_len "$MAX_HISTORY_LEN"
    --max_seq_len "$MAX_SEQ_LEN"
    --seed "$SEED")
  if [[ -n "$BASELINE_RL_GENERATION_BATCH_SIZE" ]]; then
    cmd+=(--generation_batch_size "$BASELINE_RL_GENERATION_BATCH_SIZE")
  fi
  if [[ "$rl_method" == "minionerec_rl" ]]; then
    cmd+=(--item_meta "$ITEM_META")
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then cmd+=(--dry_run); fi
  if [[ "$DRY_RUN" -eq 1 && ! -f "$DATA_DIR/train.jsonl" ]]; then
    printf '[dry-run] '; printf '%q ' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
}

TRAINED_MODEL=""
case "$METHOD" in
  direct_sft)
    run_sft direct_sft "$MODEL" "$MODEL_DIR/final_checkpoint"
    TRAINED_MODEL="$MODEL_DIR/best_checkpoint"
    ;;
  direct_rl)
    ensure_sft direct_sft "$MODEL" "$DIRECT_SFT"
    run_baseline_rl direct_rl "$DIRECT_SFT" "$MODEL_DIR/final_checkpoint"
    TRAINED_MODEL="$MODEL_DIR/final_checkpoint"
    ;;
  minionerec_sft)
    run_sft minionerec_sft "$MODEL" "$MODEL_DIR/final_checkpoint"
    TRAINED_MODEL="$MODEL_DIR/best_checkpoint"
    ;;
  minionerec_rl)
    ensure_sft minionerec_sft "$MODEL" "$MINIONEREC_SFT" "$ITEM_META"
    run_baseline_rl minionerec_rl "$MINIONEREC_SFT" "$MODEL_DIR/final_checkpoint"
    TRAINED_MODEL="$MODEL_DIR/final_checkpoint"
    ;;
  diprec_sft)
    ensure_sft minionerec_sft "$MODEL" "$MINIONEREC_SFT" "$ITEM_META"
    run_sft diprec_sft "$MINIONEREC_SFT" "$MODEL_DIR/final_checkpoint"
    TRAINED_MODEL="$MODEL_DIR/best_checkpoint"
    ;;
  diprec_traj_rl|diprec_plan_rl)
    ensure_sft minionerec_sft "$MODEL" "$MINIONEREC_SFT" "$ITEM_META"
    ensure_sft diprec_sft "$MINIONEREC_SFT" "$DIPREC_SFT" "$ITEM_META"
    if [[ "$METHOD" == "diprec_traj_rl" ]]; then MODE="trajectory_grpo"; else MODE="plan_grpo"; fi
    GRPO_CMD=(bash scripts/train_diprec_grpo.sh
      --mode "$MODE"
      --model "$DIPREC_SFT"
      --train_file "$DATA_DIR/train.jsonl"
      --valid_file "$DATA_DIR/valid.jsonl"
      --sid_index "$SID_INDEX"
      --item_meta "$ITEM_META"
      --output_dir "$MODEL_DIR/final_checkpoint"
      --training_metrics_file "$RUN_DIR/rl_training_metrics.json"
      --interest_topk "$INTEREST_TOPK"
      --interest_strategy "$INTEREST_STRATEGY"
      --time_decay "$TIME_DECAY"
      --sft_plan_mode "$SFT_PLAN_MODE"
      --sft_num_plans "$SFT_NUM_PLANS"
      --num_plans "$NUM_PLANS"
      --sid_beams "$SID_BEAMS"
      --conditioning "$CONDITIONING"
      --interest_parameterization "$INTEREST_PARAMETERIZATION"
      --max_history_len "$MAX_HISTORY_LEN"
      --max_seq_len "$MAX_SEQ_LEN"
      --per_device_batch_size "$DIPREC_RL_PER_DEVICE_BATCH_SIZE"
      --gradient_accumulation_steps "$DIPREC_RL_GRADIENT_ACCUMULATION_STEPS"
      --num_iterations "$DIPREC_RL_NUM_ITERATIONS"
      --beta "$DIPREC_RL_BETA"
      --eval_steps "$DIPREC_RL_EVAL_STEPS"
      --seed "$SEED")
    if [[ -n "$DIPREC_RL_GENERATION_BATCH_SIZE" ]]; then
      GRPO_CMD+=(--generation_batch_size "$DIPREC_RL_GENERATION_BATCH_SIZE")
    fi
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
  --interest_strategy "$INTEREST_STRATEGY"
  --time_decay "$TIME_DECAY"
  --sft_objective "$SFT_OBJECTIVE"
  --sft_plan_mode "$SFT_PLAN_MODE"
  --sft_num_plans "$SFT_NUM_PLANS"
  --num_plans "$NUM_PLANS"
  --sid_beams "$SID_BEAMS"
  --eval_beams "$EVAL_BEAMS"
  --eval_candidate_budget "$EVAL_CANDIDATE_BUDGET"
  --conditioning "$CONDITIONING"
  --interest_parameterization "$INTEREST_PARAMETERIZATION"
  --max_history_len "$MAX_HISTORY_LEN"
  --max_seq_len "$MAX_SEQ_LEN"
  --seed "$SEED")
case "$METHOD" in
  minionerec_sft|minionerec_rl|diprec_sft|diprec_traj_rl|diprec_plan_rl)
    EVAL_CMD+=(--item_meta "$ITEM_META")
    ;;
esac
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
