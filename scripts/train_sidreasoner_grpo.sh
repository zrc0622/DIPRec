#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=""
DATASET=""
DATA_DIR=""
SID_INDEX=""
OUTPUT_DIR=""
MAX_SEQ_LEN=2048
NUM_GENERATIONS=8
SEED=42
NUM_GPUS="${NUM_GPUS:-1}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --sid_index) SID_INDEX="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --num_generations) NUM_GENERATIONS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${MODEL:?--model is required}"
: "${DATASET:?--dataset is required}"
: "${DATA_DIR:?--data_dir is required}"
: "${SID_INDEX:?--sid_index is required}"
: "${OUTPUT_DIR:?--output_dir is required}"
SID_INDEX="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$SID_INDEX")"

VERL_DATA="$DATA_DIR/verl"
mkdir -p "$VERL_DATA" "$OUTPUT_DIR"
PREPARE=(python scripts/prepare_sidreasoner_verl.py
  --data_dir "$DATA_DIR"
  --output_dir "$VERL_DATA"
  --sid_index "$SID_INDEX")
if [[ -f "$DATA_DIR/manifest.json" ]]; then
  PREPARE+=(--max_history_len "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_history_len"])' "$DATA_DIR/manifest.json")")
elif [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Missing long-history manifest: $DATA_DIR/manifest.json" >&2
  exit 1
else
  PREPARE+=(--max_history_len 50)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '[dry-run] '; printf '%q ' "${PREPARE[@]}"; printf '\n'
else
  "${PREPARE[@]}"
fi

CMD=(python3 -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  "data.train_files=$VERL_DATA/train.parquet"
  "data.val_files=$VERL_DATA/valid.parquet"
  data.train_batch_size=128
  "+data.seed=$SEED"
  "data.max_prompt_length=$MAX_SEQ_LEN"
  data.max_response_length=512
  data.filter_overlong_prompts=True
  data.truncation=error
  "actor_rollout_ref.model.path=$MODEL"
  actor_rollout_ref.actor.optim.lr=5e-7
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size=128
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75
  "+actor_rollout_ref.rollout.seed=$SEED"
  "actor_rollout_ref.rollout.n=$NUM_GENERATIONS"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  trainer.logger=['console']
  "custom_reward_function.path=$ROOT_DIR/diprec/sidreasoner_reward.py"
  custom_reward_function.name=compute_score
  "+custom_reward_function.reward_kwargs.sid_index=$SID_INDEX"
  trainer.project_name=DIPRec
  "trainer.experiment_name=sidreasoner_${DATASET}_seed${SEED}"
  "trainer.default_local_dir=$OUTPUT_DIR"
  "trainer.n_gpus_per_node=$NUM_GPUS"
  trainer.nnodes=1
  trainer.save_freq=100
  trainer.test_freq=50
  trainer.total_epochs=1)

export DIPREC_SID_INDEX="$SID_INDEX"
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  "${CMD[@]}"
  LATEST_STEP="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
  if [[ -z "$LATEST_STEP" ]]; then
    echo "No VeRL global_step checkpoint found below $OUTPUT_DIR" >&2
    exit 1
  fi
  python3 scripts/merge_fsdp_checkpoint.py \
    --checkpoint "$LATEST_STEP/actor" \
    --output-dir "$OUTPUT_DIR/final_checkpoint"
fi
