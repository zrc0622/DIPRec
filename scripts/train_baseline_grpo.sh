#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ "${DIPREC_DDP:-0}" == "1" ]]; then
  launch_args=(scripts/train_baseline_grpo.py "$@")
  if [[ -n "${DIPREC_MAIN_PROCESS_PORT:-}" ]]; then
    launch_args=(--main_process_port "$DIPREC_MAIN_PROCESS_PORT" "${launch_args[@]}")
  fi
  exec accelerate launch --multi_gpu --num_processes "${DIPREC_NUM_PROCESSES:-2}" \
    "${launch_args[@]}"
fi
exec python3 scripts/train_baseline_grpo.py "$@"
