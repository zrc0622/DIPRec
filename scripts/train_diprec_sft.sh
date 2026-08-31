#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ "${DIPREC_DDP:-0}" == "1" ]]; then
  exec accelerate launch --multi_gpu --num_processes "${DIPREC_NUM_PROCESSES:-2}" \
    scripts/train_diprec_sft.py "$@"
fi
exec python3 scripts/train_diprec_sft.py "$@"
