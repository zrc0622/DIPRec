"""Small, checkpoint-independent training logs for TRL-based RL runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


class PersistentRLTrainingMetricsCallback(TrainerCallback):
    """Atomically persist Trainer log history whenever TRL emits a log event."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination)

    def _write(self, state: Any, status: str) -> None:
        if not state.is_world_process_zero:
            return
        payload = {
            "status": status,
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": state.epoch,
            "log_history": state.log_history,
        }
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.destination.with_name(
            f".{self.destination.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.destination)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write(state, "running")

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write(state, "running")

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write(state, "complete")
