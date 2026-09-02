"""Small, checkpoint-independent training logs for TRL-based RL runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


class PersistentRLTrainingMetricsCallback(TrainerCallback):
    """Atomically persist only periodic evaluation results outside checkpoints."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination)

    @staticmethod
    def _evaluations(state: Any) -> list[dict[str, Any]]:
        return [
            entry
            for entry in state.log_history
            if any(key.startswith("eval_") for key in entry)
        ]

    def _write(self, state: Any, status: str) -> None:
        if not state.is_world_process_zero:
            return
        payload = {
            "status": status,
            "max_steps": state.max_steps,
            "evaluations": self._evaluations(state),
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

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        logs = kwargs.get("logs") or {}
        if any(key.startswith("eval_") for key in logs):
            self._write(state, "running")

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write(state, "complete")
