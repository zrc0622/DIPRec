"""Small, checkpoint-independent training logs for TRL-based RL runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


class MilestoneSnapshotCallback(TrainerCallback):
    """Save evaluable model-only snapshots without consuming training RNG."""

    def __init__(self, trainer: Any, steps: list[int], metadata: Any, expected_steps: int = 0) -> None:
        if steps != sorted(set(steps)) or any(step <= 0 for step in steps):
            raise ValueError("Snapshot steps must be sorted, unique positive integers")
        self.trainer, self.steps, self.metadata = trainer, steps, metadata
        self.expected_steps = expected_steps

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.expected_steps and state.max_steps != self.expected_steps:
            raise ValueError(f'Expected {self.expected_steps} optimizer steps, got {state.max_steps}; input/batch contract changed')
        if any(step >= state.max_steps for step in self.steps):
            raise ValueError("Snapshot steps must precede the final training step")
        for step in self.steps:
            if (Path(args.output_dir).parent / f"step_{step}").exists():
                raise ValueError(f"Snapshot step_{step} already exists; use a new run tag")

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step not in self.steps:
            return
        import random
        import numpy as np
        import torch

        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        destination = Path(args.output_dir).parent / f"step_{state.global_step}"
        try:
            with torch.random.fork_rng(devices=device_ids):
                self.trainer.accelerator.wait_for_everyone()
                self.trainer.save_model(str(destination))
                if self.trainer.accelerator.is_main_process:
                    self.trainer.processing_class.save_pretrained(destination)
                    config = self.metadata(state.global_step, state.max_steps) | {
                        "output_dir": str(destination), "checkpoint_role": "rl_milestone",
                        "snapshot_model_only": True,
                    }
                    (destination / "training_config.json").write_text(
                        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                self.trainer.accelerator.wait_for_everyone()
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


class RLDiagnosticsCallback(TrainerCallback):
    """Append train/eval metrics without changing the legacy eval-only JSON."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination)

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not state.is_world_process_zero:
            return
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(kwargs.get("logs") or {}, step=state.global_step, epoch=state.epoch)
        with self.destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")


class StopAfterStepsCallback(TrainerCallback):
    """Stop a pilot at an update boundary without shortening its LR schedule."""

    def __init__(self, steps: int) -> None:
        if steps < 1:
            raise ValueError("stop_after_steps must be positive")
        self.steps = steps

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.global_step >= self.steps:
            control.should_training_stop = True
        return control


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
