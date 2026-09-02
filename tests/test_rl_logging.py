import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from diprec.rl_logging import PersistentRLTrainingMetricsCallback


class PersistentRLTrainingMetricsCallbackTest(unittest.TestCase):
    def test_only_evaluation_history_is_atomically_refreshed_and_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "logs" / "rl_training_metrics.json"
            callback = PersistentRLTrainingMetricsCallback(destination)
            state = SimpleNamespace(
                is_world_process_zero=True,
                global_step=3,
                max_steps=10,
                epoch=0.3,
                log_history=[
                    {"loss": 1.25, "reward": 0.2, "kl": 0.01, "step": 3},
                    {"eval_loss": 0.75, "eval_runtime": 2.0, "step": 3},
                ],
            )
            callback.on_log(None, state, None, logs={"eval_loss": 0.75})
            running = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["evaluations"], [state.log_history[1]])
            self.assertNotIn("log_history", running)

            state.global_step = 10
            state.epoch = 1.0
            state.log_history.extend(
                [
                    {"loss": 0.9, "learning_rate": 1e-6, "step": 10},
                    {"eval_loss": 0.5, "eval_runtime": 1.5, "step": 10},
                ]
            )
            callback.on_train_end(None, state, None)
            complete = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(len(complete["evaluations"]), 2)
            self.assertEqual(complete["evaluations"][-1]["eval_loss"], 0.5)
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_training_log_does_not_create_or_rewrite_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "rl_training_metrics.json"
            callback = PersistentRLTrainingMetricsCallback(destination)
            state = SimpleNamespace(
                is_world_process_zero=True,
                global_step=1,
                max_steps=2,
                epoch=0.5,
                log_history=[{"loss": 2.0, "reward": 0.0, "step": 1}],
            )
            callback.on_log(None, state, None, logs={"loss": 2.0})
            self.assertFalse(destination.exists())

    def test_non_main_process_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "rl_training_metrics.json"
            callback = PersistentRLTrainingMetricsCallback(destination)
            state = SimpleNamespace(
                is_world_process_zero=False,
                global_step=1,
                max_steps=2,
                epoch=0.5,
                log_history=[{"loss": 2.0}],
            )
            callback.on_log(None, state, None, logs={"eval_loss": 1.0})
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
