import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "clean_rl_training_metrics", ROOT / "scripts/clean_rl_training_metrics.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CleanRLTrainingMetricsTest(unittest.TestCase):
    def test_legacy_log_is_cleaned_with_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl_training_metrics.json"
            original = {
                "status": "complete",
                "global_step": 10,
                "max_steps": 10,
                "epoch": 1.0,
                "log_history": [
                    {"loss": 1.2, "reward": 0.1, "step": 1},
                    {"eval_loss": 0.8, "eval_runtime": 2.0, "step": 5},
                    {"train_runtime": 20.0, "step": 10},
                ],
            }
            path.write_text(json.dumps(original), encoding="utf-8")
            removed = MODULE.rewrite(path)
            self.assertEqual(removed, 2)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {
                    "status": "complete",
                    "max_steps": 10,
                    "evaluations": [original["log_history"][1]],
                },
            )
            self.assertEqual(
                json.loads(path.with_name(f"{path.name}.bak").read_text(encoding="utf-8")),
                original,
            )

    def test_new_format_is_idempotent_and_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl_training_metrics.json"
            payload = {
                "status": "running",
                "max_steps": 20,
                "evaluations": [{"eval_loss": 0.5, "step": 2}],
            }
            original = json.dumps(payload)
            path.write_text(original, encoding="utf-8")
            self.assertEqual(MODULE.rewrite(path, dry_run=True), 0)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertFalse(path.with_name(f"{path.name}.bak").exists())
            self.assertEqual(MODULE.rewrite(path, backup=False), 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_directory_discovery_only_selects_metrics_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wanted = root / "run" / "rl_training_metrics.json"
            wanted.parent.mkdir()
            wanted.write_text("{}", encoding="utf-8")
            (root / "other.json").write_text("{}", encoding="utf-8")
            self.assertEqual(MODULE.discover(root), [wanted])


if __name__ == "__main__":
    unittest.main()
