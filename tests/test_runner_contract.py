import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunnerContractTest(unittest.TestCase):
    def test_all_comparisons_order_is_dependency_safe(self):
        script = (ROOT / "scripts/run_all_comparisons.sh").read_text(encoding="utf-8")
        self.assertIn(
            "METHODS=(direct_sid sidreasoner diprec_sft diprec_trajectory_grpo diprec_plan_grpo)",
            script,
        )

    def test_diprec_rejects_candidate_budget_below_plan_count(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "diprec_plan_grpo",
                "--dataset",
                "Games",
                "--num_plans",
                "8",
                "--eval_beams",
                "2",
                "--eval_candidate_budget",
                "4",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("at least --num_plans", result.stderr)

    def test_games_alias_reaches_canonical_dry_run_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_file = Path(directory) / "datasets.txt"
            dataset_file.write_text("Games\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    "scripts/run_all_comparisons.sh",
                    "--dataset_file",
                    str(dataset_file),
                    "--dry_run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data/processed/Video_Games/history_50", result.stdout)
        self.assertIn("--eval_candidate_budget 80", result.stdout)


if __name__ == "__main__":
    unittest.main()
