import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunnerContractTest(unittest.TestCase):
    def test_all_comparisons_order_is_dependency_safe(self):
        script = (ROOT / "scripts/run_all_comparisons.sh").read_text(encoding="utf-8")
        self.assertIn(
            "METHODS=(direct_sft direct_rl minionerec_sft minionerec_rl "
            "diprec_sft diprec_traj_rl diprec_plan_rl)",
            script,
        )
        runner = (ROOT / "scripts/run_experiment.sh").read_text(encoding="utf-8")
        self.assertIn(
            'ensure_sft minionerec_sft "$MODEL" "$MINIONEREC_SFT" "$ITEM_META"',
            runner,
        )
        self.assertIn('run_sft diprec_sft "$MINIONEREC_SFT"', runner)
        self.assertIn(
            'ensure_sft diprec_sft "$MINIONEREC_SFT" "$DIPREC_SFT" "$ITEM_META"',
            runner,
        )

    def test_training_wrappers_expose_replicated_ddp_launch(self):
        for name in (
            "train_diprec_sft.sh",
            "train_baseline_grpo.sh",
            "train_diprec_grpo.sh",
        ):
            with self.subTest(script=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn('"${DIPREC_DDP:-0}" == "1"', script)
                self.assertIn(
                    'accelerate launch --multi_gpu --num_processes "${DIPREC_NUM_PROCESSES:-2}"',
                    script,
                )

    def test_diprec_rejects_candidate_budget_below_plan_count(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "diprec_plan_rl",
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
        self.assertIn("--source official", result.stdout)
        self.assertIn("--split_strategy official_temporal", result.stdout)
        self.assertIn("--eval_candidate_budget 80", result.stdout)

    def test_leave_last_two_out_uses_an_isolated_output_path(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "direct_sft",
                "--dataset",
                "Games",
                "--split_strategy",
                "leave_last_two_out",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("history_50_leave_last_two_out", result.stdout)

    def test_both_trl_baselines_dry_run_without_importing_trl(self):
        for method in ("direct_rl", "minionerec_rl"):
            with self.subTest(method=method):
                result = subprocess.run(
                    [
                        "bash",
                        "scripts/run_experiment.sh",
                        "--method",
                        method,
                        "--dataset",
                        "Games",
                        "--dry_run",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"scripts/train_baseline_grpo.sh --method {method}",
                    result.stdout,
                )
                self.assertIn("--num_generations 16", result.stdout)
                self.assertIn("--per_device_batch_size 32", result.stdout)
                self.assertIn("--gradient_accumulation_steps 1", result.stdout)
                self.assertIn("--valid_file", result.stdout)
                self.assertIn("--eval_steps 0.1", result.stdout)

    def test_sft_recipe_controls_and_metrics_file_are_forwarded(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "minionerec_sft",
                "--dataset",
                "Games",
                "--sft_num_epochs",
                "9",
                "--sft_micro_batch_size",
                "3",
                "--sft_gradient_accumulation_steps",
                "10",
                "--sft_learning_rate",
                "7e-5",
                "--sft_weight_decay",
                "0.02",
                "--sft_warmup_ratio",
                "0.04",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in (
            "--num_epochs 9",
            "--micro_batch_size 3",
            "--gradient_accumulation_steps 10",
            "--learning_rate 7e-5",
            "--weight_decay 0.02",
            "--warmup_ratio 0.04",
            "--training_metrics_file outputs/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42/sft_training_metrics.json",
        ):
            with self.subTest(value=value):
                self.assertIn(value, result.stdout)

    def test_run_tag_creates_an_isolated_checkpoint_and_metrics_path(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "minionerec_sft",
                "--dataset",
                "Games",
                "--run_tag",
                "sft6e",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("output_dir/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e/final_checkpoint", result.stdout)
        self.assertIn("output_dir/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e/best_checkpoint", result.stdout)
        self.assertIn(
            "outputs/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e/sft_training_metrics.json",
            result.stdout,
        )
        self.assertNotIn(
            "output_dir/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e/sft_training_metrics.json",
            result.stdout,
        )

    def test_diprec_sft_saves_and_evaluates_the_best_checkpoint(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "diprec_sft",
                "--dataset",
                "Office",
                "--run_tag",
                "sft6e_lr1e-4_best",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        best = "output_dir/Office_Products/history_50/Qwen_Qwen3-0.6B/diprec_sft/seed_42_sft6e_lr1e-4_best/best_checkpoint"
        self.assertIn(f"--best_output_dir {best}", result.stdout)
        self.assertIn(f"--model {best}", result.stdout)
        self.assertIn(
            "--training_metrics_file outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/diprec_sft/seed_42_sft6e_lr1e-4_best/sft_training_metrics.json",
            result.stdout,
        )

    def test_diprec_plan_supervision_ablation_is_forwarded(self):
        for mode in ("single", "diverse"):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [
                        "bash", "scripts/run_experiment.sh",
                        "--method", "diprec_sft", "--dataset", "Office",
                        "--sft_run_tag", "sft6e_lr1e-4_best",
                        "--run_tag", f"plan_{mode}",
                        "--sft_plan_mode", mode, "--sft_num_plans", "8",
                        "--dry_run",
                    ],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--sft_plan_mode {mode}", result.stdout)
                self.assertIn("--sft_num_plans 8", result.stdout)
                self.assertIn(f"diprec_sft/seed_42_plan_{mode}", result.stdout)

    def test_interest_activation_sft_uses_isolated_directory_and_contract(self):
        result = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh",
                "--method", "diprec_sft",
                "--dataset", "Office",
                "--sft_run_tag", "sft6e_lr1e-4_best",
                "--run_tag", "interest_activation_plan_diverse",
                "--sft_objective", "interest_activation",
                "--conditioning", "history_visible",
                "--sft_plan_mode", "diverse",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        run = "diprec_sft/seed_42_interest_activation_plan_diverse"
        self.assertIn(f"output_dir/Office_Products/history_50/Qwen_Qwen3-0.6B/{run}/best_checkpoint", result.stdout)
        self.assertIn(f"outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/{run}/sft_training_metrics.json", result.stdout)
        self.assertIn("--sft_objective interest_activation", result.stdout)
        self.assertIn("--conditioning history_visible", result.stdout)

    def test_interest_activation_rejects_hidden_history(self):
        result = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh",
                "--method", "diprec_sft", "--dataset", "Office",
                "--sft_objective", "interest_activation", "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --conditioning history_visible", result.stderr)

    def test_joint_interest_activation_has_isolated_command_and_directory(self):
        result = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh",
                "--method", "diprec_sft",
                "--dataset", "Office",
                "--sft_run_tag", "sft6e_lr1e-4_best",
                "--run_tag", "joint_interest_activation_plan_diverse",
                "--sft_objective", "joint_interest_activation",
                "--conditioning", "history_visible",
                "--sft_plan_mode", "diverse",
                "--sft_micro_batch_size", "4",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        run = "diprec_sft/seed_42_joint_interest_activation_plan_diverse"
        self.assertIn(
            f"output_dir/Office_Products/history_50/Qwen_Qwen3-0.6B/{run}/best_checkpoint",
            result.stdout,
        )
        self.assertIn("--sft_objective joint_interest_activation", result.stdout)
        self.assertIn("--micro_batch_size 4", result.stdout)

    def test_joint_interest_activation_rejects_hidden_history(self):
        result = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh",
                "--method", "diprec_sft", "--dataset", "Office",
                "--sft_objective", "joint_interest_activation", "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --conditioning history_visible", result.stderr)

    def test_joint_interest_activation_is_not_silently_used_by_two_stage_rl(self):
        result = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh",
                "--method", "diprec_plan_rl", "--dataset", "Office",
                "--sft_objective", "joint_interest_activation",
                "--conditioning", "history_visible", "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("joint-trajectory RL will be added separately", result.stderr)

    def test_diprec_rl_selects_single_or_diverse_sft_parent(self):
        for mode in ("single", "diverse"):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [
                        "bash", "scripts/run_experiment.sh",
                        "--method", "diprec_plan_rl", "--dataset", "Office",
                        "--sft_run_tag", "sft6e_lr1e-4_best",
                        "--diprec_sft_run_tag", f"plan_{mode}",
                        "--run_tag", f"plan_{mode}_rl",
                        "--sft_plan_mode", mode, "--sft_num_plans", "8",
                        "--dry_run",
                    ],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"diprec_sft/seed_42_plan_{mode}/best_checkpoint",
                    result.stdout,
                )
                self.assertIn(
                    f"diprec_plan_rl/seed_42_plan_{mode}_rl/final_checkpoint",
                    result.stdout,
                )

    def test_diprec_plan_supervision_ablation_is_forwarded(self):
        for mode in ("single", "diverse"):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [
                        "bash", "scripts/run_experiment.sh",
                        "--method", "diprec_sft", "--dataset", "Office",
                        "--run_tag", f"plan_{mode}",
                        "--sft_plan_mode", mode, "--sft_num_plans", "8",
                        "--dry_run",
                    ],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--sft_plan_mode {mode}", result.stdout)
                self.assertIn("--sft_num_plans 8", result.stdout)
                self.assertIn(f"diprec_sft/seed_42_plan_{mode}", result.stdout)

    def test_run_tag_keeps_rl_parent_checkpoint_in_the_same_lineage(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "minionerec_rl",
                "--dataset",
                "Games",
                "--run_tag",
                "sft6e",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "minionerec_sft/seed_42_sft6e/best_checkpoint",
            result.stdout,
        )
        self.assertIn(
            "--training_metrics_file outputs/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e/sft_training_metrics.json",
            result.stdout,
        )
        self.assertIn(
            "--output_dir output_dir/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_rl/seed_42_sft6e/final_checkpoint",
            result.stdout,
        )
        self.assertIn(
            "--training_metrics_file outputs/Video_Games/history_50/Qwen_Qwen3-0.6B/minionerec_rl/seed_42_sft6e/rl_training_metrics.json",
            result.stdout,
        )

    def test_diprec_rl_writes_metrics_separately_from_checkpoints(self):
        result = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh", "--method", "diprec_plan_rl",
                "--dataset", "Office", "--sft_run_tag", "sft_parent",
                "--diprec_sft_run_tag", "plan_parent", "--run_tag", "rl_child",
                "--dry_run",
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--training_metrics_file outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/diprec_plan_rl/seed_42_rl_child/rl_training_metrics.json",
            result.stdout,
        )
        self.assertNotIn(
            "final_checkpoint/rl_training_metrics.json",
            result.stdout,
        )

    def test_rl_reference_ablation_reuses_one_sft_parent_but_separates_outputs(self):
        common = [
            "bash", "scripts/run_experiment.sh", "--method", "minionerec_rl",
            "--dataset", "Office", "--sft_run_tag", "sft6e_lr1e-4_best", "--dry_run",
        ]
        fixed = subprocess.run(
            common + ["--run_tag", "rl_fixed", "--baseline_rl_reference_mode", "fixed"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        synced = subprocess.run(
            common + ["--run_tag", "rl_sync", "--baseline_rl_reference_mode", "sync"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(fixed.returncode, 0, fixed.stderr)
        self.assertEqual(synced.returncode, 0, synced.stderr)
        parent = "minionerec_sft/seed_42_sft6e_lr1e-4_best/best_checkpoint"
        self.assertIn(parent, fixed.stdout)
        self.assertIn(parent, synced.stdout)
        self.assertIn("minionerec_rl/seed_42_rl_fixed/final_checkpoint", fixed.stdout)
        self.assertIn("--reference_mode fixed", fixed.stdout)
        self.assertIn("minionerec_rl/seed_42_rl_sync/final_checkpoint", synced.stdout)
        self.assertIn("--reference_mode sync", synced.stdout)
        self.assertIn("--ref_model_sync_steps 512", synced.stdout)
        self.assertIn("--ref_model_mixup_alpha 0.6", synced.stdout)

    def test_minionerec_history_only_scope_is_explicitly_forwarded(self):
        result = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh",
                "--method", "minionerec_rl",
                "--dataset", "Office",
                "--sft_run_tag", "sft6e_lr1e-4_best",
                "--run_tag", "rl_history_only",
                "--baseline_rl_task_scope", "history_only",
                "--dry_run",
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--task_scope history_only", result.stdout)
        self.assertIn(
            "minionerec_rl/seed_42_rl_history_only/final_checkpoint",
            result.stdout,
        )

        invalid = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh",
                "--method", "minionerec_rl",
                "--dataset", "Office",
                "--baseline_rl_task_scope", "invalid",
                "--dry_run",
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("official_mixed or history_only", invalid.stderr)

    def test_baseline_rl_optimization_overrides_are_forwarded(self):
        result = subprocess.run(
            [
                "bash", "scripts/run_experiment.sh",
                "--method", "minionerec_rl",
                "--dataset", "Office",
                "--baseline_rl_learning_rate", "2e-6",
                "--baseline_rl_beta", "1e-2",
                "--baseline_rl_num_epochs", "1",
                "--dry_run",
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--learning_rate 2e-6", result.stdout)
        self.assertIn("--beta 1e-2", result.stdout)
        self.assertIn("--num_epochs 1", result.stdout)
        self.assertIn("--task_scope official_mixed", result.stdout)

    def test_diprec_rl_branches_share_the_same_sft_parent(self):
        script = (ROOT / "scripts/run_experiment.sh").read_text(encoding="utf-8")
        self.assertIn("diprec_traj_rl|diprec_plan_rl)", script)
        self.assertIn('--model "$DIPREC_SFT"', script)
        self.assertIn('--item_meta "$ITEM_META"', script)
        self.assertIn('"interest_strategy":sys.argv[7]', script)
        self.assertIn('"time_decay":float(sys.argv[8])', script)
        self.assertIn('diprec_interest_adapter.pt', script)
        self.assertIn('training.get("checkpoint_role")=="best_validation"', script)

    def test_both_diprec_rl_methods_use_corrected_trl_lifecycle(self):
        for method, mode in (
            ("diprec_traj_rl", "trajectory_grpo"),
            ("diprec_plan_rl", "plan_grpo"),
        ):
            with self.subTest(method=method):
                result = subprocess.run(
                    [
                        "bash",
                        "scripts/run_experiment.sh",
                        "--method",
                        method,
                        "--dataset",
                        "Games",
                        "--dry_run",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"scripts/train_diprec_grpo.sh --mode {mode}",
                    result.stdout,
                )
                self.assertIn("--num_iterations 2", result.stdout)
                self.assertIn("--beta 0.001", result.stdout)
                self.assertIn("--valid_file", result.stdout)
                self.assertIn("--eval_steps 0.1", result.stdout)
                self.assertIn("--interest_strategy frequency", result.stdout)
                self.assertIn("--time_decay 0.1", result.stdout)

    def test_diprec_generation_batch_defaults_to_effective_update_batch(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "diprec_plan_rl",
                "--dataset",
                "Games",
                "--num_plans",
                "4",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--num_plans 4", result.stdout)
        self.assertIn("--gradient_accumulation_steps 8", result.stdout)
        self.assertNotIn("--generation_batch_size", result.stdout)

    def test_all_comparisons_forwards_diprec_ablation_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_file = Path(directory) / "datasets.txt"
            dataset_file.write_text("Games\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    "scripts/run_all_comparisons.sh",
                    "--dataset_file",
                    str(dataset_file),
                    "--interest_topk",
                    "2",
                    "--num_plans",
                    "4",
                    "--sid_beams",
                    "6",
                    "--conditioning",
                    "history_visible",
                    "--interest_parameterization",
                    "disjoint_rows",
                    "--interest_strategy",
                    "time_decay",
                    "--time_decay",
                    "0.2",
                    "--dry_run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--num_plans 4", result.stdout)
        self.assertIn("--sid_beams 6", result.stdout)
        self.assertIn("--conditioning history_visible", result.stdout)
        self.assertIn("--interest_parameterization disjoint_rows", result.stdout)
        self.assertIn("--interest_strategy time_decay", result.stdout)
        self.assertIn("--time_decay 0.2", result.stdout)

    def test_explicit_rl_generation_batches_are_forwarded(self):
        for method, flag, batch in (
            ("direct_rl", "--baseline_rl_generation_batch_size", 32),
            ("diprec_plan_rl", "--diprec_rl_generation_batch_size", 16),
        ):
            with self.subTest(method=method):
                result = subprocess.run(
                    [
                        "bash",
                        "scripts/run_experiment.sh",
                        "--method",
                        method,
                        "--dataset",
                        "Games",
                        flag,
                        str(batch),
                        "--dry_run",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--generation_batch_size {batch}", result.stdout)

    def test_explicit_generation_batch_is_deferred_to_trainer_validation(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "direct_rl",
                "--dataset",
                "Games",
                "--baseline_rl_generation_batch_size",
                "16",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--generation_batch_size 16", result.stdout)

    def test_legacy_method_aliases_resolve_to_canonical_output_names(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_experiment.sh",
                "--method",
                "diprec_plan_grpo",
                "--dataset",
                "Games",
                "--dry_run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scripts/train_diprec_grpo.sh --mode plan_grpo", result.stdout)
        self.assertIn("/diprec_plan_rl/seed_42/final_checkpoint", result.stdout)
        self.assertIn("scripts/eval_diprec.sh --method diprec_plan_rl", result.stdout)

if __name__ == "__main__":
    unittest.main()
