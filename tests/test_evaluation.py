import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diprec.data import processed_data_fingerprint, sha256_file
from diprec.evaluation import (
    _evaluate_record,
    history_supported_plan,
    unique_top_candidates,
    unique_plans_with_indices,
    validate_evaluation_checkpoint,
)


class EvaluationContractTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "sample_id": "Video_Games:u:2",
            "dataset": "Video_Games",
            "history_item_sid": ["<a_0><b_0><c_0>"],
            "history_sid_levels": [["<a_0>", "<b_0>", "<c_0>"]],
            "history_len": 1,
            "target_sid_levels": ["<a_1>", "<b_1>", "<c_1>"],
        }

    def test_unique_top_candidates_never_pads_with_duplicates(self):
        candidates, valid = unique_top_candidates(
            [
                ["<a_0>", "<b_0>", "<c_0>"],
                ["<a_0>", "<b_0>", "<c_0>"],
                ["<a_1>", "<b_1>", "<c_1>"],
            ],
            [True, False, True],
            10,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(valid, [True, True])

    def test_plan_dedup_keeps_first_rollout_and_history_support(self):
        plans = [["<INT_000>"], ["<INT_000>"], ["<INT_001>"]]
        self.assertEqual(unique_plans_with_indices(plans), ([0, 2], [plans[0], plans[2]]))
        self.assertTrue(history_supported_plan(["<INT_000>", "<INT_PAD>"], self.record))
        self.assertFalse(history_supported_plan(["<INT_001>"], self.record))
        self.assertFalse(history_supported_plan(["<INT_PAD>"], self.record))

    def test_diprec_evaluation_uses_fixed_budget_joint_ranking_and_dedup(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        args = argparse.Namespace(
            method="diprec_sft",
            max_history_len=50,
            max_seq_len=2048,
            interest_topk=1,
            num_plans=2,
            eval_candidate_budget=4,
            eval_beams=3,
            plan_temperature=1.0,
            plan_top_p=0.95,
            plan_sampling_attempts=2,
            conditioning="interest_bottleneck",
        )
        duplicate = ["<a_0>", "<b_0>", "<c_0>"]
        target = ["<a_1>", "<b_1>", "<c_1>"]
        third = ["<a_2>", "<b_2>", "<c_2>"]
        sid_rollouts = [
            ([10], [[100], [101]], [duplicate, target], [True, True]),
            ([20], [[102], [103]], [duplicate, third], [True, True]),
        ]
        scores = [
            torch.tensor([-0.1]),
            torch.tensor([-0.6]),
            torch.tensor([-0.3]),
            torch.tensor([-0.2]),
            torch.tensor([-0.1]),
            torch.tensor([-0.4]),
        ]
        with (
            mock.patch(
                "diprec.evaluation._generate_plans",
                return_value=([1], [[11], [12]], [["<INT_0>"], ["<INT_1>"]]),
            ),
            mock.patch(
                "diprec.evaluation._generate_sid_candidates", side_effect=sid_rollouts
            ) as generate_sid,
            mock.patch("diprec.evaluation._sequence_log_probs", side_effect=scores),
        ):
            prediction, metrics = _evaluate_record(
                object(), object(), object(), object(), self.record, args
            )

        self.assertEqual([call.args[-2] for call in generate_sid.call_args_list], [2, 2])
        self.assertEqual(prediction["raw_candidate_count"], 4)
        self.assertEqual(prediction["unique_candidate_count"], 3)
        self.assertEqual(prediction["returned_candidate_count"], 3)
        self.assertEqual(prediction["per_plan_candidate_budget"], [2, 2])
        self.assertEqual(prediction["requested_plan_count"], 2)
        self.assertEqual(prediction["returned_plan_count"], 2)
        self.assertEqual(metrics["plan_valid_rate"], 1.0)
        self.assertEqual(
            prediction["candidate_sid_levels"], [duplicate, target, third]
        )
        self.assertEqual(metrics["Recall@5"], 1.0)
        self.assertAlmostEqual(metrics["NDCG@5"], 1.0 / __import__("math").log2(3))

    def test_diprec_evaluation_reallocates_budget_when_plans_collapse(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        args = argparse.Namespace(
            method="diprec_sft",
            max_history_len=50,
            max_seq_len=2048,
            interest_topk=1,
            num_plans=8,
            eval_candidate_budget=8,
            eval_beams=3,
            plan_temperature=1.0,
            plan_top_p=0.95,
            plan_sampling_attempts=8,
            conditioning="interest_bottleneck",
        )
        target = ["<a_1>", "<b_1>", "<c_1>"]
        with (
            mock.patch(
                "diprec.evaluation._generate_plans",
                return_value=([1], [[11]], [["<INT_0>"]]),
            ),
            mock.patch(
                "diprec.evaluation._generate_sid_candidates",
                return_value=([10], [[100]], [target], [True]),
            ) as generate_sid,
            mock.patch(
                "diprec.evaluation._sequence_log_probs",
                side_effect=[torch.tensor([-0.1]), torch.tensor([-0.2])],
            ),
        ):
            prediction, metrics = _evaluate_record(
                object(), object(), object(), object(), self.record, args
            )
        self.assertEqual(generate_sid.call_args.args[-2], 8)
        self.assertEqual(prediction["requested_plan_count"], 8)
        self.assertEqual(prediction["returned_plan_count"], 1)
        self.assertEqual(prediction["per_plan_candidate_budget"], [8])
        self.assertEqual(metrics["plan_valid_rate"], 0.125)

    def test_activation_sft_evaluation_measures_duplicates_without_reallocating_budget(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        args = argparse.Namespace(
            method="diprec_sft",
            sft_objective="interest_activation",
            max_history_len=50,
            max_seq_len=2048,
            interest_topk=1,
            num_plans=8,
            eval_candidate_budget=80,
            eval_beams=3,
            plan_temperature=1.0,
            plan_top_p=0.95,
            plan_sampling_attempts=8,
            conditioning="history_visible",
        )
        sampled = [["<INT_000>"]] * 6 + [["<INT_001>"], ["<INT_000>"]]
        target = ["<a_1>", "<b_1>", "<c_1>"]
        with (
            mock.patch(
                "diprec.evaluation._sample_plan_rollouts",
                return_value=([1], [[11]] * 8, sampled),
            ),
            mock.patch(
                "diprec.evaluation._generate_sid_candidates",
                side_effect=[
                    ([10], [[100]], [target], [True]),
                    ([20], [[101]], [["<a_2>", "<b_2>", "<c_2>"]], [True]),
                ],
            ) as generate_sid,
            mock.patch(
                "diprec.evaluation._sequence_log_probs",
                side_effect=[
                    torch.tensor([-0.1]), torch.tensor([-0.2]),
                    torch.tensor([-0.3]), torch.tensor([-0.4]),
                ],
            ),
        ):
            prediction, metrics = _evaluate_record(
                object(), object(), object(), object(), self.record, args
            )
        self.assertEqual([call.args[-2] for call in generate_sid.call_args_list], [10, 10])
        self.assertEqual(prediction["sampled_interest_plans"], sampled)
        self.assertEqual(prediction["returned_plan_count"], 2)
        self.assertEqual(prediction["per_plan_candidate_budget"], [10, 10])
        self.assertAlmostEqual(metrics["plan_duplicate_rate"], 0.75)
        self.assertEqual(metrics["unique_plans@8"], 2.0)
        self.assertEqual(metrics["plan_collapse_rate"], 0.0)
        self.assertAlmostEqual(metrics["plan_history_supported_rate"], 7 / 8)

    def test_joint_activation_continues_sid_in_the_sampled_plan_context(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.last_encode = (text, add_special_tokens)
                if text == "<INT_END></think>":
                    return [90, 91]
                raise AssertionError(f"Unexpected encoding request: {text}")

        args = argparse.Namespace(
            method="diprec_sft",
            sft_objective="joint_interest_activation",
            max_history_len=50,
            max_seq_len=2048,
            interest_topk=1,
            num_plans=1,
            eval_candidate_budget=2,
            eval_beams=2,
            plan_temperature=1.0,
            plan_top_p=0.95,
            plan_sampling_attempts=8,
            conditioning="history_visible",
        )
        target = ["<a_1>", "<b_1>", "<c_1>"]
        tokenizer = Tokenizer()
        with (
            mock.patch(
                "diprec.evaluation._sample_plan_rollouts",
                return_value=([7], [[11]], [["<INT_000>"]]),
            ) as sample_plans,
            mock.patch(
                "diprec.evaluation._generate_catalog_beams",
                return_value=([[101]], [target], [True]),
            ) as generate_joint_sid,
            mock.patch(
                "diprec.evaluation._generate_sid_candidates"
            ) as generate_paired_sid,
            mock.patch(
                "diprec.evaluation._sequence_log_probs",
                side_effect=[torch.tensor([-0.1]), torch.tensor([-0.2])],
            ),
        ):
            prediction, metrics = _evaluate_record(
                object(), tokenizer, object(), object(), self.record, args
            )

        self.assertTrue(sample_plans.call_args.kwargs["joint_trajectory"])
        self.assertFalse(generate_paired_sid.called)
        self.assertEqual(generate_joint_sid.call_args.args[3], [7, 11, 90, 91])
        self.assertEqual(generate_joint_sid.call_args.args[4], 2)
        self.assertEqual(prediction["candidate_sid_levels"], [target])
        self.assertEqual(metrics["Recall@5"], 1.0)

    def test_seven_model_checkpoint_contract_rejects_eval_drift(self):
        manifest = {
            "schema_version": "diprec.long_history.v1",
            "dataset": "Video_Games",
            "source_kind": "sidreasoner_official_csv_reconstruction",
            "source_sha256": {"train": "a", "valid": "b", "test": "c"},
            "sid_index_sha256": "d",
            "split_strategy": "official_temporal",
            "max_history_len": 50,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            item_meta = Path(directory) / "items.json"
            item_meta.write_text("{}\n", encoding="utf-8")
            config = {
                "method": "diprec_plan_rl",
                "data_manifest": processed_data_fingerprint(manifest),
                "item_meta_sha256": sha256_file(item_meta),
                "interest_topk": 3,
                "interest_strategy": "frequency",
                "time_decay": 0.1,
                "conditioning": "interest_bottleneck",
                "interest_parameterization": "independent_head",
                "num_plans": 8,
                "sid_beams": 8,
            }
            (checkpoint / "training_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            args = argparse.Namespace(
                method="diprec_plan_rl",
                model=str(checkpoint),
                item_meta=str(item_meta),
                interest_topk=3,
                interest_strategy="frequency",
                time_decay=0.1,
                conditioning="interest_bottleneck",
                interest_parameterization="independent_head",
                num_plans=8,
                sid_beams=8,
            )
            self.assertEqual(validate_evaluation_checkpoint(args, manifest), config)
            args.num_plans = 4
            with self.assertRaisesRegex(ValueError, "num_plans"):
                validate_evaluation_checkpoint(args, manifest)
            args.num_plans = 8
            args.interest_strategy = "time_decay"
            with self.assertRaisesRegex(ValueError, "interest_strategy"):
                validate_evaluation_checkpoint(args, manifest)


if __name__ == "__main__":
    unittest.main()
