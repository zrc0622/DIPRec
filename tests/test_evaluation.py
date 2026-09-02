import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diprec.data import processed_data_fingerprint, sha256_file
from diprec.evaluation import (
    _evaluate_record,
    unique_top_candidates,
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
