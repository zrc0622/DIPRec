import json
import tempfile
import unittest
from pathlib import Path

from diprec.data import (
    Interaction,
    build_chronological_samples,
    iter_raw_records,
    length_statistics,
    validate_history_records,
    validate_manifest_sid_index,
)
from diprec.evaluation import per_plan_candidate_budget, prediction_output_path


def event(user, item, timestamp, order):
    return Interaction(user, item, (0, timestamp), order)


class LongHistoryDataTest(unittest.TestCase):
    def setUp(self):
        self.sid_map = {
            f"i{index}": (f"<a_{index % 3}>", f"<b_{index}>", f"<c_{index}>")
            for index in range(8)
        }

    def test_split_is_prefix_only_and_test_may_see_validation(self):
        events = [event("u", f"i{index}", index, index) for index in range(6)]
        splits, counters = build_chronological_samples(events, self.sid_map, "Video_Games", max_history_len=50)
        self.assertEqual(counters["effective_users"], 1)
        self.assertEqual([row["target_item_id"] for row in splits["train"]], ["i1", "i2", "i3"])
        self.assertEqual(splits["valid"][0]["target_item_id"], "i4")
        self.assertEqual(splits["valid"][0]["history_item_id"], ["i0", "i1", "i2", "i3"])
        self.assertEqual(splits["test"][0]["target_item_id"], "i5")
        self.assertEqual(splits["test"][0]["history_item_id"], ["i0", "i1", "i2", "i3", "i4"])
        self.assertNotIn("i4", splits["train"][-1]["history_item_id"])
        self.assertNotIn("i5", splits["valid"][0]["history_item_id"])

    def test_oldest_side_truncation_and_real_length_validation(self):
        events = [event("u", f"i{index}", index, index) for index in range(6)]
        splits, _ = build_chronological_samples(events, self.sid_map, "Video_Games", max_history_len=2)
        self.assertEqual(splits["test"][0]["history_item_id"], ["i3", "i4"])
        manifest = {"source_kind": "raw_event_interactions", "max_history_len": 2}
        stats = validate_history_records(splits["test"], 2, manifest)
        self.assertEqual(stats["after"]["max"], 2)
        with self.assertRaises(ValueError):
            validate_history_records(splits["test"], 50, manifest)

    def test_materialized_history_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_text("user_id,item_id,history_item_id\nu,i1,[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "materialized history"):
                list(iter_raw_records(path))

    def test_statistics(self):
        stats = length_statistics([1, 20, 50, 100])
        self.assertEqual(stats["effective_users"], 4)
        self.assertAlmostEqual(stats["pct_ge_20"], 0.75)
        self.assertAlmostEqual(stats["pct_ge_50"], 0.5)

    def test_sid_index_checksum_contract(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text('{"i": ["<a_1>", "<b_1>", "<c_1>"]}\n', encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_manifest_sid_index({"sid_index_sha256": digest}, path)
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_manifest_sid_index({"sid_index_sha256": digest}, path)

    def test_validation_predictions_do_not_overwrite_test_predictions(self):
        metrics = Path("outputs/run/metrics.json")
        self.assertEqual(prediction_output_path(metrics, "test"), Path("outputs/run/predictions.jsonl"))
        self.assertEqual(
            prediction_output_path(metrics.with_name("valid_metrics.json"), "valid"),
            Path("outputs/run/valid_predictions.jsonl"),
        )

    def test_evaluation_candidate_budget_is_fixed_across_plans(self):
        self.assertEqual(per_plan_candidate_budget(10, 3), [4, 3, 3])
        self.assertEqual(sum(per_plan_candidate_budget(80, 8)), 80)
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            per_plan_candidate_budget(2, 3)
        with self.assertRaisesRegex(ValueError, "positive"):
            per_plan_candidate_budget(0, 1)


if __name__ == "__main__":
    unittest.main()
