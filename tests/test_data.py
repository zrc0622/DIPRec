import csv
import json
import tempfile
import unittest
from pathlib import Path

from diprec.data import (
    Interaction,
    build_chronological_samples,
    build_official_temporal_samples,
    interactions_from_sequences,
    iter_raw_records,
    length_statistics,
    official_history_statistics,
    processed_data_fingerprint,
    reconstruct_official_sequences,
    resolve_official_csv_paths,
    resolve_sid_index,
    validate_history_records,
    validate_manifest_sid_index,
    validate_manifest_sources,
    validate_processed_manifest,
    validate_checkpoint_training_contract,
)
from diprec.evaluation import per_plan_candidate_budget, prediction_output_path


def event(user, item, timestamp, order):
    return Interaction(user, item, (0, timestamp), order)


def write_official_csv(path, rows, sid_map):
    fields = ["user_id", "history_item_id", "item_id", "history_item_sid", "item_sid"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for user_id, history, target in rows:
            writer.writerow(
                {
                    "user_id": user_id,
                    "history_item_id": repr(history),
                    "item_id": target,
                    "history_item_sid": repr(["".join(sid_map[item]) for item in history]),
                    "item_sid": "".join(sid_map[target]),
                }
            )


class LongHistoryDataTest(unittest.TestCase):
    def setUp(self):
        self.sid_map = {
            f"i{index}": (f"<a_{index % 3}>", f"<b_{index}>", f"<c_{index}>")
            for index in range(16)
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
        official_manifest = {
            "source_kind": "sidreasoner_official_csv_reconstruction",
            "max_history_len": 2,
        }
        validate_history_records(splits["test"], 2, official_manifest)
        with self.assertRaises(ValueError):
            validate_history_records(splits["test"], 50, manifest)

    def test_materialized_history_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_text("user_id,item_id,history_item_id\nu,i1,[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "materialized history"):
                list(iter_raw_records(path))

    def test_complete_official_windows_reconstruct_full_sequence(self):
        items = [f"i{index}" for index in range(13)]

        def row(position):
            return ("u", items[max(0, position - 10) : position], items[position])

        rows = {
            "train": [row(position) for position in range(1, 11)],
            "valid": [row(11)],
            "test": [row(12)],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for split, split_rows in rows.items():
                paths[split] = Path(directory) / f"{split}.csv"
                write_official_csv(paths[split], split_rows, self.sid_map)

            sequences, metadata = reconstruct_official_sequences(paths, self.sid_map)
            stats = official_history_statistics(paths, self.sid_map)

        self.assertEqual(sequences["u"], items)
        self.assertEqual(metadata["official_rows_by_split"], {"train": 10, "valid": 1, "test": 1})
        self.assertEqual(stats["max"], 13)
        rebuilt, counters = build_chronological_samples(
            interactions_from_sequences(sequences),
            self.sid_map,
            "Video_Games",
            max_history_len=50,
        )
        self.assertEqual(counters["dropped_unknown_sid_events"], 0)
        self.assertEqual(rebuilt["valid"][0]["target_item_id"], "i11")
        self.assertEqual(rebuilt["valid"][0]["history_item_id"], items[:11])
        self.assertEqual(rebuilt["test"][0]["target_item_id"], "i12")

    def test_official_temporal_preserves_rows_and_expands_only_the_past(self):
        items = [f"i{index}" for index in range(16)]

        def row(position):
            return ("u", items[max(0, position - 10) : position], items[position])

        rows = {
            "train": [row(position) for position in range(1, 12)],
            "valid": [row(12), row(13)],
            "test": [row(14), row(15)],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for split, split_rows in rows.items():
                paths[split] = Path(directory) / f"{split}.csv"
                write_official_csv(paths[split], split_rows, self.sid_map)
            splits, counters, metadata = build_official_temporal_samples(
                paths,
                self.sid_map,
                "Video_Games",
                max_history_len=50,
            )

        self.assertEqual(
            {split: len(records) for split, records in splits.items()},
            {"train": 11, "valid": 2, "test": 2},
        )
        self.assertEqual(
            metadata["official_rows_by_split"],
            {"train": 11, "valid": 2, "test": 2},
        )
        self.assertEqual(counters["mapped_sid_events"], 16)
        self.assertEqual([row["target_item_id"] for row in splits["valid"]], ["i12", "i13"])
        self.assertEqual(splits["valid"][0]["history_item_id"], items[:12])
        self.assertEqual(splits["test"][-1]["history_item_id"], items[:15])
        self.assertNotIn("i12", splits["valid"][0]["history_item_id"])
        self.assertNotIn("i15", splits["test"][-1]["history_item_id"])

    def test_official_temporal_truncates_recovered_history_at_requested_cap(self):
        items = [f"i{index}" for index in range(16)]

        def row(position):
            return ("u", items[max(0, position - 10) : position], items[position])

        rows = {
            "train": [row(position) for position in range(1, 12)],
            "valid": [row(12), row(13)],
            "test": [row(14), row(15)],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for split, split_rows in rows.items():
                paths[split] = Path(directory) / f"{split}.csv"
                write_official_csv(paths[split], split_rows, self.sid_map)
            splits, _, _ = build_official_temporal_samples(
                paths,
                self.sid_map,
                "Video_Games",
                max_history_len=12,
            )

        row = splits["test"][-1]
        self.assertEqual(row["history_len_before_truncation"], 15)
        self.assertEqual(row["history_item_id"], items[3:15])

    def test_official_reconstruction_rejects_discontinuous_window(self):
        rows = {
            "train": [("u", ["i0"], "i1")],
            "valid": [("u", ["i7"], "i2")],
            "test": [("u", ["i1", "i2"], "i3")],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for split, split_rows in rows.items():
                paths[split] = Path(directory) / f"{split}.csv"
                write_official_csv(paths[split], split_rows, self.sid_map)
            with self.assertRaisesRegex(ValueError, "discontinuous window"):
                reconstruct_official_sequences(paths, self.sid_map)

    def test_official_reconstruction_rejects_sid_mismatch(self):
        rows = {
            "train": [("u", ["i0"], "i1")],
            "valid": [("u", ["i0", "i1"], "i2")],
            "test": [("u", ["i0", "i1", "i2"], "i3")],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for split, split_rows in rows.items():
                paths[split] = Path(directory) / f"{split}.csv"
                write_official_csv(paths[split], split_rows, self.sid_map)
            text = paths["valid"].read_text(encoding="utf-8")
            paths["valid"].write_text(text.replace("<c_1>", "<c_9>"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SID mismatch"):
                reconstruct_official_sequences(paths, self.sid_map)

    def test_official_path_resolution_requires_complete_triplet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "valid", "test"):
                (root / split).mkdir()
                (root / split / "Video_Games_5_2016-10-2018-11.csv").touch()
            (root / "index").mkdir()
            index_path = root / "index" / "Video_Games.index.json"
            index_path.write_text("{}\n", encoding="utf-8")

            paths = resolve_official_csv_paths("Games", root)
            self.assertEqual(paths["valid"], root / "valid" / "Video_Games_5_2016-10-2018-11.csv")
            self.assertEqual(resolve_sid_index("Games", root), index_path)

            paths["valid"].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing official valid CSV"):
                resolve_official_csv_paths("Games", root)

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

    def test_manifest_source_checksum_contract(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.jsonl"
            source.write_text('{"item_id": "i"}\n', encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = {
                "source_kind": "raw_event_interactions",
                "split_strategy": "leave_last_two_out",
                "source_files": {"raw": str(source)},
                "source_sha256": {"raw": digest},
            }
            validate_manifest_sources(manifest, "raw_event_interactions")
            with self.assertRaisesRegex(ValueError, "Data source mismatch"):
                validate_manifest_sources(manifest, "sidreasoner_official_csv_reconstruction")
            source.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                validate_manifest_sources(manifest)

    def test_processed_manifest_reuse_checks_dataset_and_schema(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            index = root / "index.json"
            index.write_text("{}\n", encoding="utf-8")
            manifest = {
                "schema_version": "diprec.long_history.v1",
                "dataset": "Video_Games",
                "max_history_len": 50,
                "source_kind": "raw_event_interactions",
                "split_strategy": "leave_last_two_out",
                "source_files": {"raw": str(source)},
                "source_sha256": {"raw": hashlib.sha256(source.read_bytes()).hexdigest()},
                "sid_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            }
            validate_processed_manifest(
                manifest,
                dataset="Games",
                max_history_len=50,
                source_kind="raw_event_interactions",
                split_strategy="leave_last_two_out",
                sid_index_path=index,
            )
            with self.assertRaisesRegex(ValueError, "Dataset mismatch"):
                validate_processed_manifest(
                    manifest,
                    dataset="Office",
                    max_history_len=50,
                    source_kind="raw_event_interactions",
                    split_strategy="leave_last_two_out",
                    sid_index_path=index,
                )
            with self.assertRaisesRegex(ValueError, "Split strategy mismatch"):
                validate_processed_manifest(
                    manifest,
                    dataset="Games",
                    max_history_len=50,
                    source_kind="raw_event_interactions",
                    split_strategy="official_temporal",
                    sid_index_path=index,
                )

    def test_checkpoint_data_fingerprint_includes_split_strategy(self):
        manifest = {
            "schema_version": "diprec.long_history.v1",
            "dataset": "Video_Games",
            "source_kind": "sidreasoner_official_csv_reconstruction",
            "source_sha256": {"train": "a", "valid": "b", "test": "c"},
            "sid_index_sha256": "d",
            "split_strategy": "official_temporal",
            "max_history_len": 50,
            "unrelated": "ignored",
        }
        fingerprint = processed_data_fingerprint(manifest)
        self.assertEqual(fingerprint["split_strategy"], "official_temporal")
        self.assertNotIn("unrelated", fingerprint)

    def test_checkpoint_contract_rejects_wrong_method_or_item_metadata(self):
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
            item_meta.write_text("{}", encoding="utf-8")
            config = {
                "method": "minionerec_sft",
                "data_manifest": processed_data_fingerprint(manifest),
                "max_history_len": 50,
            }
            from diprec.data import sha256_file

            config["item_meta_sha256"] = sha256_file(item_meta)
            (checkpoint / "training_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            validate_checkpoint_training_contract(
                checkpoint,
                expected_method="minionerec_sft",
                manifest=manifest,
                item_meta_path=item_meta,
                expected_config={"max_history_len": 50},
            )
            with self.assertRaisesRegex(ValueError, "expected 'direct_sft'"):
                validate_checkpoint_training_contract(
                    checkpoint, expected_method="direct_sft", manifest=manifest
                )
            item_meta.write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different item metadata"):
                validate_checkpoint_training_contract(
                    checkpoint,
                    expected_method="minionerec_sft",
                    manifest=manifest,
                    item_meta_path=item_meta,
                )
            with self.assertRaisesRegex(ValueError, "max_history_len"):
                validate_checkpoint_training_contract(
                    checkpoint,
                    expected_method="minionerec_sft",
                    manifest=manifest,
                    expected_config={"max_history_len": 10},
                )

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
