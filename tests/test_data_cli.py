import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_FIELDS = [
    "user_id",
    "history_item_id",
    "item_id",
    "history_item_sid",
    "item_sid",
]


def sid_map(size):
    return {
        str(index): [f"<a_{index % 3}>", f"<b_{index}>", f"<c_{index}>"]
        for index in range(size)
    }


def write_dataset(root, dataset, sequence_length):
    mapping = sid_map(sequence_length)
    (root / "index").mkdir(parents=True, exist_ok=True)
    (root / "index" / f"{dataset}.index.json").write_text(
        json.dumps(mapping) + "\n",
        encoding="utf-8",
    )

    target_positions = list(range(1, sequence_length))
    train_end = int(len(target_positions) * 0.8)
    valid_end = int(len(target_positions) * 0.9)
    positions = {
        "train": target_positions[:train_end],
        "valid": target_positions[train_end:valid_end],
        "test": target_positions[valid_end:],
    }
    items = list(mapping)
    for split, split_positions in positions.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{dataset}_5_2016-10-2018-11.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OFFICIAL_FIELDS)
            writer.writeheader()
            for position in split_positions:
                history = items[max(0, position - 10) : position]
                writer.writerow(
                    {
                        "user_id": "u",
                        "history_item_id": repr(history),
                        "item_id": items[position],
                        "history_item_sid": repr(
                            ["".join(mapping[item]) for item in history]
                        ),
                        "item_sid": "".join(mapping[items[position]]),
                    }
                )


class DataCliTest(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_official_selector_and_default_builder_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            official_root = temp / "Amazon"
            write_dataset(official_root, "Video_Games", 13)
            write_dataset(official_root, "Office_Products", 7)
            stats_path = temp / "stats.csv"
            selection_path = temp / "selected.txt"

            selection = self.run_cli(
                "scripts/select_long_history_datasets.py",
                "--datasets",
                "Office_Products,Games",
                "--top_n",
                "1",
                "--official_data_root",
                str(official_root),
                "--sid_data_root",
                str(official_root),
                "--stats_output",
                str(stats_path),
                "--selection_output",
                str(selection_path),
            )
            self.assertEqual(selection.returncode, 0, selection.stderr)
            self.assertEqual(selection_path.read_text(encoding="utf-8"), "Video_Games\n")
            with stats_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["dataset"] for row in rows], ["Video_Games", "Office_Products"])
            self.assertTrue(all(row["source_kind"] == "official" for row in rows))

            output = temp / "processed"
            build = self.run_cli(
                "scripts/build_long_history_data.py",
                "--dataset",
                "Games",
                "--official_data_root",
                str(official_root),
                "--sid_data_root",
                str(official_root),
                "--output_dir",
                str(output),
                "--max_history_len",
                "10",
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_kind"], "sidreasoner_official_csv_reconstruction")
            self.assertEqual(manifest["split_strategy"], "official_temporal")
            self.assertEqual(set(manifest["source_files"]), {"train", "valid", "test"})
            self.assertEqual(manifest["mapped_sid_events"], 13)
            self.assertEqual(
                {split: manifest["split_statistics"][split]["samples"] for split in ("train", "valid", "test")},
                {"train": 9, "valid": 1, "test": 2},
            )
            test_rows = [
                json.loads(line)
                for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            test_row = test_rows[-1]
            self.assertEqual(test_row["history_len_before_truncation"], 12)
            self.assertEqual(test_row["history_len"], 10)

    def test_leave_last_two_out_remains_an_explicit_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            official_root = temp / "Amazon"
            write_dataset(official_root, "Video_Games", 13)
            output = temp / "processed"
            build = self.run_cli(
                "scripts/build_long_history_data.py",
                "--dataset",
                "Games",
                "--official_data_root",
                str(official_root),
                "--sid_data_root",
                str(official_root),
                "--output_dir",
                str(output),
                "--max_history_len",
                "10",
                "--split_strategy",
                "leave_last_two_out",
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_strategy"], "leave_last_two_out")
            self.assertEqual(
                {split: manifest["split_statistics"][split]["samples"] for split in ("train", "valid", "test")},
                {"train": 10, "valid": 1, "test": 1},
            )

    def test_raw_builder_reports_item_namespace_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            raw_path = temp / "reviews.jsonl"
            raw_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "reviewerID": "u",
                            "asin": f"ASIN{index}",
                            "unixReviewTime": index,
                        }
                    )
                    + "\n"
                    for index in range(4)
                ),
                encoding="utf-8",
            )
            index_path = temp / "index.json"
            index_path.write_text(
                json.dumps({"0": ["<a_0>", "<b_0>", "<c_0>"]}) + "\n",
                encoding="utf-8",
            )

            result = self.run_cli(
                "scripts/build_long_history_data.py",
                "--dataset",
                "Industrial",
                "--source",
                "raw",
                "--split_strategy",
                "leave_last_two_out",
                "--raw_path",
                str(raw_path),
                "--sid_index",
                str(index_path),
                "--output_dir",
                str(temp / "processed"),
                "--max_history_len",
                "10",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No source item IDs match the SID index", result.stderr)
            self.assertIn("Amazon review dumps use ASINs", result.stderr)


if __name__ == "__main__":
    unittest.main()
