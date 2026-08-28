#!/usr/bin/env python3
"""Convert the shared long-history split into SIDReasoner's VeRL format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diprec.data import read_jsonl, validate_history_records, validate_manifest_sid_index
from diprec.prompts import history_prompt, messages


def convert(source: Path, destination: Path, max_history_len: int, sid_index: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and a parquet engine (pyarrow) are required") from exc
    manifest_path = source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_sid_index(manifest, sid_index)
    records = read_jsonl(source)
    validate_history_records(records, max_history_len, manifest)
    rows = []
    for index, record in enumerate(records):
        prompt = messages(history_prompt(record, max_history_len, reasoning=True))
        rows.append(
            {
                "data_source": "diprec/sidreasoner",
                "prompt": prompt,
                "ability": "Recommendation",
                "reward_model": {"style": "rule", "ground_truth": record["target_item_sid"]},
                "extra_info": {
                    "index": index,
                    "sample_id": record["sample_id"],
                    "split": record["split"],
                    "target_sid_levels": record["target_sid_levels"],
                },
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(destination, index=False)
    print(f"Saved {len(rows)} rows to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sid_index", required=True)
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    args = parser.parse_args()
    source = Path(args.data_dir)
    output = Path(args.output_dir)
    sid_index = Path(args.sid_index)
    convert(source / "train.jsonl", output / "train.parquet", args.max_history_len, sid_index)
    convert(source / "valid.jsonl", output / "valid.parquet", args.max_history_len, sid_index)
    convert(source / "test.jsonl", output / "test.parquet", args.max_history_len, sid_index)


if __name__ == "__main__":
    main()
