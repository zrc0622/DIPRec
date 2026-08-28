#!/usr/bin/env python3
"""Build shared chronological train/valid/test data from raw interactions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diprec.constants import SCHEMA_VERSION, canonical_dataset
from diprec.data import (
    build_chronological_samples,
    iter_interactions,
    load_sid_map,
    resolve_raw_path,
    resolve_sid_index,
    sample_length_statistics,
    sha256_file,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw_path")
    parser.add_argument("--raw_data_root", default="data/Amazon/raw")
    parser.add_argument("--sid_index")
    parser.add_argument("--sid_data_root", default="data/Amazon")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    parser.add_argument("--min_user_interactions", type=int, default=3)
    args = parser.parse_args()

    dataset = canonical_dataset(args.dataset)
    raw_path = Path(args.raw_path) if args.raw_path else resolve_raw_path(dataset, args.raw_data_root)
    sid_path = Path(args.sid_index) if args.sid_index else resolve_sid_index(dataset, args.sid_data_root)
    output = Path(args.output_dir or f"data/processed/{dataset}/history_{args.max_history_len}")

    sid_map = load_sid_map(sid_path)
    splits, counters = build_chronological_samples(
        iter_interactions(raw_path),
        sid_map,
        dataset,
        max_history_len=args.max_history_len,
        min_user_interactions=args.min_user_interactions,
    )
    if not splits["train"]:
        raise RuntimeError(
            "No train samples remain after SID filtering and leave-last-two-out splitting; "
            "users need at least four mapped interactions to contribute a train target"
        )

    output.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for split, records in splits.items():
        write_jsonl(output / f"{split}.jsonl", records)
        split_stats[split] = sample_length_statistics(records, args.max_history_len)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_kind": "raw_event_interactions",
        "raw_file": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "sid_index": str(sid_path),
        "sid_index_sha256": sha256_file(sid_path),
        "split_strategy": "per_user_leave_last_two_out_after_sid_filtering",
        "train_sample_strategy": "all_prefix_targets_before_validation",
        "test_history_includes_prior_validation_event": True,
        "max_history_len": args.max_history_len,
        "truncation_side": "oldest",
        "min_user_interactions": args.min_user_interactions,
        **counters,
        "split_statistics": split_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (output / "history_length_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "stage", "samples", "mean", "p50", "p90", "max", "fraction_reaching_cap", "fraction_reaching_50"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, stats in split_stats.items():
            for stage in ("before", "after"):
                values = stats[stage]
                writer.writerow(
                    {
                        "split": split,
                        "stage": stage,
                        "samples": stats["samples"],
                        "mean": f"{values['mean']:.4f}",
                        "p50": f"{values['p50']:.4f}",
                        "p90": f"{values['p90']:.4f}",
                        "max": values["max"],
                        "fraction_reaching_cap": f"{stats['fraction_reaching_cap']:.8f}",
                        "fraction_reaching_50": f"{stats['fraction_reaching_50']:.8f}",
                    }
                )
    print(json.dumps({"output_dir": str(output), **counters, "split_statistics": split_stats}, indent=2))


if __name__ == "__main__":
    main()
