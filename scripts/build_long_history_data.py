#!/usr/bin/env python3
"""Build shared chronological data from official windows or raw events."""

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
    build_official_temporal_samples,
    interactions_from_sequences,
    iter_interactions,
    load_sid_map,
    reconstruct_official_sequences,
    resolve_official_csv_paths,
    resolve_raw_path,
    resolve_sid_index,
    sample_length_statistics,
    sha256_file,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--source",
        choices=("official", "raw"),
        default="official",
        help="Use the complete SIDReasoner CSV triplet (default) or one event-level raw file",
    )
    parser.add_argument("--official_data_root", default="data/Amazon")
    parser.add_argument("--raw_path")
    parser.add_argument("--raw_data_root", default="data/Amazon/raw")
    parser.add_argument("--sid_index")
    parser.add_argument("--sid_data_root", default="data/Amazon")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--split_strategy",
        choices=("official_temporal", "leave_last_two_out"),
        default="official_temporal",
        help="Preserve SIDReasoner's published split (default) or run the per-user ablation",
    )
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    parser.add_argument("--min_user_interactions", type=int, default=3)
    args = parser.parse_args()

    dataset = canonical_dataset(args.dataset)
    sid_path = Path(args.sid_index) if args.sid_index else resolve_sid_index(dataset, args.sid_data_root)
    default_variant = f"history_{args.max_history_len}"
    if args.split_strategy != "official_temporal":
        default_variant += f"_{args.split_strategy}"
    output = Path(args.output_dir or f"data/processed/{dataset}/{default_variant}")

    sid_map = load_sid_map(sid_path)
    reconstruction: dict[str, object] = {}
    if args.source == "official":
        if args.raw_path:
            parser.error("--raw_path can only be used with --source raw")
        split_paths = resolve_official_csv_paths(dataset, args.official_data_root)
        source_files = {split: str(path) for split, path in split_paths.items()}
        source_checksums = {split: sha256_file(path) for split, path in split_paths.items()}
        source_kind = "sidreasoner_official_csv_reconstruction"
        if args.split_strategy == "official_temporal":
            splits, counters, reconstruction = build_official_temporal_samples(
                split_paths,
                sid_map,
                dataset,
                max_history_len=args.max_history_len,
            )
        else:
            sequences, reconstruction = reconstruct_official_sequences(split_paths, sid_map)
            splits, counters = build_chronological_samples(
                interactions_from_sequences(sequences),
                sid_map,
                dataset,
                max_history_len=args.max_history_len,
                min_user_interactions=args.min_user_interactions,
            )
    else:
        if args.split_strategy != "leave_last_two_out":
            parser.error(
                "--source raw has no published SIDReasoner split; use "
                "--split_strategy leave_last_two_out explicitly"
            )
        raw_path = Path(args.raw_path) if args.raw_path else resolve_raw_path(dataset, args.raw_data_root)
        source_files = {"raw": str(raw_path)}
        source_checksums = {"raw": sha256_file(raw_path)}
        source_kind = "raw_event_interactions"
        splits, counters = build_chronological_samples(
            iter_interactions(raw_path),
            sid_map,
            dataset,
            max_history_len=args.max_history_len,
            min_user_interactions=args.min_user_interactions,
        )
    if not splits["train"]:
        if counters["mapped_sid_events"] == 0:
            raise RuntimeError(
                "No source item IDs match the SID index. SIDReasoner's index normally uses remapped "
                "numeric item IDs, while Amazon review dumps use ASINs. Use the default --source official "
                "with the complete data/Amazon/{train,valid,test} CSVs, or provide a SID index keyed by "
                "the raw file's item IDs."
            )
        if args.split_strategy == "official_temporal":
            raise RuntimeError("The published official train split contains no samples")
        raise RuntimeError(
            "No train samples remain after SID filtering and leave-last-two-out splitting; "
            f"mapped {counters['mapped_sid_events']}/{counters['total_source_events']} events, and users "
            "need at least four mapped interactions to contribute a train target"
        )

    output.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for split, records in splits.items():
        write_jsonl(output / f"{split}.jsonl", records)
        split_stats[split] = sample_length_statistics(records, args.max_history_len)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_kind": source_kind,
        "source_files": source_files,
        "source_sha256": source_checksums,
        "sid_index": str(sid_path),
        "sid_index_sha256": sha256_file(sid_path),
        "split_strategy": args.split_strategy,
        "train_sample_strategy": (
            "published_global_target_time_8_1_1"
            if args.split_strategy == "official_temporal"
            else "all_prefix_targets_before_validation"
        ),
        "test_history_includes_prior_validation_event": True,
        "max_history_len": args.max_history_len,
        "truncation_side": "oldest",
        "min_user_interactions": (
            None if args.split_strategy == "official_temporal" else args.min_user_interactions
        ),
        **reconstruction,
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
