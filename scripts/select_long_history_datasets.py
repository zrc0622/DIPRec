#!/usr/bin/env python3
"""Rank Amazon categories using untruncated event-level history lengths."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diprec.constants import canonical_dataset
from diprec.data import raw_history_statistics, resolve_raw_path


def _overrides(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--raw_paths expects DATASET=PATH, got {value!r}")
        dataset, path = value.split("=", 1)
        result[canonical_dataset(dataset)] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names")
    parser.add_argument("--top_n", type=int, default=2, choices=(1, 2))
    parser.add_argument("--data_root", default="data/Amazon/raw")
    parser.add_argument("--raw_paths", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--min_user_interactions", type=int, default=3)
    parser.add_argument("--stats_output", default="outputs/history_length_stats.csv")
    parser.add_argument("--selection_output", default="configs/selected_long_history_datasets.txt")
    args = parser.parse_args()

    datasets = [canonical_dataset(name) for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        parser.error("--datasets must not be empty")
    overrides = _overrides(args.raw_paths)
    rows = []
    for dataset in datasets:
        path = overrides.get(dataset) or resolve_raw_path(dataset, args.data_root)
        stats = raw_history_statistics(path, args.min_user_interactions)
        rows.append({"dataset": dataset, "raw_file": str(path), **stats})

    rows.sort(
        key=lambda row: (
            -float(row["pct_ge_50"]),
            -float(row["p90"]),
            -float(row["mean"]),
            -int(row["effective_users"]),
            str(row["dataset"]),
        )
    )
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "dataset",
        "raw_file",
        "total_users",
        "effective_users",
        "interactions",
        "mean",
        "p50",
        "p90",
        "max",
        "pct_ge_20",
        "pct_ge_50",
        "min_user_interactions",
    ]
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            formatted = dict(row, rank=rank)
            for key in ("mean", "p50", "p90"):
                formatted[key] = f"{float(formatted[key]):.4f}"
            for key in ("pct_ge_20", "pct_ge_50"):
                formatted[key] = f"{float(formatted[key]):.8f}"
            writer.writerow(formatted)

    selected = rows[: min(args.top_n, len(rows))]
    selection_path = Path(args.selection_output)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text("".join(f"{row['dataset']}\n" for row in selected), encoding="utf-8")
    for row in rows:
        print(
            f"#{rows.index(row) + 1} {row['dataset']}: users={row['effective_users']} "
            f"mean={row['mean']:.2f} p50={row['p50']:.2f} p90={row['p90']:.2f} "
            f"max={row['max']} >=20={row['pct_ge_20']:.2%} >=50={row['pct_ge_50']:.2%}"
        )
    print(f"Selected: {', '.join(row['dataset'] for row in selected)}")


if __name__ == "__main__":
    main()
