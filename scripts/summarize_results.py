#!/usr/bin/env python3
"""Collect the unified metrics JSON files into one comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/")
    parser.add_argument("--output", default="outputs/comparison.csv")
    args = parser.parse_args()
    root = Path(args.input)
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "diprec.metrics.v1":
            continue
        metrics = payload.get("metrics", {})
        config = payload.get("training_config", {})
        rows.append(
            {
                "dataset": payload.get("dataset"),
                "method": payload.get("method"),
                "model": payload.get("model"),
                "seed": payload.get("seed"),
                "num_examples": payload.get("num_examples"),
                "Recall@5": metrics.get("Recall@5"),
                "Recall@10": metrics.get("Recall@10"),
                "NDCG@5": metrics.get("NDCG@5"),
                "NDCG@10": metrics.get("NDCG@10"),
                "sid_valid_rate": metrics.get("sid_valid_rate"),
                "interest_diversity": metrics.get("interest_diversity"),
                "sid_level1_hit": metrics.get("sid_level1_hit"),
                "sid_level2_hit": metrics.get("sid_level2_hit"),
                "sid_level3_hit": metrics.get("sid_level3_hit"),
                "max_history_len": config.get("max_history_len"),
                "max_seq_len": config.get("max_seq_len"),
                "interest_topk": config.get("interest_topk"),
                "num_plans": config.get("num_plans"),
                "sid_beams": config.get("sid_beams"),
                "eval_beams": config.get("eval_beams"),
                "eval_candidate_budget": config.get("eval_candidate_budget"),
                "conditioning": config.get("conditioning"),
                "result_file": str(path),
            }
        )
    if not rows:
        raise SystemExit(f"No diprec.metrics.v1 files found below {root}")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} result rows to {destination}")


if __name__ == "__main__":
    main()
