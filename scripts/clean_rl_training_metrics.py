#!/usr/bin/env python3
"""Remove per-step training entries from existing RL metrics JSON files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


FILENAME = "rl_training_metrics.json"


def evaluation_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("evaluations", payload.get("log_history", []))
    if not isinstance(history, list):
        raise ValueError("expected 'evaluations' or 'log_history' to be a list")
    return [
        entry
        for entry in history
        if isinstance(entry, dict)
        and any(str(key).startswith("eval_") for key in entry)
    ]


def cleaned_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": payload.get("status", "unknown"),
    }
    if "max_steps" in payload:
        result["max_steps"] = payload["max_steps"]
    result["evaluations"] = evaluation_entries(payload)
    return result


def discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob(FILENAME))
    raise FileNotFoundError(path)


def rewrite(path: Path, *, backup: bool = True, dry_run: bool = False) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    result = cleaned_payload(payload)
    removed = len(payload.get("log_history", payload.get("evaluations", []))) - len(
        result["evaluations"]
    )
    if dry_run:
        return removed
    if backup:
        backup_path = path.with_name(f"{path.name}.bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only eval_* records in one RL metrics file or every "
            f"{FILENAME} below a directory."
        )
    )
    parser.add_argument("path", type=Path, help="Metrics JSON file or outputs directory")
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying files")
    parser.add_argument(
        "--no-backup", action="store_true", help="Do not create rl_training_metrics.json.bak"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = discover(args.path)
    if not files:
        raise SystemExit(f"No {FILENAME} files found under {args.path}")
    total_removed = 0
    for path in files:
        removed = rewrite(path, backup=not args.no_backup, dry_run=args.dry_run)
        total_removed += removed
        action = "would clean" if args.dry_run else "cleaned"
        print(f"{action}: {path} (removed {removed} non-eval entries)")
    print(f"files={len(files)} removed_entries={total_removed}")


if __name__ == "__main__":
    main()
