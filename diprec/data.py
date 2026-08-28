"""Raw interaction IO and leak-free long-history sample construction.

The functions in this module intentionally do not accept SIDReasoner's legacy
history CSV as a raw source.  Those files already contain materialized (and
often ten-item-truncated) histories, so they cannot support dataset selection
or a genuine 50-item experiment.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .constants import SCHEMA_VERSION, canonical_dataset

USER_FIELDS = ("user_id", "reviewerID", "reviewer_id", "user", "uid")
ITEM_FIELDS = ("item_id", "asin", "parent_asin", "item", "iid")
TIME_FIELDS = ("timestamp", "unixReviewTime", "unix_review_time", "time", "date")
FORBIDDEN_HISTORY_FIELDS = {
    "history_item_id",
    "history_item_sid",
    "history_item_title",
    "history",
}
SID_RE = re.compile(r"<[^<>]+>")
INDEX_RE = re.compile(r"(-?\d+)(?!.*\d)")


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: tuple[int, Any]
    source_order: int


def _pick(record: Mapping[str, Any], fields: Sequence[str], kind: str) -> Any:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip() != "":
            return value
    raise ValueError(f"Raw interaction has no {kind} field; tried {fields}: {record}")


def _sortable_time(value: Any, source_order: int) -> tuple[int, Any]:
    if value is None or str(value).strip() == "":
        return (2, source_order)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _parse_json_line(line: str) -> Mapping[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        # Older Amazon dumps are Python-dict literals rather than strict JSON.
        value = ast.literal_eval(line)
    if not isinstance(value, Mapping):
        raise ValueError("Each raw JSONL line must be an object")
    return value


def iter_raw_records(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Yield event-level records from JSONL/JSON/CSV/TSV, optionally gzipped."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Raw interaction file not found: {source}")
    name = source.name.lower()
    with _open_text(source) as handle:
        if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
            delimiter = "\t" if ".tsv" in name else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            fields = set(reader.fieldnames or [])
            bad = fields & FORBIDDEN_HISTORY_FIELDS
            if bad:
                raise ValueError(
                    f"{source} is a materialized history table ({sorted(bad)}), not an "
                    "untruncated event-level interaction file"
                )
            yield from reader
            return

        first_nonempty = None
        buffered: list[str] = []
        for line in handle:
            if line.strip():
                first_nonempty = line
                buffered.append(line)
                break
        if first_nonempty is None:
            return
        if first_nonempty.lstrip().startswith("["):
            rest = "".join(buffered) + handle.read()
            values = json.loads(rest)
            if not isinstance(values, list):
                raise ValueError(f"Expected a JSON array in {source}")
            for value in values:
                if not isinstance(value, Mapping):
                    raise ValueError(f"JSON array entries in {source} must be objects")
                yield value
        else:
            for line in buffered:
                yield _parse_json_line(line)
            for line in handle:
                if line.strip():
                    yield _parse_json_line(line)


def iter_interactions(path: str | Path) -> Iterator[Interaction]:
    for order, record in enumerate(iter_raw_records(path)):
        bad = set(record) & FORBIDDEN_HISTORY_FIELDS
        if bad:
            raise ValueError(
                f"{path} contains materialized history fields {sorted(bad)}; "
                "dataset selection and long-history construction require event-level interactions"
            )
        user = str(_pick(record, USER_FIELDS, "user id"))
        item = str(_pick(record, ITEM_FIELDS, "item id"))
        raw_time = next((record.get(key) for key in TIME_FIELDS if record.get(key) is not None), None)
        yield Interaction(user, item, _sortable_time(raw_time, order), order)


def resolve_raw_path(dataset: str, data_root: str | Path = "data/Amazon/raw") -> Path:
    """Resolve a category's event file without ever falling back to history CSVs."""

    category = canonical_dataset(dataset)
    root = Path(data_root)
    stems = (category, f"{category}_5", f"reviews_{category}")
    suffixes = (".jsonl.gz", ".json.gz", ".jsonl", ".json", ".csv.gz", ".csv", ".tsv.gz", ".tsv")
    candidates = [root / f"{stem}{suffix}" for stem in stems for suffix in suffixes]
    for path in candidates:
        if path.is_file():
            return path
    rendered = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No untruncated raw interaction file found for {category}. Tried:\n  - {rendered}\n"
        "Pass --raw_path/--raw_paths explicitly if your layout differs."
    )


def resolve_sid_index(dataset: str, data_root: str | Path = "data/Amazon") -> Path:
    category = canonical_dataset(dataset)
    root = Path(data_root)
    candidates = (
        root / "index" / f"{category}.index.json",
        root / category / f"{category}.index.json",
        Path(f"data/Amazon_{'Games' if category == 'Video_Games' else category.split('_')[0]}")
        / category
        / f"{category}.index.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"SID index for {category} not found. Pass --sid_index; expected e.g. "
        f"data/Amazon/index/{category}.index.json"
    )


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def length_statistics(lengths: Sequence[int]) -> dict[str, float | int]:
    if not lengths:
        return {
            "effective_users": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0,
            "pct_ge_20": 0.0,
            "pct_ge_50": 0.0,
        }
    count = len(lengths)
    return {
        "effective_users": count,
        "mean": sum(lengths) / count,
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "max": max(lengths),
        "pct_ge_20": sum(length >= 20 for length in lengths) / count,
        "pct_ge_50": sum(length >= 50 for length in lengths) / count,
    }


def raw_history_statistics(path: str | Path, min_user_interactions: int = 3) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    interactions = 0
    for event in iter_interactions(path):
        counts[event.user_id] += 1
        interactions += 1
    effective_lengths = [length for length in counts.values() if length >= min_user_interactions]
    stats = length_statistics(effective_lengths)
    stats.update(
        total_users=len(counts),
        interactions=interactions,
        min_user_interactions=min_user_interactions,
    )
    return stats


def parse_sid_levels(value: Any) -> tuple[str, str, str]:
    if isinstance(value, str):
        tokens = SID_RE.findall(value)
        if len(tokens) < 3:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                tokens = [str(token) for token in parsed]
    elif isinstance(value, (list, tuple)):
        tokens = [str(token) for token in value]
    else:
        tokens = []
    if len(tokens) < 3:
        raise ValueError(f"Expected three SID levels, got {value!r}")
    return tuple(tokens[:3])  # type: ignore[return-value]


def load_sid_map(path: str | Path) -> dict[str, tuple[str, str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"SID index must be a JSON object: {path}")
    return {str(item): parse_sid_levels(levels) for item, levels in raw.items()}


def sid_index(level1_sid: str) -> int:
    match = INDEX_RE.search(level1_sid)
    if not match:
        raise ValueError(f"Cannot extract numeric level-1 SID index from {level1_sid!r}")
    return int(match.group(1))


def joined_sid(levels: Sequence[str]) -> str:
    if len(levels) != 3:
        raise ValueError(f"A target SID must have exactly three levels, got {levels}")
    return "".join(levels)


def _sample_record(
    *,
    dataset: str,
    user_id: str,
    split: str,
    target_position: int,
    ordered_items: Sequence[str],
    sid_map: Mapping[str, Sequence[str]],
    max_history_len: int,
) -> dict[str, Any]:
    # This slice is the leakage boundary: target and every later event are absent.
    full_prefix = list(ordered_items[:target_position])
    retained = full_prefix[-max_history_len:]
    target_item = ordered_items[target_position]
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{dataset}:{user_id}:{target_position}",
        "dataset": dataset,
        "user_id": user_id,
        "split": split,
        "target_position": target_position,
        "history_item_id": retained,
        "history_item_sid": [joined_sid(sid_map[item]) for item in retained],
        "history_sid_levels": [list(sid_map[item]) for item in retained],
        "target_item_id": target_item,
        "target_item_sid": joined_sid(sid_map[target_item]),
        "target_sid_levels": list(sid_map[target_item]),
        "history_len_before_truncation": len(full_prefix),
        "history_len": len(retained),
        "max_history_len": max_history_len,
    }


def build_chronological_samples(
    events: Iterable[Interaction],
    sid_map: Mapping[str, Sequence[str]],
    dataset: str,
    max_history_len: int = 50,
    min_user_interactions: int = 3,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Leave-last-two-out split, then create targets from each legal prefix.

    Train targets end before the validation target.  Validation targets the
    penultimate event; test targets the last event and may observe the earlier
    validation event, as is standard in chronological next-item evaluation.
    """

    if max_history_len < 1:
        raise ValueError("max_history_len must be positive")
    users: dict[str, list[Interaction]] = defaultdict(list)
    dropped_unknown = 0
    for event in events:
        if event.item_id not in sid_map:
            dropped_unknown += 1
            continue
        users[event.user_id].append(event)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    effective_users = 0
    train_users = 0
    for user_id in sorted(users):
        ordered_events = sorted(users[user_id], key=lambda event: (event.timestamp, event.source_order))
        items = [event.item_id for event in ordered_events]
        if len(items) < min_user_interactions:
            continue
        effective_users += 1
        train_positions = list(range(1, len(items) - 2))
        if train_positions:
            train_users += 1
        for position in train_positions:
            splits["train"].append(
                _sample_record(
                    dataset=dataset,
                    user_id=user_id,
                    split="train",
                    target_position=position,
                    ordered_items=items,
                    sid_map=sid_map,
                    max_history_len=max_history_len,
                )
            )
        for split, position in (("valid", len(items) - 2), ("test", len(items) - 1)):
            splits[split].append(
                _sample_record(
                    dataset=dataset,
                    user_id=user_id,
                    split=split,
                    target_position=position,
                    ordered_items=items,
                    sid_map=sid_map,
                    max_history_len=max_history_len,
                )
            )
    counters = {
        "users_with_mapped_events": len(users),
        "effective_users": effective_users,
        "train_users": train_users,
        "dropped_unknown_sid_events": dropped_unknown,
    }
    return splits, counters


def sample_length_statistics(records: Sequence[Mapping[str, Any]], cap: int) -> dict[str, Any]:
    before = [int(record["history_len_before_truncation"]) for record in records]
    after = [len(record["history_item_id"]) for record in records]
    return {
        "samples": len(records),
        "before": length_statistics(before),
        "after": length_statistics(after),
        "fraction_reaching_cap": (sum(length >= cap for length in before) / len(before)) if before else 0.0,
        "fraction_reaching_50": (sum(length >= 50 for length in before) / len(before)) if before else 0.0,
    }


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                records.append(value)
    return records


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_history_records(
    records: Sequence[Mapping[str, Any]],
    requested_max_history_len: int,
    require_raw_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Dataset split is empty")
    for record in records:
        history_ids = record.get("history_item_id")
        history_sids = record.get("history_item_sid")
        if not isinstance(history_ids, list) or not isinstance(history_sids, list):
            raise ValueError(f"Malformed history in sample {record.get('sample_id')}")
        if len(history_ids) != len(history_sids) or len(history_ids) != int(record.get("history_len", -1)):
            raise ValueError(f"History length fields disagree in sample {record.get('sample_id')}")
        if len(history_ids) > requested_max_history_len:
            raise ValueError(
                f"Sample {record.get('sample_id')} has {len(history_ids)} history items, "
                f"above requested cap {requested_max_history_len}"
            )
        before = int(record.get("history_len_before_truncation", -1))
        if before < len(history_ids):
            raise ValueError(f"Invalid pre-truncation length in sample {record.get('sample_id')}")
    if require_raw_manifest is not None:
        if require_raw_manifest.get("source_kind") != "raw_event_interactions":
            raise ValueError("Manifest does not certify an untruncated event-level source")
        built_cap = int(require_raw_manifest.get("max_history_len", -1))
        if built_cap != requested_max_history_len:
            raise ValueError(f"Data were built with cap {built_cap}, requested {requested_max_history_len}")
    return sample_length_statistics(records, requested_max_history_len)


def validate_manifest_sid_index(manifest: Mapping[str, Any], sid_index_path: str | Path) -> None:
    expected = manifest.get("sid_index_sha256")
    if not expected:
        raise ValueError("Long-history manifest is missing sid_index_sha256")
    actual = sha256_file(sid_index_path)
    if actual != expected:
        raise ValueError(
            f"SID index checksum mismatch: data were built with {expected}, but {sid_index_path} is {actual}. "
            "All methods must use the exact same SID mapping; rebuild or pass the matching file."
        )
