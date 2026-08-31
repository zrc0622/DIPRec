"""Interaction IO and leak-free long-history sample construction.

SIDReasoner's released CSV rows contain ten-item sliding windows, but the
complete train/valid/test triplet contains consecutive targets.  The windows
can therefore be joined back into full per-user sequences after strict
continuity and SID checks.  Raw event files remain supported as an explicit
alternative when their item IDs already use the SID index namespace.
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
OFFICIAL_SPLITS = ("train", "valid", "test")
OFFICIAL_HISTORY_WINDOW = 10


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


def resolve_official_csv_paths(
    dataset: str,
    data_root: str | Path = "data/Amazon",
) -> dict[str, Path]:
    """Resolve one released SIDReasoner CSV for each official split."""

    category = canonical_dataset(dataset)
    root = Path(data_root)
    result: dict[str, Path] = {}
    for split in OFFICIAL_SPLITS:
        directory = root / split
        exact = directory / f"{category}_5_2016-10-2018-11.csv"
        if exact.is_file():
            result[split] = exact
            continue
        matches = sorted(directory.glob(f"{category}*.csv"))
        if len(matches) == 1:
            result[split] = matches[0]
            continue
        if not matches:
            raise FileNotFoundError(
                f"Missing official {split} CSV for {category} below {directory}. "
                "Download the complete SIDReasoner data package (train, valid, test, and index)."
            )
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple official {split} CSV candidates for {category}: {rendered}. "
            "Keep only the matching SIDReasoner release file."
        )
    return result


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


def resolve_item_metadata(dataset: str, data_root: str | Path = "data/Amazon") -> Path:
    """Resolve the item metadata paired with a released SID index."""

    category = canonical_dataset(dataset)
    root = Path(data_root)
    candidates = (
        root / "index" / f"{category}.item.json",
        root / category / f"{category}.item.json",
        Path(f"data/Amazon_{'Games' if category == 'Video_Games' else category.split('_')[0]}")
        / category
        / f"{category}.item.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Item metadata for {category} not found. Pass --item_meta; expected e.g. "
        f"data/Amazon/index/{category}.item.json"
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


def load_item_metadata(
    path: str | Path,
    sid_map: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load MiniOneRec item features and validate the SID/catalog namespace."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Item metadata must be a JSON object: {source}")
    metadata: dict[str, dict[str, Any]] = {}
    for raw_item_id, raw_features in raw.items():
        item_id = str(raw_item_id)
        if not isinstance(raw_features, Mapping):
            raise ValueError(f"Item metadata for {item_id!r} must be an object: {source}")
        title = str(raw_features.get("title") or "").strip()
        if not title:
            raise ValueError(f"Item metadata for {item_id!r} has no title: {source}")
        metadata[item_id] = {str(key): value for key, value in raw_features.items()}
        metadata[item_id]["title"] = title
    if sid_map is not None:
        missing = sorted(set(map(str, sid_map)) - set(metadata))
        if missing:
            raise ValueError(
                f"Item metadata {source} is missing {len(missing)} SID-index items; "
                f"first missing IDs: {', '.join(missing[:5])}"
            )
    return metadata


def _literal_list(value: Any, *, field: str, source: Path, line_number: int) -> list[Any]:
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{source}:{line_number}: invalid {field} list") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{source}:{line_number}: {field} must be a list")
    return parsed


def reconstruct_official_sequences(
    split_paths: Mapping[str, str | Path],
    sid_map: Mapping[str, Sequence[str]],
    history_window: int = OFFICIAL_HISTORY_WINDOW,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Reconstruct full item-ID sequences from released sliding-window CSVs.

    Files must be supplied in the original train/valid/test triplet.  For each
    user, every row's materialized history must exactly equal the suffix of the
    sequence reconstructed from earlier rows.  This makes missing, reordered,
    or mixed-release files fail instead of silently producing incorrect SIDs.
    """

    if history_window < 1:
        raise ValueError("history_window must be positive")
    missing = [split for split in OFFICIAL_SPLITS if split not in split_paths]
    if missing:
        raise ValueError(f"Official CSV set is incomplete; missing splits: {', '.join(missing)}")

    sequences: dict[str, list[str]] = {}
    rows_by_split: dict[str, int] = {}
    item_ids: set[str] = set()
    required_fields = {
        "user_id",
        "history_item_id",
        "item_id",
        "history_item_sid",
        "item_sid",
    }
    for split in OFFICIAL_SPLITS:
        source = Path(split_paths[split])
        if not source.is_file():
            raise FileNotFoundError(f"Official {split} CSV not found: {source}")
        row_count = 0
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_fields = required_fields - set(reader.fieldnames or [])
            if missing_fields:
                raise ValueError(
                    f"{source} is missing required official CSV fields: {sorted(missing_fields)}"
                )
            for line_number, row in enumerate(reader, 2):
                row_count += 1
                user_id = str(row["user_id"]).strip()
                if not user_id:
                    raise ValueError(f"{source}:{line_number}: empty user_id")
                history_ids = [
                    str(item) for item in _literal_list(
                        row["history_item_id"],
                        field="history_item_id",
                        source=source,
                        line_number=line_number,
                    )
                ]
                history_sids = [
                    str(sid) for sid in _literal_list(
                        row["history_item_sid"],
                        field="history_item_sid",
                        source=source,
                        line_number=line_number,
                    )
                ]
                target_id = str(row["item_id"]).strip()
                target_sid = str(row["item_sid"]).strip()
                if not history_ids or len(history_ids) > history_window:
                    raise ValueError(
                        f"{source}:{line_number}: expected 1..{history_window} official history items, "
                        f"got {len(history_ids)}"
                    )
                if len(history_ids) != len(history_sids):
                    raise ValueError(
                        f"{source}:{line_number}: history item/SID lengths differ "
                        f"({len(history_ids)} != {len(history_sids)})"
                    )

                for item_id, observed_sid in zip(history_ids, history_sids):
                    if item_id not in sid_map:
                        raise ValueError(f"{source}:{line_number}: history item {item_id!r} is absent from SID index")
                    expected_sid = joined_sid(sid_map[item_id])
                    if observed_sid != expected_sid:
                        raise ValueError(
                            f"{source}:{line_number}: SID mismatch for history item {item_id!r}: "
                            f"CSV={observed_sid!r}, index={expected_sid!r}"
                        )
                if target_id not in sid_map:
                    raise ValueError(f"{source}:{line_number}: target item {target_id!r} is absent from SID index")
                expected_target_sid = joined_sid(sid_map[target_id])
                if target_sid != expected_target_sid:
                    raise ValueError(
                        f"{source}:{line_number}: SID mismatch for target item {target_id!r}: "
                        f"CSV={target_sid!r}, index={expected_target_sid!r}"
                    )

                if user_id not in sequences:
                    if len(history_ids) != 1:
                        raise ValueError(
                            f"{source}:{line_number}: first row for user {user_id!r} starts with "
                            f"{len(history_ids)} history items; the official split set is incomplete or reordered"
                        )
                    sequences[user_id] = history_ids + [target_id]
                else:
                    previous = sequences[user_id]
                    expected_history = previous[-history_window:]
                    if history_ids != expected_history:
                        raise ValueError(
                            f"{source}:{line_number}: discontinuous window for user {user_id!r}; "
                            "train/valid/test may be missing, reordered, or from different releases"
                        )
                    previous.append(target_id)
                item_ids.update(history_ids)
                item_ids.add(target_id)
        rows_by_split[split] = row_count

    return sequences, {
        "official_rows_by_split": rows_by_split,
        "official_users": len(sequences),
        "official_items": len(item_ids),
        "official_history_window": history_window,
    }


def interactions_from_sequences(sequences: Mapping[str, Sequence[str]]) -> Iterator[Interaction]:
    """Yield ordered events from already reconstructed per-user sequences."""

    source_order = 0
    for user_id, items in sequences.items():
        for position, item_id in enumerate(items):
            yield Interaction(str(user_id), str(item_id), (0, position), source_order)
            source_order += 1


def official_history_statistics(
    split_paths: Mapping[str, str | Path],
    sid_map: Mapping[str, Sequence[str]],
    min_user_interactions: int = 3,
) -> dict[str, Any]:
    sequences, reconstruction = reconstruct_official_sequences(split_paths, sid_map)
    effective_lengths = [
        len(items) for items in sequences.values() if len(items) >= min_user_interactions
    ]
    stats = length_statistics(effective_lengths)
    stats.update(
        total_users=len(sequences),
        interactions=sum(len(items) for items in sequences.values()),
        min_user_interactions=min_user_interactions,
        **reconstruction,
    )
    return stats


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


def build_official_temporal_samples(
    split_paths: Mapping[str, str | Path],
    sid_map: Mapping[str, Sequence[str]],
    dataset: str,
    max_history_len: int = 50,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], dict[str, Any]]:
    """Expand histories while preserving every released split row.

    SIDReasoner creates one next-item row for every legal user prefix, sorts
    those rows globally by target time, and then publishes an 8:1:1 split.
    Reconstruction recovers the prefix hidden by its ten-item CSV window; this
    function changes only that history field and never reallocates a target.
    """

    if max_history_len < 1:
        raise ValueError("max_history_len must be positive")
    sequences, reconstruction = reconstruct_official_sequences(split_paths, sid_map)
    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in OFFICIAL_SPLITS}
    next_position: dict[str, int] = {}
    train_users: set[str] = set()

    for split in OFFICIAL_SPLITS:
        source = Path(split_paths[split])
        with source.open("r", encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), 2):
                user_id = str(row["user_id"]).strip()
                target_id = str(row["item_id"]).strip()
                target_position = next_position.get(user_id, 1)
                ordered_items = sequences[user_id]
                if target_position >= len(ordered_items) or ordered_items[target_position] != target_id:
                    raise ValueError(
                        f"{source}:{line_number}: target position changed after official "
                        "sequence validation"
                    )
                splits[split].append(
                    _sample_record(
                        dataset=dataset,
                        user_id=user_id,
                        split=split,
                        target_position=target_position,
                        ordered_items=ordered_items,
                        sid_map=sid_map,
                        max_history_len=max_history_len,
                    )
                )
                next_position[user_id] = target_position + 1
                if split == "train":
                    train_users.add(user_id)

    incomplete = [
        user_id
        for user_id, items in sequences.items()
        if next_position.get(user_id, 1) != len(items)
    ]
    if incomplete:
        raise ValueError(
            "Official targets did not consume the reconstructed sequences for users: "
            + ", ".join(sorted(incomplete)[:5])
        )
    observed_counts = {split: len(records) for split, records in splits.items()}
    if observed_counts != reconstruction["official_rows_by_split"]:
        raise ValueError(
            "Official split row counts changed during history expansion: "
            f"expected {reconstruction['official_rows_by_split']}, got {observed_counts}"
        )

    total_events = sum(len(items) for items in sequences.values())
    counters = {
        "total_source_events": total_events,
        "mapped_sid_events": total_events,
        "users_with_mapped_events": len(sequences),
        "effective_users": len(sequences),
        "train_users": len(train_users),
        "dropped_unknown_sid_events": 0,
    }
    return splits, counters, reconstruction


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
    total_events = 0
    mapped_events = 0
    dropped_unknown = 0
    for event in events:
        total_events += 1
        if event.item_id not in sid_map:
            dropped_unknown += 1
            continue
        mapped_events += 1
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
        "total_source_events": total_events,
        "mapped_sid_events": mapped_events,
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
        supported_sources = {
            "raw_event_interactions",
            "sidreasoner_official_csv_reconstruction",
        }
        if require_raw_manifest.get("source_kind") not in supported_sources:
            raise ValueError(
                "Manifest does not certify a complete event sequence or a validated "
                "SIDReasoner sliding-window reconstruction"
            )
        built_cap = int(require_raw_manifest.get("max_history_len", -1))
        if built_cap != requested_max_history_len:
            raise ValueError(f"Data were built with cap {built_cap}, requested {requested_max_history_len}")
    return sample_length_statistics(records, requested_max_history_len)


def processed_data_fingerprint(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable data identity stored with a model checkpoint."""

    return {
        "schema_version": manifest.get("schema_version"),
        "dataset": manifest.get("dataset"),
        "source_kind": manifest.get("source_kind"),
        "source_sha256": manifest.get("source_sha256"),
        "sid_index_sha256": manifest.get("sid_index_sha256"),
        "split_strategy": manifest.get("split_strategy"),
        "max_history_len": manifest.get("max_history_len"),
    }


def validate_checkpoint_training_contract(
    checkpoint: str | Path,
    *,
    expected_method: str,
    manifest: Mapping[str, Any],
    item_meta_path: str | Path | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject a dependency checkpoint trained under an incompatible contract."""

    config_path = Path(checkpoint) / "training_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} has no training_config.json; expected a completed "
            f"{expected_method} dependency"
        )
    training = json.loads(config_path.read_text(encoding="utf-8"))
    if training.get("method") != expected_method:
        raise ValueError(
            f"Checkpoint {checkpoint} was trained as {training.get('method')!r}, "
            f"expected {expected_method!r}"
        )
    expected_data = processed_data_fingerprint(manifest)
    if training.get("data_manifest") != expected_data:
        raise ValueError(f"Checkpoint {checkpoint} was trained from a different processed-data manifest")
    if item_meta_path is not None:
        actual_item_hash = sha256_file(item_meta_path)
        if training.get("item_meta_sha256") != actual_item_hash:
            raise ValueError(
                f"Checkpoint {checkpoint} was trained with different item metadata"
            )
    if expected_config:
        mismatches = {
            key: (training.get(key), expected)
            for key, expected in expected_config.items()
            if training.get(key) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={actual!r}, requested={expected!r}"
                for key, (actual, expected) in mismatches.items()
            )
            raise ValueError(
                f"Checkpoint {checkpoint} has an incompatible training configuration "
                f"({details})"
            )
    return training


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


def validate_manifest_sources(
    manifest: Mapping[str, Any],
    expected_source_kind: str | None = None,
) -> None:
    """Verify the source files recorded by a long-history manifest."""

    source_kind = manifest.get("source_kind")
    if expected_source_kind is not None and source_kind != expected_source_kind:
        raise ValueError(
            f"Data source mismatch: manifest uses {source_kind!r}, requested {expected_source_kind!r}"
        )
    source_files = manifest.get("source_files")
    source_checksums = manifest.get("source_sha256")
    if not isinstance(source_files, Mapping) or not isinstance(source_checksums, Mapping):
        # Backward compatibility for manifests written before source_files was introduced.
        raw_file = manifest.get("raw_file")
        raw_sha256 = manifest.get("raw_sha256")
        if source_kind == "raw_event_interactions" and raw_file and raw_sha256:
            source_files = {"raw": raw_file}
            source_checksums = {"raw": raw_sha256}
        else:
            raise ValueError("Long-history manifest is missing source files or checksums")
    if set(source_files) != set(source_checksums):
        raise ValueError("Long-history manifest source files and checksums do not have matching keys")
    if source_kind == "sidreasoner_official_csv_reconstruction" and set(source_files) != set(OFFICIAL_SPLITS):
        raise ValueError("Official reconstruction manifest must contain train, valid, and test sources")
    for label, value in source_files.items():
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"Long-history source file is missing ({label}): {path}")
        actual = sha256_file(path)
        expected = str(source_checksums[label])
        if actual != expected:
            raise ValueError(
                f"Source checksum mismatch for {label}: data were built with {expected}, "
                f"but {path} is {actual}. Rebuild the processed data."
            )


def validate_processed_manifest(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    max_history_len: int,
    source_kind: str,
    split_strategy: str,
    sid_index_path: str | Path,
) -> None:
    """Validate an existing processed-data manifest before reusing it."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Data schema mismatch: expected {SCHEMA_VERSION!r}, "
            f"found {manifest.get('schema_version')!r}"
        )
    expected_dataset = canonical_dataset(dataset)
    if manifest.get("dataset") != expected_dataset:
        raise ValueError(
            f"Dataset mismatch: manifest uses {manifest.get('dataset')!r}, "
            f"requested {expected_dataset!r}"
        )
    built_cap = int(manifest.get("max_history_len", -1))
    if built_cap != max_history_len:
        raise ValueError(f"Data were built with cap {built_cap}, requested {max_history_len}")
    built_strategy = manifest.get("split_strategy")
    if built_strategy != split_strategy:
        raise ValueError(
            f"Split strategy mismatch: manifest uses {built_strategy!r}, "
            f"requested {split_strategy!r}. Rebuild or select the matching processed data."
        )
    validate_manifest_sources(manifest, source_kind)
    validate_manifest_sid_index(manifest, sid_index_path)
