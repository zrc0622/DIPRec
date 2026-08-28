"""Catalog-aware SIDReasoner reward for any selected category.

The index path is supplied via ``DIPREC_SID_INDEX`` by the launcher. This
replaces the three legacy category-specific hard-coded reward files without
altering them.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from diprec.data import load_sid_map, parse_sid_levels
from diprec.rewards import sid_level_hits

SID_RE = re.compile(r"<[^<>]+>")


@lru_cache(maxsize=None)
def _catalog(index_path: str | None = None) -> set[tuple[str, str, str]]:
    index_path = index_path or os.environ.get("DIPREC_SID_INDEX")
    if not index_path:
        raise RuntimeError("DIPREC_SID_INDEX must point to the selected dataset's .index.json")
    return {parse_sid_levels(levels) for levels in load_sid_map(index_path).values()}


def parse_response(solution: str) -> tuple[str, str, str] | None:
    match = re.search(r"</think>\s*(.*)", solution, re.DOTALL)
    if match is None:
        return None
    answer = match.group(1)
    tokens = SID_RE.findall(answer)
    return tuple(tokens[:3]) if len(tokens) >= 3 else None  # type: ignore[return-value]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, sid_index=None):
    del data_source, extra_info
    candidate = parse_response(solution_str)
    target = parse_sid_levels(ground_truth)
    if candidate is None:
        return {"score": 0.0, "valid": 0.0, "level1": 0.0, "level2": 0.0, "level3": 0.0}
    valid = float(candidate in _catalog(sid_index))
    hits = sid_level_hits(candidate, target)
    score = 0.1 * valid + 0.1 * hits[0] + 0.2 * hits[1] + 1.0 * hits[2]
    return {
        "score": score,
        "valid": valid,
        "level1": float(hits[0]),
        "level2": float(hits[1]),
        "level3": float(hits[2]),
    }
