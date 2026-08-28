"""Leak-free discrete-interest labels and token registry."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .constants import INTEREST_BEGIN, INTEREST_END, INTEREST_PAD
from .data import parse_sid_levels, sid_index


def interest_token(index: int) -> str:
    if index < 0:
        raise ValueError("Interest indices must be non-negative")
    return f"<INT_{index:03d}>"


def topk_interest_indices(
    history_sid_levels: Sequence[Sequence[str] | str],
    k: int,
    strategy: str = "frequency",
    time_decay: float = 0.1,
) -> list[int | None]:
    """Return top-k level-1 SID indices using only the supplied prefix."""

    if k < 1:
        raise ValueError("k must be positive")
    if strategy not in {"frequency", "time_decay"}:
        raise ValueError("strategy must be 'frequency' or 'time_decay'")
    if time_decay < 0:
        raise ValueError("time_decay must be non-negative")

    scores: dict[int, float] = defaultdict(float)
    total = len(history_sid_levels)
    for position, levels in enumerate(history_sid_levels):
        level1 = parse_sid_levels(levels)[0]
        index = sid_index(level1)
        weight = 1.0
        if strategy == "time_decay":
            age_from_newest = total - position - 1
            weight = math.exp(-time_decay * age_from_newest)
        scores[index] += weight
    ranked = sorted(scores, key=lambda index: (-scores[index], index))[:k]
    return ranked + [None] * (k - len(ranked))


def interest_tokens_from_history(
    history_sid_levels: Sequence[Sequence[str] | str],
    k: int,
    strategy: str = "frequency",
    time_decay: float = 0.1,
) -> list[str]:
    return [
        INTEREST_PAD if index is None else interest_token(index)
        for index in topk_interest_indices(history_sid_levels, k, strategy, time_decay)
    ]


def interest_plan_text(tokens: Sequence[str]) -> str:
    return f"{INTEREST_BEGIN}{''.join(tokens)}{INTEREST_END}"


def diprec_response(tokens: Sequence[str], target_sid: str) -> str:
    return f"<think>{interest_plan_text(tokens)}</think>{target_sid}"


def assert_prefix_only_label(record: Mapping[str, Any], label_tokens: Sequence[str]) -> None:
    expected = interest_tokens_from_history(
        record["history_sid_levels"],
        len(label_tokens),
        str(record.get("interest_strategy", "frequency")),
        float(record.get("time_decay", 0.1)),
    )
    if list(label_tokens) != expected:
        raise AssertionError(
            f"Interest label for {record.get('sample_id')} is not a function of its history prefix alone"
        )


@dataclass(frozen=True)
class TokenRegistry:
    sid_tokens: tuple[str, ...]
    interest_tokens: tuple[str, ...]
    sid_token_ids: tuple[int, ...]
    interest_token_ids: tuple[int, ...]
    interest_begin_id: int
    interest_end_id: int
    interest_pad_id: int

    def assert_disjoint(self) -> None:
        all_interest_ids = {
            self.interest_begin_id,
            self.interest_end_id,
            self.interest_pad_id,
            *self.interest_token_ids,
        }
        overlap = set(self.sid_token_ids) & all_interest_ids
        if overlap:
            raise AssertionError(f"Interest and SID token IDs overlap: {sorted(overlap)}")
        if len(all_interest_ids) != len(self.interest_token_ids) + 3:
            raise AssertionError("Interest code/control token IDs are not unique")


def _single_token_id(tokenizer: Any, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Token {token!r} does not map to exactly one tokenizer ID: {ids}")
    return int(ids[0])


def register_tokens(tokenizer: Any, model: Any, sid_map: Mapping[str, Sequence[str]]) -> TokenRegistry:
    sid_tokens = sorted({str(token) for levels in sid_map.values() for token in parse_sid_levels(levels)})
    level1_indices = sorted({sid_index(parse_sid_levels(levels)[0]) for levels in sid_map.values()})
    interest_codes = [interest_token(index) for index in level1_indices]
    all_interest = [INTEREST_BEGIN, INTEREST_END, INTEREST_PAD, *interest_codes]

    overlap = set(sid_tokens) & set(all_interest)
    if overlap:
        raise AssertionError(f"Interest token strings overlap SID tokens: {sorted(overlap)}")
    existing = tokenizer.get_vocab()
    to_add = [token for token in [*sid_tokens, *all_interest] if token not in existing]
    if to_add:
        # Keep these as ordinary added tokens. VeRL's reward manager decodes
        # with skip_special_tokens=True; marking SIDs special would erase the
        # predicted answer before reward parsing.
        tokenizer.add_tokens(to_add)
        model.resize_token_embeddings(len(tokenizer))
    registry = TokenRegistry(
        sid_tokens=tuple(sid_tokens),
        interest_tokens=tuple(interest_codes),
        sid_token_ids=tuple(_single_token_id(tokenizer, token) for token in sid_tokens),
        interest_token_ids=tuple(_single_token_id(tokenizer, token) for token in interest_codes),
        interest_begin_id=_single_token_id(tokenizer, INTEREST_BEGIN),
        interest_end_id=_single_token_id(tokenizer, INTEREST_END),
        interest_pad_id=_single_token_id(tokenizer, INTEREST_PAD),
    )
    registry.assert_disjoint()
    return registry


def register_sid_tokens(tokenizer: Any, model: Any, sid_map: Mapping[str, Sequence[str]]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    sid_tokens = tuple(sorted({str(token) for levels in sid_map.values() for token in parse_sid_levels(levels)}))
    existing = tokenizer.get_vocab()
    to_add = [token for token in sid_tokens if token not in existing]
    if to_add:
        tokenizer.add_tokens(to_add)
        model.resize_token_embeddings(len(tokenizer))
    return sid_tokens, tuple(_single_token_id(tokenizer, token) for token in sid_tokens)
