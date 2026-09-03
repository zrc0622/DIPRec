"""Leak-free discrete-interest labels and token registry."""

from __future__ import annotations

import math
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
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


def interest_activation_plan_pool(
    history_sid_levels: Sequence[Sequence[str] | str],
    k: int,
    max_plans: int = 8,
    strategy: str = "frequency",
    time_decay: float = 0.1,
) -> list[list[str]]:
    """Build the compact history-only plan pool used by interest activation.

    Unlike the legacy diverse-label builder, this function does not enumerate
    arbitrary subsets merely to reach ``max_plans``.  It keeps a configured
    aggregate plan, a genuinely recent-interest plan when its content differs,
    and singleton plans for observed interests.  Plans are deduplicated by
    their non-padding interest *set*, so permutations do not count as new
    supervision and histories with little evidence naturally return fewer
    plans.
    """

    if k < 1:
        raise ValueError("k must be positive")
    if max_plans < 1:
        raise ValueError("max_plans must be positive")
    if strategy not in {"frequency", "time_decay"}:
        raise ValueError("strategy must be 'frequency' or 'time_decay'")
    if time_decay < 0:
        raise ValueError("time_decay must be non-negative")
    if not history_sid_levels:
        raise ValueError("Interest activation requires a non-empty history")

    frequency_scores: dict[int, int] = defaultdict(int)
    latest: dict[int, int] = {}
    for position, levels in enumerate(history_sid_levels):
        index = sid_index(parse_sid_levels(levels)[0])
        frequency_scores[index] += 1
        latest[index] = position
    frequency_ranked = sorted(
        frequency_scores,
        key=lambda index: (-frequency_scores[index], index),
    )
    recent_ranked = sorted(latest, key=lambda index: (-latest[index], index))

    plans: list[list[str]] = []
    seen_content: set[tuple[int, ...]] = set()

    def add(indices: Sequence[int | None]) -> None:
        if len(plans) >= max_plans:
            return
        actual = [int(index) for index in indices if index is not None]
        if not actual:
            return
        content_key = tuple(sorted(set(actual)))
        if content_key in seen_content:
            return
        seen_content.add(content_key)
        # All plan-generation prompts retain the exact-k grammar.  Padding is
        # used only where a meaningful candidate (not combinatorial filling)
        # contains fewer than k observed interests.
        ordered_unique = list(dict.fromkeys(actual))[:k]
        padded: list[int | None] = ordered_unique + [None] * (k - len(ordered_unique))
        plans.append(
            [
                INTEREST_PAD if index is None else interest_token(index)
                for index in padded
            ]
        )

    # Preserve the configured legacy primary plan as pool index zero.  With
    # the default frequency strategy this is the aggregate-frequency label;
    # time-decay ablations retain their former primary label.
    add(topk_interest_indices(history_sid_levels, k, strategy, time_decay))
    # Ensure that a frequency aggregate remains represented even in a
    # time-decay ablation, then add a recent-distinct-interest view.
    add(frequency_ranked[:k])
    add(recent_ranked[:k])
    # Single-interest labels activate each substantial observed region without
    # manufacturing every possible subset.  Frequency order is deterministic;
    # recent-only ordering is appended to cover ties before the max-plans cap.
    for index in [*frequency_ranked, *recent_ranked]:
        add([index])
        if len(plans) == max_plans:
            break
    return plans


def select_interest_activation_plan(
    plans: Sequence[Sequence[str]],
    mode: str,
    epoch: int,
    seed: int,
    sample_id: str,
) -> tuple[int, list[str]]:
    """Select one plan deterministically for a record and training epoch.

    Diverse mode traverses a seeded shuffle without replacement.  Once every
    plan has been used, the next cycle is independently reshuffled.  Selection
    depends only on explicit stable inputs, never Python's randomized hash.
    """

    if not plans:
        raise ValueError("plans must not be empty")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if mode == "single":
        return 0, list(plans[0])
    if mode != "diverse":
        raise ValueError("SFT plan mode must be 'single' or 'diverse'")
    cycle, position = divmod(epoch, len(plans))
    material = f"{seed}\0{sample_id}\0{cycle}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = list(range(len(plans)))
    random.Random(stable_seed).shuffle(order)
    selected = order[position]
    return selected, list(plans[selected])


def diverse_interest_plans_from_history(
    history_sid_levels: Sequence[Sequence[str] | str],
    k: int,
    max_plans: int = 8,
    strategy: str = "frequency",
    time_decay: float = 0.1,
) -> list[list[str]]:
    """Build deterministic, content-distinct plan labels from history only.

    The first plan is exactly the legacy top-k label. Additional labels are
    different subsets of level-1 interests observed in the same prefix; mere
    permutations are deduplicated. If fewer than ``k`` interests are selected,
    shorter observed subsets are padded rather than inventing unseen interests.
    """

    if max_plans < 1:
        raise ValueError("max_plans must be positive")
    primary = topk_interest_indices(history_sid_levels, k, strategy, time_decay)
    observed = [index for index in primary if index is not None]

    scores: dict[int, float] = defaultdict(float)
    latest: dict[int, int] = {}
    total = len(history_sid_levels)
    for position, levels in enumerate(history_sid_levels):
        index = sid_index(parse_sid_levels(levels)[0])
        weight = 1.0
        if strategy == "time_decay":
            weight = math.exp(-time_decay * (total - position - 1))
        scores[index] += weight
        latest[index] = position
    ranked = sorted(scores, key=lambda index: (-scores[index], -latest[index], index))

    plans: list[list[str]] = []
    seen_content: set[tuple[int, ...]] = set()

    def add_plan(indices: Sequence[int | None]) -> bool:
        actual = [int(index) for index in indices if index is not None]
        content_key = tuple(sorted(actual))
        if content_key in seen_content:
            return False
        seen_content.add(content_key)
        padded: list[int | None] = actual + [None] * (k - len(actual))
        plans.append(
            [
                INTEREST_PAD if index is None else interest_token(index)
                for index in padded
            ]
        )
        return len(plans) == max_plans

    if add_plan(primary):
        return plans
    width = min(k, len(ranked))
    if len(ranked) > k:
        recent = sorted(ranked, key=lambda index: (-latest[index], index))[:k]
        if add_plan(recent):
            return plans
    for subset_width in range(width, 0, -1):
        for subset_positions in combinations(range(len(ranked)), subset_width):
            if add_plan([ranked[position] for position in subset_positions]):
                return plans
    return plans


def interest_plans_from_history(
    history_sid_levels: Sequence[Sequence[str] | str],
    k: int,
    mode: str = "single",
    max_plans: int = 8,
    strategy: str = "frequency",
    time_decay: float = 0.1,
) -> list[list[str]]:
    if mode == "single":
        return [interest_tokens_from_history(history_sid_levels, k, strategy, time_decay)]
    if mode == "diverse":
        return diverse_interest_plans_from_history(
            history_sid_levels, k, max_plans, strategy, time_decay
        )
    raise ValueError("SFT plan mode must be 'single' or 'diverse'")


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
