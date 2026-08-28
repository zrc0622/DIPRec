"""Plan/candidate rewards, metrics and hierarchical advantages."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


def sid_level_hits(candidate: Sequence[str], target: Sequence[str]) -> tuple[int, int, int]:
    values = []
    prefix_ok = True
    for level in range(3):
        prefix_ok = prefix_ok and len(candidate) > level and len(target) > level and candidate[level] == target[level]
        values.append(int(prefix_ok))
    return tuple(values)  # type: ignore[return-value]


def ranking_metrics(candidates: Sequence[Sequence[str]], target: Sequence[str], cutoffs: Sequence[int] = (5, 10)) -> dict[str, float]:
    exact_rank = next((rank for rank, value in enumerate(candidates, 1) if list(value) == list(target)), None)
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        result[f"Recall@{cutoff}"] = float(exact_rank is not None and exact_rank <= cutoff)
        result[f"NDCG@{cutoff}"] = (1.0 / math.log2(exact_rank + 1)) if exact_rank and exact_rank <= cutoff else 0.0
    return result


def duplicate_rate(candidates: Sequence[Sequence[str]]) -> float:
    if not candidates:
        return 0.0
    return 1.0 - len({tuple(value) for value in candidates}) / len(candidates)


@dataclass(frozen=True)
class RewardWeights:
    hr: float = 1.0
    ndcg: float = 1.0
    level1: float = 0.1
    level2: float = 0.2
    level3: float = 0.4
    valid: float = 0.1
    duplicate: float = 0.1


def score_plan(
    candidates: Sequence[Sequence[str]],
    target: Sequence[str],
    valid: Sequence[bool],
    weights: RewardWeights = RewardWeights(),
) -> tuple[float, list[float], dict[str, float]]:
    if len(candidates) != len(valid):
        raise ValueError("Candidate and validity counts differ")
    metrics = ranking_metrics(candidates, target)
    level_hits = [sid_level_hits(candidate, target) for candidate in candidates]
    valid_rate = sum(valid) / len(valid) if valid else 0.0
    repeat = duplicate_rate(candidates)
    best_levels = [max((hits[level] for hits in level_hits), default=0) for level in range(3)]
    plan_reward = (
        weights.hr * metrics["Recall@10"]
        + weights.ndcg * metrics["NDCG@10"]
        + weights.level1 * best_levels[0]
        + weights.level2 * best_levels[1]
        + weights.level3 * best_levels[2]
        + weights.valid * valid_rate
        - weights.duplicate * repeat
    )
    candidate_rewards = [
        weights.level1 * hits[0]
        + weights.level2 * hits[1]
        + weights.level3 * hits[2]
        + (
            weights.hr + weights.ndcg / math.log2(rank + 1)
            if list(candidate) == list(target)
            else 0.0
        )
        + weights.valid * float(is_valid)
        for rank, (candidate, hits, is_valid) in enumerate(zip(candidates, level_hits, valid), 1)
    ]
    details = {
        **metrics,
        "sid_level1_hit": float(best_levels[0]),
        "sid_level2_hit": float(best_levels[1]),
        "sid_level3_hit": float(best_levels[2]),
        "sid_valid_rate": valid_rate,
        "duplicate_rate": repeat,
        "plan_reward": plan_reward,
    }
    return plan_reward, candidate_rewards, details


def _standardize(values: Sequence[float], epsilon: float = 1e-6) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= epsilon * epsilon:
        return [0.0] * len(values)
    scale = math.sqrt(variance) + epsilon
    return [(value - mean) / scale for value in values]


def hierarchical_advantages(
    plan_rewards: Sequence[float],
    candidate_rewards: Sequence[Sequence[float]],
    mode: str = "plan_grpo",
    epsilon: float = 1e-6,
) -> tuple[list[float], list[list[float]]]:
    if len(plan_rewards) != len(candidate_rewards):
        raise ValueError("Expected one candidate-reward list per plan")
    if mode == "plan_grpo":
        return _standardize(plan_rewards, epsilon), [_standardize(values, epsilon) for values in candidate_rewards]
    if mode == "trajectory_grpo":
        flat_rewards = []
        locations = []
        for plan_index, rewards in enumerate(candidate_rewards):
            for candidate_index, reward in enumerate(rewards):
                flat_rewards.append(plan_rewards[plan_index] + reward)
                locations.append((plan_index, candidate_index))
        flat_advantages = _standardize(flat_rewards, epsilon)
        nested = [[0.0 for _ in values] for values in candidate_rewards]
        plan_accumulator: dict[int, list[float]] = defaultdict(list)
        for advantage, (plan_index, candidate_index) in zip(flat_advantages, locations):
            nested[plan_index][candidate_index] = advantage
            plan_accumulator[plan_index].append(advantage)
        plan_advantages = [
            sum(plan_accumulator[index]) / len(plan_accumulator[index]) if plan_accumulator[index] else 0.0
            for index in range(len(plan_rewards))
        ]
        return plan_advantages, nested
    raise ValueError("mode must be plan_grpo or trajectory_grpo")


def token_advantage_mask(
    response_token_ids: Sequence[int],
    interest_token_ids: Iterable[int],
    sid_token_ids: Iterable[int],
    plan_advantage: float,
    candidate_advantage: float,
    interest_loss_weight: float,
    sid_loss_weight: float,
) -> list[float]:
    interest_set = set(map(int, interest_token_ids))
    sid_set = set(map(int, sid_token_ids))
    if interest_set & sid_set:
        raise AssertionError("Interest and SID token masks overlap")
    return [
        plan_advantage * interest_loss_weight
        if int(token_id) in interest_set
        else candidate_advantage * sid_loss_weight
        if int(token_id) in sid_set
        else 0.0
        for token_id in response_token_ids
    ]


def aggregate_metric_rows(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    collected: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            collected[key].append(float(value))
    return {key: sum(values) / len(values) for key, values in collected.items() if values}


def interest_diversity(plans: Sequence[Sequence[str]]) -> float:
    tokens = [token for plan in plans for token in plan if token != "<INT_PAD>"]
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def select_unique_plans(plans: Iterable[Sequence[str]], count: int) -> list[list[str]]:
    unique = []
    seen = set()
    for plan in plans:
        key = tuple(plan)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(plan))
        if len(unique) == count:
            return unique
    raise RuntimeError(f"Only sampled {len(unique)} distinct plans; required {count}")
