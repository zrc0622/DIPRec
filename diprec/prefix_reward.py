"""Bounded, centered SID-prefix assistance for all-miss recommendation groups."""

from __future__ import annotations

import math
import re
from typing import Sequence

REWARD_MODES = ("official", "main_miss_prefix")
RL_TASKS = ("history_sid_to_sid", "title_to_sid", "description_to_sid", "title_history_to_sid")
_SID = re.compile(r"(<a_\d+>)(<b_\d+>)(<c_\d+>)")


def validate_prefix_reward(mode: str, strength: float) -> None:
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode}")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("prefix_reward_strength must be finite and in [0, 1]")


def prefix_group_signal(
    completions: Sequence[str], targets: Sequence[str], tasks: Sequence[str],
    num_generations: int, strength: float,
) -> tuple[list[float], list[dict]]:
    """Return additive advantages and diagnostics for complete contiguous groups.

    Call AFTER official group normalization and BEFORE microbatch shuffling.
    No group std division is applied to this signal: strength bounds its magnitude.
    Invalid/malformed catalog SIDs fail closed instead of earning partial credit.
    """
    validate_prefix_reward("main_miss_prefix", strength)
    if num_generations < 2 or len(completions) % num_generations:
        raise ValueError("Prefix assistance requires complete G-sized groups")
    if not len(completions) == len(targets) == len(tasks):
        raise ValueError("Prefix completion/target/task counts differ")

    def parse(value: str) -> tuple[str, str, str]:
        canonical = "".join(value.strip('\n" ').split())
        match = _SID.fullmatch(canonical)
        if match is None:
            raise ValueError(f"Expected a complete three-level catalog SID: {value!r}")
        return match.groups()

    deltas, groups = [], []
    for start in range(0, len(completions), num_generations):
        end = start + num_generations
        if len(set(targets[start:end])) != 1 or len(set(tasks[start:end])) != 1:
            raise ValueError("A prefix reward group contains different targets or tasks")
        task = tasks[start]
        if task not in RL_TASKS:
            raise ValueError(f"Unknown RL task for prefix assistance: {task!r}")
        target = parse(targets[start])
        candidates = [parse(value) for value in completions[start:end]]
        hit = target in candidates
        scores = [0.5 * (c[:1] == target[:1]) + 0.5 * (c[:2] == target[:2]) for c in candidates]
        mean_score = sum(scores) / num_generations
        eligible = task == "history_sid_to_sid" and not hit
        aux = [strength * (h - mean_score) if eligible else 0.0 for h in scores]
        deltas.extend(aux)
        groups.append({
            "task": task, "exact_hit": hit, "eligible": eligible,
            "prefix_informative": eligible and max(scores) > min(scores),
            "aux_active": max(aux) > min(aux),
            "prefix_mean": mean_score,
            "aux_abs_mean": sum(abs(x) for x in aux) / num_generations,
            "aux_abs_max": max(abs(x) for x in aux),
        })
    return deltas, groups
