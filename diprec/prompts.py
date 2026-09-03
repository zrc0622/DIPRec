"""Shared prompts for baselines and DIPRec."""

from __future__ import annotations

from typing import Mapping, Sequence

from .constants import DATASET_LABELS, INTEREST_BEGIN, INTEREST_END, canonical_dataset

SYSTEM = (
    "You are a recommendation model. Follow the requested output grammar exactly; "
    "do not invent item identifiers."
)


def _history(record: Mapping[str, object], max_history_len: int) -> list[str]:
    values = list(record["history_item_sid"])  # type: ignore[arg-type]
    if len(values) > max_history_len:
        values = values[-max_history_len:]
    if len(values) != int(record["history_len"]):
        raise ValueError(
            f"Sample {record.get('sample_id')} history has {record['history_len']} items but "
            f"the prompt retained {len(values)}; rebuild data with the requested cap"
        )
    return values


def history_prompt(record: Mapping[str, object], max_history_len: int, reasoning: bool = False) -> str:
    history = ", ".join(_history(record, max_history_len))
    category = DATASET_LABELS[canonical_dataset(str(record["dataset"]))]
    suffix = (
        "Reason step by step inside <think>...</think>, then output exactly one three-level SID."
        if reasoning
        else "Output exactly one valid three-level SID and no explanation."
    )
    return (
        f"The user interacted with these {category} in chronological order: {history}. "
        f"Recommend the next item. {suffix}"
    )


def title_to_sid_prompt(title: str) -> str:
    return f'Which item has the title: "{title}"? Output exactly one valid three-level SID.'


def description_to_sid_prompt(description: str) -> str:
    return (
        f'An item is described as follows: "{description}". '
        "Which item is it? Output exactly one valid three-level SID."
    )


def sid_to_title_prompt(sid: str) -> str:
    return f'What is the title of item "{sid}"? Output only its title.'


def history_to_title_prompt(record: Mapping[str, object], max_history_len: int) -> str:
    history = ", ".join(_history(record, max_history_len))
    return (
        f"The user interacted with these item SIDs in chronological order: {history}. "
        "Recommend the next item and output only its title."
    )


def title_history_to_sid_prompt(
    record: Mapping[str, object],
    item_metadata: Mapping[str, Mapping[str, object]],
    max_history_len: int,
) -> str:
    retained_ids = list(record["history_item_id"])[-max_history_len:]  # type: ignore[arg-type]
    titles = [f'"{item_metadata[str(item_id)]["title"]}"' for item_id in retained_ids]
    return (
        "Given the title sequence of the user's historical items: "
        f"{', '.join(titles)}, recommend the next item. "
        "Output exactly one valid three-level SID."
    )


def plan_prompt(record: Mapping[str, object], max_history_len: int, interest_topk: int) -> str:
    history = ", ".join(_history(record, max_history_len))
    return (
        f"The user's chronological item SID history is: {history}. "
        f"Generate exactly {interest_topk} discrete interest tokens. Do not produce natural language. "
        f"Use the grammar <think>{INTEREST_BEGIN}<INT_xxx>...{INTEREST_END}</think>."
    )


def joint_trajectory_prompt(
    record: Mapping[str, object], max_history_len: int, interest_topk: int
) -> str:
    history = ", ".join(_history(record, max_history_len))
    return (
        f"The user's chronological item SID history is: {history}. "
        f"Generate exactly {interest_topk} discrete interest tokens inside <think>...</think>, "
        "then output exactly one valid three-level item SID. Do not produce natural language. "
        f"Use the grammar <think>{INTEREST_BEGIN}<INT_xxx>...{INTEREST_END}</think>"
        "<a_xxx><b_xxx><c_xxx>."
    )


def sid_prompt(
    record: Mapping[str, object],
    plan_tokens: Sequence[str],
    max_history_len: int,
    conditioning: str,
) -> str:
    plan = f"{INTEREST_BEGIN}{''.join(plan_tokens)}{INTEREST_END}"
    if conditioning == "history_visible":
        history = ", ".join(_history(record, max_history_len))
        return (
            f"History: {history}. Discrete interest plan: {plan}. "
            "Output exactly one valid three-level item SID."
        )
    if conditioning == "interest_bottleneck":
        result = f"Discrete interest plan: {plan}. Output exactly one valid three-level item SID."
        for history_sid in record["history_item_sid"]:  # type: ignore[index]
            if str(history_sid) in result:
                raise AssertionError("Strict interest bottleneck leaked a history SID into the SID prompt")
        return result
    raise ValueError("conditioning must be history_visible or interest_bottleneck")


def messages(user_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}]
