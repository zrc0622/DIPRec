"""SFT data encoding and a compact Accelerate training loop."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import (
    joined_sid,
    load_item_metadata,
    load_sid_map,
    processed_data_fingerprint,
    read_jsonl,
    sha256_file,
    validate_history_records,
    validate_checkpoint_training_contract,
    validate_manifest_sid_index,
)
from .interest import (
    assert_prefix_only_label,
    diprec_response,
    interest_activation_plan_pool,
    interest_plan_text,
    interest_plans_from_history,
    interest_tokens_from_history,
    select_interest_activation_plan,
)
from .prompts import (
    history_prompt,
    history_to_title_prompt,
    joint_trajectory_prompt,
    messages,
    plan_prompt,
    sid_prompt,
    sid_to_title_prompt,
    title_to_sid_prompt,
)
from .runtime import apply_chat_template, load_model_runtime, save_runtime, set_seed, thinking_prompt_ids

SFT_METHOD_ALIASES = {
    "direct_sid": "direct_sft",
    "direct_sft": "direct_sft",
    "minionerec_sft": "minionerec_sft",
    "diprec_sft": "diprec_sft",
    # Retained for the original SIDReasoner-compatible entrypoint.
    "sidreasoner_sft": "sidreasoner_sft",
}

ACTIVATION_OBJECTIVES = {"interest_activation", "joint_interest_activation"}


def canonical_sft_method(method: str) -> str:
    try:
        return SFT_METHOD_ALIASES[method]
    except KeyError as exc:
        raise ValueError(
            "SFT method must be direct_sft, minionerec_sft, diprec_sft, "
            "or a supported legacy alias"
        ) from exc


def catalog_alignment_maps(
    item_metadata: Mapping[str, Mapping[str, Any]],
    sid_map: Mapping[str, Sequence[str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Reproduce MiniOneRec's last-write-wins SID/title dictionaries."""

    sid_to_title: dict[str, str] = {}
    title_to_sid: dict[str, str] = {}
    for item_id, sid_levels in sid_map.items():
        if item_id not in item_metadata:
            raise ValueError(f"SID-index item {item_id!r} is absent from item metadata")
        sid = joined_sid(sid_levels)
        title = str(item_metadata[item_id]["title"])
        sid_to_title[sid] = title
        title_to_sid[title] = sid
    return sid_to_title, title_to_sid


def response_for_record(
    record: Mapping[str, Any],
    method: str,
    interest_topk: int,
    interest_strategy: str,
    time_decay: float,
) -> tuple[str, list[str]]:
    method = canonical_sft_method(method)
    target_sid = str(record["target_item_sid"])
    if method in {"direct_sft", "minionerec_sft"}:
        return target_sid, []
    if method == "sidreasoner_sft":
        interests = interest_tokens_from_history(
            record["history_sid_levels"], interest_topk, interest_strategy, time_decay
        )
        observed = ", ".join(
            f"level-1 semantic group {token.removeprefix('<INT_').removesuffix('>')}"
            for token in interests
            if token != "<INT_PAD>"
        )
        reasoning = (
            "The chronological history repeatedly reflects the coarse semantic groups "
            f"{observed}. I will prioritize those observed preferences while selecting a catalog item."
        )
        return f"<think>{reasoning}</think>{target_sid}", interests
    if method != "diprec_sft":
        raise ValueError("Unsupported SFT method")
    label_record = dict(record, interest_strategy=interest_strategy, time_decay=time_decay)
    tokens = interest_tokens_from_history(record["history_sid_levels"], interest_topk, interest_strategy, time_decay)
    assert_prefix_only_label(label_record, tokens)
    return diprec_response(tokens, target_sid), tokens


def _encode_pair(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_len: int,
    metadata: Mapping[str, Any],
    thinking: bool = False,
) -> dict[str, Any]:
    if thinking:
        prompt_ids = thinking_prompt_ids(tokenizer, messages(prompt))
        if response.startswith("<think>"):
            response = response[len("<think>") :]
    else:
        prompt_ids = apply_chat_template(tokenizer, messages(prompt), add_generation_prompt=True, enable_thinking=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids.append(int(tokenizer.eos_token_id))
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"Sample {metadata.get('sample_id')} has {len(input_ids)} tokens (> {max_seq_len}); "
            "increase --max_seq_len or rebuild with a smaller --max_history_len"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
        **metadata,
    }


def encode_sft_records(
    tokenizer: Any,
    record: Mapping[str, Any],
    method: str,
    max_history_len: int,
    max_seq_len: int,
    interest_topk: int,
    interest_strategy: str,
    time_decay: float,
    conditioning: str,
    item_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    sid_to_title: Mapping[str, str] | None = None,
    sft_plan_mode: str = "single",
    sft_num_plans: int = 8,
) -> list[dict[str, Any]]:
    method = canonical_sft_method(method)
    response, interest_tokens = response_for_record(
        record, method, interest_topk, interest_strategy, time_decay
    )
    metadata = {"sample_id": record.get("sample_id"), "interest_tokens": interest_tokens}
    if method in {"direct_sft", "sidreasoner_sft"}:
        prompt = history_prompt(record, max_history_len, reasoning=method == "sidreasoner_sft")
        return [
            _encode_pair(
                tokenizer,
                prompt,
                response,
                max_seq_len,
                metadata | {"stage": method},
                thinking=method == "sidreasoner_sft",
            )
        ]
    if method == "minionerec_sft":
        if item_metadata is None:
            raise ValueError("MiniOneRec-SFT requires item metadata")
        target_id = str(record["target_item_id"])
        if target_id not in item_metadata:
            raise ValueError(f"Target item {target_id!r} is absent from item metadata")
        target_title = (
            str(sid_to_title[str(record["target_item_sid"])])
            if sid_to_title is not None
            else str(item_metadata[target_id]["title"])
        )
        return [
            _encode_pair(
                tokenizer,
                history_prompt(record, max_history_len, reasoning=False),
                response,
                max_seq_len,
                metadata | {"stage": "history_sid_to_sid"},
            ),
            _encode_pair(
                tokenizer,
                history_to_title_prompt(record, max_history_len),
                target_title,
                max_seq_len,
                metadata | {"stage": "history_sid_to_title"},
            ),
        ]

    plans = interest_plans_from_history(
        record["history_sid_levels"],
        interest_topk,
        sft_plan_mode,
        sft_num_plans,
        interest_strategy,
        time_decay,
    )
    rows = []
    for plan_index, plan_tokens in enumerate(plans):
        plan_metadata = {
            "sample_id": record.get("sample_id"),
            "interest_tokens": plan_tokens,
            "plan_index": plan_index,
            "plan_count": len(plans),
        }
        plan_response = diprec_response(plan_tokens, str(record["target_item_sid"]))
        plan_response = plan_response[: plan_response.index("</think>") + len("</think>")]
        rows.append(
            _encode_pair(
                tokenizer,
                plan_prompt(record, max_history_len, interest_topk),
                plan_response,
                max_seq_len,
                plan_metadata | {"stage": "interest_plan"},
                thinking=True,
            )
        )
    # The target must not be used to decide which alternative history-only
    # plans are relevant. Pairing that same target with every alternative
    # would instead teach the decoder to ignore the plan. Preserve the legacy
    # primary-plan SID task and let RL assign utility to sampled alternatives.
    primary_tokens = plans[0]
    rows.append(
        _encode_pair(
            tokenizer,
            sid_prompt(record, primary_tokens, max_history_len, conditioning),
            str(record["target_item_sid"]),
            max_seq_len,
            {
                "sample_id": record.get("sample_id"),
                "interest_tokens": primary_tokens,
                "plan_index": 0,
                "plan_count": len(plans),
                "stage": "sid_prediction",
            },
        )
    )
    return rows


def encode_interest_activation_pair(
    tokenizer: Any,
    record: Mapping[str, Any],
    plan_tokens: Sequence[str],
    plan_index: int,
    plan_count: int,
    max_history_len: int,
    max_seq_len: int,
    interest_topk: int,
) -> dict[str, dict[str, Any]]:
    """Encode the balanced plan/SID pair for one history-only plan.

    The plan label is determined before this helper is called and therefore
    cannot inspect the future target.  The SID stage deliberately keeps the
    history visible: activation SFT teaches the model to use a sampled interest
    view without pretending that a heuristic plan is the sole cause of the
    observed future item.
    """

    tokens = list(plan_tokens)
    metadata = {
        "sample_id": record.get("sample_id"),
        "interest_tokens": tokens,
        "plan_index": int(plan_index),
        "plan_count": int(plan_count),
        "sft_objective": "interest_activation",
    }
    plan_response = f"<think>{interest_plan_text(tokens)}</think>"
    return {
        "plan": _encode_pair(
            tokenizer,
            plan_prompt(record, max_history_len, interest_topk),
            plan_response,
            max_seq_len,
            metadata | {"stage": "interest_plan"},
            thinking=True,
        ),
        "sid": _encode_pair(
            tokenizer,
            sid_prompt(record, tokens, max_history_len, "history_visible"),
            str(record["target_item_sid"]),
            max_seq_len,
            metadata | {"stage": "sid_prediction"},
        ),
    }


def encode_joint_interest_activation_trajectory(
    tokenizer: Any,
    record: Mapping[str, Any],
    plan_tokens: Sequence[str],
    plan_index: int,
    plan_count: int,
    max_history_len: int,
    max_seq_len: int,
    interest_topk: int,
) -> dict[str, Any]:
    """Encode one history -> plan -> target-SID autoregressive trajectory."""

    tokens = list(plan_tokens)
    prompt_ids = thinking_prompt_ids(
        tokenizer,
        messages(joint_trajectory_prompt(record, max_history_len, interest_topk)),
    )
    # ``thinking_prompt_ids`` already ends at the opening <think> marker.
    plan_ids = tokenizer.encode(
        f"{interest_plan_text(tokens)}</think>", add_special_tokens=False
    )
    sid_ids = tokenizer.encode(str(record["target_item_sid"]), add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        sid_ids.append(int(tokenizer.eos_token_id))
    input_ids = [*prompt_ids, *plan_ids, *sid_ids]
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"Sample {record.get('sample_id')} has {len(input_ids)} tokens (> {max_seq_len}); "
            "increase --max_seq_len or rebuild with a smaller --max_history_len"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + plan_ids + sid_ids,
        # Token-aligned stages make it possible to optimize one joint forward
        # without allowing the longer segment to receive more aggregate weight.
        "token_stage_ids": [-1] * len(prompt_ids)
        + [0] * len(plan_ids)
        + [1] * len(sid_ids),
        "sample_id": record.get("sample_id"),
        "interest_tokens": tokens,
        "plan_index": int(plan_index),
        "plan_count": int(plan_count),
        "sft_objective": "joint_interest_activation",
        "stage": "joint_plan_sid_trajectory",
    }


def encode_catalog_sft_records(
    tokenizer: Any,
    item_metadata: Mapping[str, Mapping[str, Any]],
    sid_map: Mapping[str, Sequence[str]],
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """Build MiniOneRec's bidirectional title/SID alignment tasks."""

    sid_to_title, title_to_sid = catalog_alignment_maps(item_metadata, sid_map)
    rows = [
        _encode_pair(
            tokenizer,
            sid_to_title_prompt(sid),
            title,
            max_seq_len,
            {"sample_id": f"catalog:{sid}:sid_to_title", "stage": "sid_to_title"},
        )
        for sid, title in sid_to_title.items()
    ]
    rows.extend(
        _encode_pair(
            tokenizer,
            title_to_sid_prompt(title),
            sid,
            max_seq_len,
            {"sample_id": f"catalog:{title}:title_to_sid", "stage": "title_to_sid"},
        )
        for title, sid in title_to_sid.items()
    )
    return rows


def encode_sft_record(
    tokenizer: Any,
    record: Mapping[str, Any],
    method: str,
    max_history_len: int,
    max_seq_len: int,
    interest_topk: int,
    interest_strategy: str,
    time_decay: float,
    conditioning: str = "interest_bottleneck",
    item_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    sid_to_title: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compatibility helper returning the first encoded task for a record."""

    return encode_sft_records(
        tokenizer,
        record,
        method,
        max_history_len,
        max_seq_len,
        interest_topk,
        interest_strategy,
        time_decay,
        conditioning,
        item_metadata,
        sid_to_title,
    )[0]


class _EncodedDataset:
    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class InterestActivationDataset:
    """Epoch-aware paired supervision for the activation objective."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_history_len: int,
        max_seq_len: int,
        interest_topk: int,
        interest_strategy: str,
        time_decay: float,
        sft_plan_mode: str,
        sft_num_plans: int,
        seed: int,
        rotate: bool,
    ):
        self.records = [dict(record) for record in records]
        self.tokenizer = tokenizer
        self.max_history_len = max_history_len
        self.max_seq_len = max_seq_len
        self.interest_topk = interest_topk
        self.sft_plan_mode = sft_plan_mode
        self.seed = seed
        self.rotate = rotate
        self.epoch = 0
        self.plan_pools = [
            interest_activation_plan_pool(
                record["history_sid_levels"],
                interest_topk,
                sft_num_plans,
                interest_strategy,
                time_decay,
            )
            for record in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def selected_plan(self, index: int, epoch: int | None = None) -> tuple[int, list[str]]:
        record = self.records[index]
        selection_epoch = (
            self.epoch if epoch is None else int(epoch)
        ) if self.rotate else 0
        sample_id = str(record.get("sample_id", index))
        return select_interest_activation_plan(
            self.plan_pools[index],
            self.sft_plan_mode,
            selection_epoch,
            self.seed,
            sample_id,
        )

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        record = self.records[index]
        plan_index, plan = self.selected_plan(index)
        return encode_interest_activation_pair(
            self.tokenizer,
            record,
            plan,
            plan_index,
            len(self.plan_pools[index]),
            self.max_history_len,
            self.max_seq_len,
            self.interest_topk,
        )


class JointInterestActivationDataset(InterestActivationDataset):
    """Epoch-aware one-sequence trajectories using the activation plan pool."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        plan_index, plan = self.selected_plan(index)
        return encode_joint_interest_activation_trajectory(
            self.tokenizer,
            record,
            plan,
            plan_index,
            len(self.plan_pools[index]),
            self.max_history_len,
            self.max_seq_len,
            self.interest_topk,
        )


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(row["input_ids"]) for row in rows)
        input_ids, attention_mask, labels = [], [], []
        for row in rows:
            pad = width - len(row["input_ids"])
            input_ids.append(list(row["input_ids"]) + [self.pad_token_id] * pad)
            attention_mask.append(list(row["attention_mask"]) + [0] * pad)
            labels.append(list(row["labels"]) + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class InterestActivationCollator:
    """Flatten record pairs into an exactly balanced mixed sequence batch."""

    def __init__(self, pad_token_id: int):
        self.causal = CausalCollator(pad_token_id)

    def __call__(self, pairs: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
        import torch

        plan_rows = [pair["plan"] for pair in pairs]
        sid_rows = [pair["sid"] for pair in pairs]
        batch = self.causal([*plan_rows, *sid_rows])
        batch["stage_ids"] = torch.tensor(
            [0] * len(plan_rows) + [1] * len(sid_rows), dtype=torch.long
        )
        return batch


class JointInterestActivationCollator:
    """Pad joint trajectories and retain their token-aligned stage masks."""

    def __init__(self, pad_token_id: int):
        self.causal = CausalCollator(pad_token_id)

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        batch = self.causal(rows)
        width = batch["input_ids"].shape[1]
        stage_ids = [
            list(row["token_stage_ids"])
            + [-1] * (width - len(row["token_stage_ids"]))
            for row in rows
        ]
        batch["stage_ids"] = torch.tensor(stage_ids, dtype=torch.long)
        return batch


def _causal_stage_losses(logits: Any, labels: Any, stage_ids: Any) -> tuple[Any, Any, int, int]:
    """Return token-mean plan and SID losses from paired or joint batches."""

    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    def stage_loss(stage: int) -> tuple[Any, int]:
        if stage_ids.ndim == 1:
            selected_labels = shifted_labels[stage_ids == stage]
            selected_logits = shifted_logits[stage_ids == stage]
        elif stage_ids.ndim == 2:
            if stage_ids.shape != labels.shape:
                raise ValueError("Token stage IDs must have the same shape as labels")
            selected = (stage_ids[:, 1:] == stage) & shifted_labels.ne(-100)
            selected_labels = shifted_labels[selected]
            selected_logits = shifted_logits[selected]
        else:
            raise ValueError("Stage IDs must be sequence-aligned or token-aligned")
        token_count = int(selected_labels.ne(-100).sum().item())
        if token_count == 0:
            raise RuntimeError(f"Activation batch has no supervised tokens for stage {stage}")
        loss = functional.cross_entropy(
            selected_logits.view(-1, selected_logits.size(-1)),
            selected_labels.view(-1),
            ignore_index=-100,
        )
        return loss, token_count

    plan_loss, plan_tokens = stage_loss(0)
    sid_loss, sid_tokens = stage_loss(1)
    return plan_loss, sid_loss, plan_tokens, sid_tokens


def _manifest_for(split_path: Path) -> dict[str, Any]:
    path = split_path.parent / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing long-history manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_loss(model: Any, loader: Any, accelerator: Any) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            output = model(**batch)
            losses.append(accelerator.gather(output.loss.detach().reshape(1)))
    model.train()
    if not losses:
        return float("nan")
    return torch.cat(losses).float().mean().item()


def _evaluate_activation_losses(
    model: Any, loader: Any, accelerator: Any
) -> tuple[float, float, float]:
    """Evaluate both activation stages and their equal-weight objective."""

    import torch

    model.eval()
    plan_numerator = torch.zeros((), device=accelerator.device)
    sid_numerator = torch.zeros((), device=accelerator.device)
    plan_denominator = torch.zeros((), dtype=torch.long, device=accelerator.device)
    sid_denominator = torch.zeros((), dtype=torch.long, device=accelerator.device)
    with torch.no_grad():
        for batch in loader:
            stage_ids = batch.pop("stage_ids")
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            plan_loss, sid_loss, plan_tokens, sid_tokens = _causal_stage_losses(
                output.logits, batch["labels"], stage_ids
            )
            plan_numerator += plan_loss.detach().float() * plan_tokens
            sid_numerator += sid_loss.detach().float() * sid_tokens
            plan_denominator += plan_tokens
            sid_denominator += sid_tokens
    gathered = accelerator.gather(
        torch.stack(
            [
                plan_numerator,
                sid_numerator,
                plan_denominator.float(),
                sid_denominator.float(),
            ]
        )
    ).reshape(-1, 4).sum(dim=0)
    model.train()
    if gathered[2].item() == 0 or gathered[3].item() == 0:
        return float("nan"), float("nan"), float("nan")
    plan_loss = (gathered[0] / gathered[2]).item()
    sid_loss = (gathered[1] / gathered[3]).item()
    return plan_loss, sid_loss, 0.5 * (plan_loss + sid_loss)


def _write_training_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist SFT epoch summaries outside the checkpoint directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _plan_pool_statistics(dataset: InterestActivationDataset) -> dict[str, Any]:
    sizes = [len(pool) for pool in dataset.plan_pools]
    histogram = Counter(sizes)
    return {
        "records": len(sizes),
        "minimum_pool_size": min(sizes) if sizes else 0,
        "maximum_pool_size": max(sizes) if sizes else 0,
        "mean_pool_size": sum(sizes) / len(sizes) if sizes else 0.0,
        "pool_size_histogram": {
            str(size): histogram[size] for size in sorted(histogram)
        },
        "records_with_multiple_plans": sum(size > 1 for size in sizes),
    }


def _sampled_plan_examples(
    train_dataset: InterestActivationDataset,
    valid_dataset: InterestActivationDataset,
    num_epochs: int,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, dataset in (("train", train_dataset), ("valid", valid_dataset)):
        for index, record in enumerate(dataset.records[:limit]):
            epochs = range(num_epochs) if split == "train" else range(1)
            selections = []
            for epoch in epochs:
                plan_index, plan = dataset.selected_plan(index, epoch)
                selections.append(
                    {"epoch": epoch + 1, "plan_index": plan_index, "plan": plan}
                )
            rows.append(
                {
                    "split": split,
                    "sample_id": record.get("sample_id"),
                    "plan_pool": dataset.plan_pools[index],
                    "selections": selections,
                }
            )
    return rows


def _training_config(
    args: argparse.Namespace,
    method: str,
    manifest: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    valid_stats: Mapping[str, Any],
    *,
    checkpoint_role: str,
    selected_epoch: int,
    selected_validation_loss: float,
) -> dict[str, Any]:
    checkpoint_selection_metric = (
        "valid_sid_loss"
        if method == "diprec_sft" and args.sft_objective in ACTIVATION_OBJECTIVES
        else "validation_loss"
    )
    return vars(args) | {
        "method": method,
        "item_meta_sha256": sha256_file(args.item_meta) if args.item_meta else None,
        "train_history": train_stats,
        "valid_history": valid_stats,
        "data_manifest": processed_data_fingerprint(manifest),
        "checkpoint_role": checkpoint_role,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "selected_epoch": selected_epoch,
        "selected_validation_loss": selected_validation_loss,
    }


def _checkpoint_selection_loss(
    activation: bool,
    validation_loss: float,
    valid_sid_loss: float | None = None,
) -> float:
    """Return the loss used to rank SFT checkpoints.

    Interest-activation SFT keeps its balanced plan/SID training objective, but
    selects checkpoints by SID validation loss because downstream recommendation
    quality depends on SID prediction. Legacy objectives retain their existing
    aggregate validation-loss behavior.
    """
    if not activation:
        return validation_loss
    if valid_sid_loss is None:
        raise ValueError("valid_sid_loss is required for an activation objective")
    return valid_sid_loss


def _save_sft_checkpoint(
    model: Any,
    tokenizer: Any,
    router: Any,
    accelerator: Any,
    destination: Path,
    args: argparse.Namespace,
    method: str,
    manifest: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    valid_stats: Mapping[str, Any],
    *,
    checkpoint_role: str,
    selected_epoch: int,
    selected_validation_loss: float,
) -> None:
    unwrapped = accelerator.unwrap_model(model)
    save_runtime(
        unwrapped,
        tokenizer,
        router,
        destination,
        args.interest_parameterization,
    )
    config = _training_config(
        args,
        method,
        manifest,
        train_stats,
        valid_stats,
        checkpoint_role=checkpoint_role,
        selected_epoch=selected_epoch,
        selected_validation_loss=selected_validation_loss,
    )
    (destination / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.sft_num_plans < 1:
        raise ValueError("sft_num_plans must be positive")
    method = canonical_sft_method(args.method)
    activation = method == "diprec_sft" and args.sft_objective in ACTIVATION_OBJECTIVES
    paired_activation = activation and args.sft_objective == "interest_activation"
    joint_activation = activation and args.sft_objective == "joint_interest_activation"
    if args.sft_objective in ACTIVATION_OBJECTIVES and method != "diprec_sft":
        raise ValueError("Activation objectives are only supported by --method diprec_sft")
    if activation and args.conditioning != "history_visible":
        raise ValueError(
            f"{args.sft_objective} requires --conditioning history_visible so heuristic plans "
            "do not hide the evidence used for target-SID supervision"
        )
    if paired_activation and (args.micro_batch_size < 2 or args.micro_batch_size % 2):
        raise ValueError(
            "interest_activation requires an even --micro_batch_size of at least 2; half of each mixed "
            "batch is plan supervision and half is SID supervision"
        )
    if args.plan_example_limit < 0:
        raise ValueError("plan_example_limit must be non-negative")
    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)
    manifest = _manifest_for(train_path)
    train_records = read_jsonl(train_path)
    valid_records = read_jsonl(valid_path)
    train_stats = validate_history_records(train_records, args.max_history_len, manifest)
    valid_stats = validate_history_records(valid_records, args.max_history_len, manifest)
    validate_manifest_sid_index(manifest, args.sid_index)
    sid_map = load_sid_map(args.sid_index)
    item_metadata = None
    sid_to_title = None
    title_to_sid = None
    if method == "minionerec_sft":
        if not args.item_meta:
            raise ValueError("MiniOneRec-SFT requires --item_meta")
        item_metadata = load_item_metadata(args.item_meta, sid_map)
        sid_to_title, title_to_sid = catalog_alignment_maps(item_metadata, sid_map)
    elif method == "diprec_sft":
        if not args.item_meta:
            raise ValueError("DIPRec-SFT requires --item_meta to validate its MiniOneRec-SFT parent")
        if not args.dry_run:
            validate_checkpoint_training_contract(
                args.model,
                expected_method="minionerec_sft",
                manifest=manifest,
                item_meta_path=args.item_meta,
            )
    if args.dry_run:
        train_samples = len(train_records)
        valid_samples = len(valid_records)
        task_counts = {"history_sid_to_sid": len(train_records)}
        if method == "minionerec_sft":
            assert sid_to_title is not None and title_to_sid is not None
            train_samples = 2 * len(train_records) + len(sid_to_title) + len(title_to_sid)
            task_counts = {
                "history_sid_to_sid": len(train_records),
                "title_to_sid": len(title_to_sid),
                "sid_to_title": len(sid_to_title),
                "history_sid_to_title": len(train_records),
            }
        elif method == "diprec_sft":
            if activation:
                train_pools = [
                    interest_activation_plan_pool(
                        record["history_sid_levels"],
                        args.interest_topk,
                        args.sft_num_plans,
                        args.interest_strategy,
                        args.time_decay,
                    )
                    for record in train_records
                ]
                pool_sizes = [len(pool) for pool in train_pools]
                if paired_activation:
                    train_samples = 2 * len(train_records)
                    valid_samples = 2 * len(valid_records)
                    task_counts = {
                        "interest_plan": len(train_records),
                        "sid_prediction": len(train_records),
                    }
                else:
                    train_samples = len(train_records)
                    valid_samples = len(valid_records)
                    task_counts = {
                        "joint_plan_sid_trajectory": len(train_records),
                    }
            else:
                train_plan_count = sum(
                    len(interest_plans_from_history(
                        record["history_sid_levels"], args.interest_topk,
                        args.sft_plan_mode, args.sft_num_plans,
                        args.interest_strategy, args.time_decay,
                    ))
                    for record in train_records
                )
                valid_plan_count = sum(
                    len(interest_plans_from_history(
                        record["history_sid_levels"], args.interest_topk,
                        args.sft_plan_mode, args.sft_num_plans,
                        args.interest_strategy, args.time_decay,
                    ))
                    for record in valid_records
                )
                train_samples = train_plan_count + len(train_records)
                valid_samples = valid_plan_count + len(valid_records)
                task_counts = {
                    "interest_plan": train_plan_count,
                    "sid_prediction": len(train_records),
                }
        print(
            json.dumps(
                {
                    "method": method,
                    "train_samples": train_samples,
                    "valid_samples": valid_samples,
                    "task_counts": task_counts,
                    "train_history": train_stats,
                    "valid_history": valid_stats,
                    "catalog_items": len(sid_map),
                    "item_metadata": len(item_metadata) if item_metadata is not None else 0,
                    "model": args.model,
                    "sft_objective": args.sft_objective,
                    "interest_parameterization": args.interest_parameterization,
                    "conditioning": args.conditioning,
                    "sft_plan_mode": args.sft_plan_mode,
                    "sft_num_plans": args.sft_num_plans,
                    "plan_pool_statistics": (
                        {
                            "minimum": min(pool_sizes) if pool_sizes else 0,
                            "maximum": max(pool_sizes) if pool_sizes else 0,
                            "mean": sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0.0,
                        }
                        if activation
                        else None
                    ),
                },
                indent=2,
            )
        )
        return
    try:
        import torch
        from accelerate import Accelerator
        from torch.utils.data import DataLoader
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Training requires torch, transformers, and accelerate") from exc
    model, tokenizer, registry, router = load_model_runtime(
        args.model,
        sid_map,
        args.interest_parameterization,
        training=True,
        include_interest=method == "diprec_sft",
    )

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    generator = torch.Generator().manual_seed(args.seed)
    activation_train_dataset = None
    activation_valid_dataset = None
    if activation:
        activation_dataset_class = (
            InterestActivationDataset
            if paired_activation
            else JointInterestActivationDataset
        )
        activation_train_dataset = activation_dataset_class(
            train_records,
            tokenizer,
            max_history_len=args.max_history_len,
            max_seq_len=args.max_seq_len,
            interest_topk=args.interest_topk,
            interest_strategy=args.interest_strategy,
            time_decay=args.time_decay,
            sft_plan_mode=args.sft_plan_mode,
            sft_num_plans=args.sft_num_plans,
            seed=args.seed,
            rotate=True,
        )
        activation_valid_dataset = activation_dataset_class(
            valid_records,
            tokenizer,
            max_history_len=args.max_history_len,
            max_seq_len=args.max_seq_len,
            interest_topk=args.interest_topk,
            interest_strategy=args.interest_strategy,
            time_decay=args.time_decay,
            sft_plan_mode=args.sft_plan_mode,
            sft_num_plans=args.sft_num_plans,
            seed=args.seed,
            rotate=False,
        )
        activation_collator = (
            InterestActivationCollator(tokenizer.pad_token_id)
            if paired_activation
            else JointInterestActivationCollator(tokenizer.pad_token_id)
        )
        records_per_batch = (
            args.micro_batch_size // 2 if paired_activation else args.micro_batch_size
        )
        train_loader = DataLoader(
            activation_train_dataset,
            batch_size=records_per_batch,
            shuffle=True,
            collate_fn=activation_collator,
            generator=generator,
        )
        valid_loader = DataLoader(
            activation_valid_dataset,
            batch_size=records_per_batch,
            shuffle=False,
            collate_fn=activation_collator,
        )
    else:
        train_rows = [
            row
            for record in train_records
            for row in encode_sft_records(
                tokenizer,
                record,
                method,
                args.max_history_len,
                args.max_seq_len,
                args.interest_topk,
                args.interest_strategy,
                args.time_decay,
                args.conditioning,
                item_metadata,
                sid_to_title,
                args.sft_plan_mode,
                args.sft_num_plans,
            )
        ]
        if method == "minionerec_sft":
            assert item_metadata is not None
            train_rows.extend(
                encode_catalog_sft_records(tokenizer, item_metadata, sid_map, args.max_seq_len)
            )
        valid_rows = [
            row
            for record in valid_records
            for row in encode_sft_records(
                tokenizer,
                record,
                "direct_sft" if method == "minionerec_sft" else method,
                args.max_history_len,
                args.max_seq_len,
                args.interest_topk,
                args.interest_strategy,
                args.time_decay,
                args.conditioning,
                item_metadata,
                sid_to_title,
                args.sft_plan_mode,
                args.sft_num_plans,
            )
        ]
        collator = CausalCollator(tokenizer.pad_token_id)
        train_loader = DataLoader(
            _EncodedDataset(train_rows),
            batch_size=args.micro_batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=generator,
        )
        valid_loader = DataLoader(
            _EncodedDataset(valid_rows),
            batch_size=args.micro_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, update_steps)
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    metrics_path = (
        Path(args.training_metrics_file)
        if args.training_metrics_file
        else Path(args.output_dir).parent / "sft_training_metrics.json"
    )
    best_output_dir = (
        Path(args.best_output_dir)
        if args.best_output_dir
        else Path(args.output_dir).with_name("best_checkpoint")
    )
    training_metrics: dict[str, Any] = {
        "status": "running",
        "method": method,
        "model": args.model,
        "sft_objective": args.sft_objective,
        "checkpoint_selection_metric": (
            "valid_sid_loss" if activation else "validation_loss"
        ),
        "hyperparameters": {
            "num_epochs": args.num_epochs,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size_per_process": args.micro_batch_size
            * args.gradient_accumulation_steps,
            "world_size": accelerator.num_processes,
            "effective_global_batch_size": args.micro_batch_size
            * args.gradient_accumulation_steps
            * accelerator.num_processes,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_seq_len": args.max_seq_len,
            "sft_objective": args.sft_objective,
            "sft_plan_mode": args.sft_plan_mode,
            "sft_num_plans": args.sft_num_plans,
            "conditioning": args.conditioning,
        },
        "best_checkpoint": str(best_output_dir),
        "final_checkpoint": str(args.output_dir),
        "best_epoch": None,
        "best_validation_loss": None,
        "best_checkpoint_selection_loss": None,
        "epochs": [],
    }
    if accelerator.is_main_process:
        if activation:
            assert activation_train_dataset is not None
            assert activation_valid_dataset is not None
            artifact_dir = metrics_path.parent
            pool_statistics_path = artifact_dir / "plan_pool_statistics.json"
            plan_examples_path = artifact_dir / "sampled_plan_examples.jsonl"
            pool_statistics = {
                "schema_version": "diprec.sft_plan_pool.v1",
                "objective": args.sft_objective,
                "mode": args.sft_plan_mode,
                "maximum_plans": args.sft_num_plans,
                "train": _plan_pool_statistics(activation_train_dataset),
                "valid": _plan_pool_statistics(activation_valid_dataset),
            }
            _write_training_metrics(pool_statistics_path, pool_statistics)
            _write_jsonl(
                plan_examples_path,
                _sampled_plan_examples(
                    activation_train_dataset,
                    activation_valid_dataset,
                    args.num_epochs,
                    args.plan_example_limit,
                ),
            )
            training_metrics["artifacts"] = {
                "plan_pool_statistics": str(pool_statistics_path),
                "sampled_plan_examples": str(plan_examples_path),
            }
        _write_training_metrics(metrics_path, training_metrics)
    accelerator.wait_for_everyone()
    # Accelerate may shard the loader after preparation. The scheduler remains
    # configured from the pre-prepare global data-loader length (and Accelerate
    # adjusts its stepping under replicated DDP), while this value records the
    # optimizer steps observed by each process.
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        print(
            "sft_config "
            f"epochs={args.num_epochs} micro_batch={args.micro_batch_size} "
            f"accumulation={args.gradient_accumulation_steps} "
            f"global_batch={args.micro_batch_size * args.gradient_accumulation_steps * accelerator.num_processes} "
            f"learning_rate={args.learning_rate:g} warmup_steps={warmup_steps} "
            f"scheduler_total_steps={update_steps} "
            f"optimizer_steps_per_process={optimizer_steps_per_epoch * args.num_epochs}",
            flush=True,
        )
    model.train()
    optimizer_steps = 0
    best_validation_loss = math.inf
    best_epoch: int | None = None
    final_validation_loss = float("nan")
    for epoch in range(args.num_epochs):
        if activation:
            assert activation_train_dataset is not None
            activation_train_dataset.set_epoch(epoch)
            if hasattr(train_loader, "set_epoch"):
                train_loader.set_epoch(epoch)
        running = 0.0
        train_loss_sum = torch.zeros((), device=accelerator.device)
        train_loss_count = 0
        train_plan_numerator = torch.zeros((), device=accelerator.device)
        train_sid_numerator = torch.zeros((), device=accelerator.device)
        train_plan_denominator = torch.zeros((), dtype=torch.long, device=accelerator.device)
        train_sid_denominator = torch.zeros((), dtype=torch.long, device=accelerator.device)
        for step, batch in enumerate(train_loader, 1):
            with accelerator.accumulate(model):
                if activation:
                    stage_ids = batch.pop("stage_ids")
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    plan_loss, sid_loss, plan_tokens, sid_tokens = _causal_stage_losses(
                        output.logits, batch["labels"], stage_ids
                    )
                    loss = 0.5 * plan_loss + 0.5 * sid_loss
                else:
                    loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer_steps += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running += loss.detach().float().item()
            train_loss_sum += loss.detach().float()
            train_loss_count += 1
            if activation:
                train_plan_numerator += plan_loss.detach().float() * plan_tokens
                train_sid_numerator += sid_loss.detach().float() * sid_tokens
                train_plan_denominator += plan_tokens
                train_sid_denominator += sid_tokens
            if accelerator.is_main_process and step % args.log_every == 0:
                stage_progress = (
                    f" train_plan_loss={plan_loss.detach().float().item():.6f} "
                    f"train_sid_loss={sid_loss.detach().float().item():.6f}"
                    if activation
                    else ""
                )
                print(
                    f"epoch={epoch + 1} step={step} train_loss={running / step:.6f}"
                    f"{stage_progress}"
                )
        if activation:
            valid_plan_loss, valid_sid_loss, validation_loss = (
                _evaluate_activation_losses(model, valid_loader, accelerator)
            )
        else:
            validation_loss = _evaluate_loss(model, valid_loader, accelerator)
        final_validation_loss = validation_loss
        checkpoint_selection_loss = _checkpoint_selection_loss(
            activation,
            validation_loss,
            valid_sid_loss if activation else None,
        )
        gathered_train_loss_sum = accelerator.gather(train_loss_sum.reshape(1)).float().sum()
        gathered_train_loss_count = accelerator.gather(
            torch.tensor([train_loss_count], device=accelerator.device)
        ).sum()
        train_loss = (
            (gathered_train_loss_sum / gathered_train_loss_count).item()
            if gathered_train_loss_count.item()
            else float("nan")
        )
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "checkpoint_selection_loss": checkpoint_selection_loss,
            "micro_batches_per_process": len(train_loader),
            "optimizer_steps_completed": optimizer_steps,
        }
        if activation:
            stage_totals = accelerator.gather(
                torch.stack(
                    [
                        train_plan_numerator,
                        train_sid_numerator,
                        train_plan_denominator.float(),
                        train_sid_denominator.float(),
                    ]
                )
            ).reshape(-1, 4).sum(dim=0)
            train_plan_loss = (
                (stage_totals[0] / stage_totals[2]).item()
                if stage_totals[2].item()
                else float("nan")
            )
            train_sid_loss = (
                (stage_totals[1] / stage_totals[3]).item()
                if stage_totals[3].item()
                else float("nan")
            )
            epoch_summary.update(
                train_plan_loss=train_plan_loss,
                train_sid_loss=train_sid_loss,
                valid_plan_loss=valid_plan_loss,
                valid_sid_loss=valid_sid_loss,
                balanced_valid_loss=validation_loss,
            )
        if accelerator.is_main_process:
            training_metrics["epochs"].append(epoch_summary)
            training_metrics["completed_epochs"] = epoch + 1
            training_metrics["status"] = "complete" if epoch + 1 == args.num_epochs else "running"
            if (
                math.isfinite(checkpoint_selection_loss)
                and checkpoint_selection_loss < best_validation_loss
            ):
                best_validation_loss = checkpoint_selection_loss
                best_epoch = epoch + 1
                training_metrics["best_epoch"] = best_epoch
                training_metrics["best_validation_loss"] = best_validation_loss
                training_metrics["best_checkpoint_selection_loss"] = best_validation_loss
                _save_sft_checkpoint(
                    model,
                    tokenizer,
                    router,
                    accelerator,
                    best_output_dir,
                    args,
                    method,
                    manifest,
                    train_stats,
                    valid_stats,
                    checkpoint_role="best_validation",
                    selected_epoch=best_epoch,
                    selected_validation_loss=best_validation_loss,
                )
            _write_training_metrics(metrics_path, training_metrics)
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                + (
                    f"train_plan_loss={train_plan_loss:.6f} train_sid_loss={train_sid_loss:.6f} "
                    f"valid_plan_loss={valid_plan_loss:.6f} valid_sid_loss={valid_sid_loss:.6f} "
                    f"balanced_valid_loss={validation_loss:.6f} "
                    if activation
                    else f"validation_loss={validation_loss:.6f} "
                )
                + f"optimizer_steps={optimizer_steps} "
                f"best_epoch={best_epoch} "
                f"best_checkpoint_selection_loss={best_validation_loss:.6f}",
                flush=True,
            )
        accelerator.wait_for_everyone()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if best_epoch is None:
            raise RuntimeError("SFT validation produced no finite loss; no best checkpoint was saved")
        _save_sft_checkpoint(
            model,
            tokenizer,
            router,
            accelerator,
            Path(args.output_dir),
            args,
            method,
            manifest,
            train_stats,
            valid_stats,
            checkpoint_role="final",
            selected_epoch=args.num_epochs,
            selected_validation_loss=_checkpoint_selection_loss(
                activation,
                final_validation_loss,
                valid_sid_loss if activation else None,
            ),
        )
    accelerator.wait_for_everyone()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("direct_sft", "minionerec_sft", "diprec_sft", "direct_sid", "sidreasoner_sft"),
        required=True,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--sid_index", required=True)
    parser.add_argument("--item_meta")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--interest_topk", type=int, default=3)
    parser.add_argument("--interest_strategy", choices=("frequency", "time_decay"), default="frequency")
    parser.add_argument("--time_decay", type=float, default=0.1)
    parser.add_argument(
        "--sft_objective",
        choices=("legacy", "interest_activation", "joint_interest_activation"),
        default="legacy",
        help=(
            "DIPRec-SFT objective; interest_activation uses paired plan/SID rows and "
            "joint_interest_activation uses one autoregressive plan-to-SID trajectory"
        ),
    )
    parser.add_argument("--sft_plan_mode", choices=("single", "diverse"), default="single")
    parser.add_argument("--sft_num_plans", type=int, default=8)
    parser.add_argument("--interest_parameterization", choices=("independent_head", "disjoint_rows"), default="independent_head")
    parser.add_argument("--conditioning", choices=("history_visible", "interest_bottleneck"), default="interest_bottleneck")
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--plan_example_limit",
        type=int,
        default=16,
        help="Number of train and validation histories recorded in sampled_plan_examples.jsonl",
    )
    parser.add_argument(
        "--training_metrics_file",
        help="JSON file updated after every completed SFT epoch (runner stores this under outputs/)",
    )
    parser.add_argument(
        "--best_output_dir",
        help="Checkpoint updated whenever validation loss improves (defaults beside output_dir)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
