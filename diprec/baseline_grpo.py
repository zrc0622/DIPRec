"""TRL-GRPO baselines adapted from the official MiniOneRec training contract."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constraints import build_sid_trie, sid_prefix_allowed_fn
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
from .prompts import (
    description_to_sid_prompt,
    history_prompt,
    messages,
    title_history_to_sid_prompt,
    title_to_sid_prompt,
)
from .runtime import load_model_runtime, require_replicated_generation_backend, set_seed
from .rl_logging import PersistentRLTrainingMetricsCallback
from .sft import catalog_alignment_maps

RL_METHODS = ("direct_rl", "minionerec_rl")


def canonical_rl_method(method: str) -> str:
    if method not in RL_METHODS:
        raise ValueError(f"RL method must be one of: {', '.join(RL_METHODS)}")
    return method


def catalog_training_generation_kwargs(
    num_generations: int, temperature: float
) -> dict[str, Any]:
    """Return MiniOneRec's train-time catalog beam-sampling contract."""

    if num_generations < 2:
        raise ValueError("num_generations must be at least two for group-relative training")
    if temperature <= 0:
        raise ValueError("temperature must be positive for beam sampling")
    return {
        "do_sample": True,
        "temperature": float(temperature),
        "top_k": None,
        "top_p": None,
        "num_beams": int(num_generations),
        "num_return_sequences": int(num_generations),
    }


def baseline_batch_contract(
    num_generations: int,
    per_device_batch_size: int,
    generation_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    num_iterations: int = 1,
    steps_per_generation: int | None = None,
) -> dict[str, int]:
    """Validate aligned TRL generation, update, and rollout-reuse batches."""

    values = {
        "num_generations": num_generations,
        "per_device_batch_size": per_device_batch_size,
        "generation_batch_size": generation_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "num_iterations": num_iterations,
    }
    if any(value < 1 for value in values.values()):
        raise ValueError(f"Batch settings must be positive: {values}")
    if num_generations < 2:
        raise ValueError("num_generations must be at least two for group-relative training")
    global_micro_batch = per_device_batch_size * world_size
    if generation_batch_size % num_generations:
        raise ValueError("generation_batch_size must be divisible by num_generations")
    if generation_batch_size % global_micro_batch:
        raise ValueError(
            "generation_batch_size must be divisible by "
            "per_device_batch_size * world_size"
        )
    local_generation_batch = generation_batch_size // world_size
    if local_generation_batch % num_generations:
        raise ValueError(
            "generation_batch_size / world_size must be divisible by num_generations "
            "so every rank generates complete GRPO prompt groups"
        )
    inferred_steps = generation_batch_size // global_micro_batch
    if steps_per_generation is not None and steps_per_generation != inferred_steps:
        raise ValueError(
            "steps_per_generation must equal generation_batch_size / global_micro_batch"
        )
    steps_per_generation = inferred_steps
    if steps_per_generation != gradient_accumulation_steps:
        raise ValueError(
            "generation_batch_size must equal per_device_batch_size * world_size * "
            "gradient_accumulation_steps so each generation maps to one optimizer update"
        )
    effective_update_batch = global_micro_batch * gradient_accumulation_steps
    if effective_update_batch % num_generations:
        raise ValueError(
            "per_device_batch_size * world_size * gradient_accumulation_steps "
            "must be divisible by num_generations"
        )
    return {
        **values,
        "global_micro_batch": global_micro_batch,
        "local_generation_batch": local_generation_batch,
        "steps_per_generation": steps_per_generation,
        "effective_update_batch": effective_update_batch,
        "unique_prompts_per_generation": generation_batch_size // num_generations,
        "local_unique_prompts_per_generation": local_generation_batch // num_generations,
        "optimizer_updates_per_rollout": num_iterations,
        "sampler_repeat_count": num_iterations * steps_per_generation,
    }


def _description_text(value: Any) -> str:
    """Match MiniOneRec's enabled RLTitle2SidDataset normalization.

    The upstream dataset only unwraps strings written as a single-quoted
    Python list and takes the first element.  In particular, it preserves
    leading/trailing whitespace and does not replace an empty description
    with the title; both details affect last-write-wins task deduplication.
    """

    if not isinstance(value, str):
        raise ValueError("MiniOneRec-RL expects item descriptions to be strings")
    if value.startswith("['") and value.endswith("']"):
        try:
            descriptions = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        if isinstance(descriptions, list) and descriptions:
            return str(descriptions[0])
    return value


def _history_rl_row(record: Mapping[str, Any], max_history_len: int) -> dict[str, str]:
    return {
        "prompt": history_prompt(record, max_history_len, reasoning=False),
        "target_sid": str(record["target_item_sid"]),
        "task": "history_sid_to_sid",
        "sample_id": str(record.get("sample_id", "")),
    }


def build_baseline_rl_rows(
    method: str,
    records: Sequence[Mapping[str, Any]],
    sid_map: Mapping[str, Sequence[str]],
    item_metadata: Mapping[str, Mapping[str, Any]] | None,
    max_history_len: int,
    title_sequence_limit: int = 10_000,
    seed: int = 42,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Build the enabled Direct/MiniOneRec RL tasks without heavy dependencies."""

    method = canonical_rl_method(method)
    rows = [_history_rl_row(record, max_history_len) for record in records]
    task_counts = {"history_sid_to_sid": len(rows)}
    if method == "direct_rl":
        return rows, task_counts
    if item_metadata is None:
        raise ValueError("MiniOneRec-RL requires item metadata")

    _, title_to_sid = catalog_alignment_maps(item_metadata, sid_map)
    title_rows = [
        {
            "prompt": title_to_sid_prompt(title),
            "target_sid": sid,
            "task": "title_to_sid",
            "sample_id": f"catalog:{title}:title_to_sid",
        }
        for title, sid in title_to_sid.items()
    ]
    description_to_sid: dict[str, str] = {}
    for item_id, sid_levels in sid_map.items():
        if item_id not in item_metadata:
            raise ValueError(f"SID-index item {item_id!r} is absent from item metadata")
        features = item_metadata[item_id]
        sid = joined_sid(sid_levels)
        description_to_sid[
            _description_text(features["description"])
        ] = sid
    description_rows = [
        {
            "prompt": description_to_sid_prompt(description),
            "target_sid": sid,
            "task": "description_to_sid",
            "sample_id": f"catalog:{description}:description_to_sid",
        }
        for description, sid in description_to_sid.items()
    ]

    sequence_records = list(records)
    if title_sequence_limit < 0:
        raise ValueError("title_sequence_limit must be non-negative")
    if title_sequence_limit == 0:
        sequence_records = []
    elif len(sequence_records) > title_sequence_limit:
        # MiniOneRec's CSVBaseDataset uses
        # ``DataFrame.sample(n=limit, random_state=seed)``.  Its integer-seed
        # path is NumPy RandomState.choice without replacement; reproduce that
        # index order without requiring pandas in this lightweight builder.
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - numpy is a core requirement
            raise RuntimeError("MiniOneRec-compatible sampling requires numpy") from exc
        indices = np.random.RandomState(seed).choice(
            len(sequence_records), size=title_sequence_limit, replace=False
        )
        sequence_records = [sequence_records[int(index)] for index in indices]
    sequence_rows = [
        {
            "prompt": title_history_to_sid_prompt(record, item_metadata, max_history_len),
            "target_sid": str(record["target_item_sid"]),
            "task": "title_history_to_sid",
            "sample_id": str(record.get("sample_id", "")),
        }
        for record in sequence_records
    ]
    rows.extend(title_rows)
    rows.extend(description_rows)
    rows.extend(sequence_rows)
    task_counts.update(
        {
            "title_to_sid": len(title_rows),
            "description_to_sid": len(description_rows),
            "title_history_to_sid": len(sequence_rows),
        }
    )
    return rows, task_counts


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip('\n" ')
    if isinstance(value, Sequence) and value:
        last = value[-1]
        if isinstance(last, Mapping):
            return str(last.get("content", "")).strip('\n" ')
    return str(value).strip('\n" ')


def _canonical_sid_text(value: Any) -> str:
    """Normalize tokenizer-inserted whitespace without weakening exact SID matching."""

    return "".join(_completion_text(value).split())


def exact_match_reward(
    completions: Sequence[Any],
    target_sid: Sequence[str],
    **_: Any,
) -> list[float]:
    if len(completions) != len(target_sid):
        raise ValueError("Completion and target counts differ")
    return [
        float(_canonical_sid_text(completion) == _canonical_sid_text(target))
        for completion, target in zip(completions, target_sid)
    ]


def _rank_aware_group_scores(
    completions: Sequence[str],
    targets: Sequence[str],
    num_generations: int,
    penalties: Sequence[float],
) -> list[float]:
    if len(completions) != len(targets):
        raise ValueError("Completion and target counts differ")
    if len(completions) % num_generations:
        raise ValueError("Reward batch is not divisible by num_generations")
    rewards: list[float] = []
    for start in range(0, len(completions), num_generations):
        group_completions = list(completions[start : start + num_generations])
        group_targets = list(targets[start : start + num_generations])
        if len(set(group_targets)) != 1:
            raise ValueError("A GRPO prompt group contains different targets")
        target = group_targets[0]
        if target not in group_completions:
            rewards.extend([0.0] * num_generations)
            continue
        rewards.extend(
            0.0 if completion == target else penalties[rank]
            for rank, completion in enumerate(group_completions)
        )
    return rewards


def make_rank_aware_reward(num_generations: int):
    """Match MiniOneRec's group reward: penalize wrong ranked beams if the target appears."""

    if num_generations < 2:
        raise ValueError("num_generations must be at least two for group-relative training")
    discount = [1.0 / math.log2(rank + 2) for rank in range(num_generations)]
    total = sum(discount)
    penalties = [-value / total for value in discount]

    def rank_aware_reward(
        completions: Sequence[Any],
        target_sid: Sequence[str],
        **_: Any,
    ) -> list[float]:
        local_completions = [_canonical_sid_text(value) for value in completions]
        local_targets = [_canonical_sid_text(value) for value in target_sid]
        if len(local_completions) != len(local_targets):
            raise ValueError("Completion and target counts differ")

        # TRL may distribute one G-sized group across ranks when the optimization
        # micro-batch is small. Assemble the rank-ordered group for this
        # list-level reward, then return only this rank's slice; TRL gathers the
        # resulting scalar rewards before computing group advantages.
        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - only the dependency-free tests
            dist = None
        if dist is not None and dist.is_available() and dist.is_initialized():
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, (local_completions, local_targets))
            all_completions = [value for batch, _ in gathered for value in batch]
            all_targets = [value for _, batch in gathered for value in batch]
            all_rewards = _rank_aware_group_scores(
                all_completions, all_targets, num_generations, penalties
            )
            start = sum(len(gathered[index][0]) for index in range(dist.get_rank()))
            return all_rewards[start : start + len(local_completions)]
        return _rank_aware_group_scores(
            local_completions, local_targets, num_generations, penalties
        )

    rank_aware_reward.__name__ = "rank_aware_reward"
    return rank_aware_reward


def _render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages(prompt), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages(prompt), **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("Tokenizer chat template did not return text")
    return rendered


def _catalog_trainer_class(base_class: type):
    """Create the pinned TRL 0.24 trainer override lazily for dependency-free dry-runs."""

    class CatalogGRPOTrainer(base_class):
        def __init__(self, *args: Any, sid_trie: Any, **kwargs: Any):
            self.sid_trie = sid_trie
            super().__init__(*args, **kwargs)
            require_replicated_generation_backend(self, "Catalog beam rollout")
            baseline_batch_contract(
                self.num_generations,
                int(self.args.per_device_train_batch_size),
                int(self.args.generation_batch_size),
                int(self.args.gradient_accumulation_steps),
                int(self.args.world_size),
                self.num_iterations,
                int(self.args.steps_per_generation),
            )
            if self.use_vllm:
                raise ValueError("CatalogGRPOTrainer requires use_vllm=False")

        def _generate_single_turn(self, prompts: list[str], images: list[Any] | None):
            import torch

            if images is not None:
                raise ValueError("Catalog SID generation does not support image prompts")
            group_size = self.num_generations
            if len(prompts) % group_size:
                raise ValueError(
                    f"Local prompt batch {len(prompts)} on rank "
                    f"{self.accelerator.process_index} is not divisible by "
                    f"num_generations={group_size}; each rank must receive complete groups"
                )
            unique_prompts: list[str] = []
            for start in range(0, len(prompts), group_size):
                group = prompts[start : start + group_size]
                if any(prompt != group[0] for prompt in group[1:]):
                    raise ValueError(
                        "TRL sampler did not produce contiguous repeated prompt groups "
                        f"within rank {self.accelerator.process_index}"
                    )
                unique_prompts.append(group[0])

            tokenizer = self.processing_class
            from trl.models import unwrap_model_for_generation

            # Match upstream MiniOneRec's non-vLLM path: every DDP rank owns a
            # complete policy replica and generates only its local prompt groups.
            # TRL gathers rewards after generation, so group normalization remains
            # global without concentrating rollout memory on rank zero.
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator
            ) as unwrapped_model:
                encode_kwargs: dict[str, Any] = {
                    "text": unique_prompts,
                    "return_tensors": "pt",
                    "padding": True,
                    "padding_side": "left",
                    "truncation": True,
                    "add_special_tokens": False,
                }
                if self.max_prompt_length is not None:
                    encode_kwargs["max_length"] = self.max_prompt_length
                encoded = tokenizer(**encode_kwargs)
                encoded = {
                    key: value.to(self.accelerator.device)
                    for key, value in encoded.items()
                }
                prompt_width = int(encoded["input_ids"].shape[1])
                allowed = sid_prefix_allowed_fn(
                    self.sid_trie, prompt_width, int(tokenizer.eos_token_id)
                )
                was_training = unwrapped_model.training
                unwrapped_model.eval()
                try:
                    with torch.no_grad():
                        generation_kwargs = catalog_training_generation_kwargs(
                            group_size, self.temperature
                        )
                        generated = unwrapped_model.generate(
                            **encoded,
                            max_new_tokens=4,
                            min_new_tokens=3,
                            early_stopping=True,
                            length_penalty=0.0,
                            prefix_allowed_tokens_fn=allowed,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                            **generation_kwargs,
                        )
                finally:
                    if was_training:
                        unwrapped_model.train()

            prompt_ids: list[list[int]] = []
            completion_ids: list[list[int]] = []
            for prompt_index, attention_mask in enumerate(encoded["attention_mask"]):
                retained_prompt = encoded["input_ids"][prompt_index][attention_mask.bool()].tolist()
                for beam_index in range(group_size):
                    output_index = prompt_index * group_size + beam_index
                    continuation = generated[output_index, prompt_width:].tolist()
                    if tokenizer.eos_token_id in continuation:
                        eos = continuation.index(tokenizer.eos_token_id)
                        continuation = continuation[: eos + 1]
                    prompt_ids.append([int(value) for value in retained_prompt])
                    completion_ids.append([int(value) for value in continuation])
            return prompt_ids, completion_ids, None, {}

    return CatalogGRPOTrainer


def _manifest_for(split_path: Path) -> dict[str, Any]:
    path = split_path.parent / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing long-history manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    method = canonical_rl_method(args.method)
    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)
    manifest = _manifest_for(train_path)
    train_records = read_jsonl(train_path)
    valid_records = read_jsonl(valid_path)
    train_history = validate_history_records(train_records, args.max_history_len, manifest)
    valid_history = validate_history_records(valid_records, args.max_history_len, manifest)
    validate_manifest_sid_index(manifest, args.sid_index)
    sid_map = load_sid_map(args.sid_index)
    item_metadata = None
    if method == "minionerec_rl":
        if not args.item_meta:
            raise ValueError("MiniOneRec-RL requires --item_meta")
        item_metadata = load_item_metadata(args.item_meta, sid_map)
    if not args.dry_run:
        validate_checkpoint_training_contract(
            args.model,
            expected_method="direct_sft" if method == "direct_rl" else "minionerec_sft",
            manifest=manifest,
            item_meta_path=args.item_meta if method == "minionerec_rl" else None,
        )
    train_rows, task_counts = build_baseline_rl_rows(
        method,
        train_records,
        sid_map,
        item_metadata,
        args.max_history_len,
        args.title_sequence_limit,
        args.seed,
    )
    valid_rows, valid_task_counts = build_baseline_rl_rows(
        "direct_rl",
        valid_records,
        sid_map,
        None,
        args.max_history_len,
        args.title_sequence_limit,
        args.seed,
    )
    generation_kwargs = catalog_training_generation_kwargs(
        args.num_generations, args.temperature
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    global_micro_batch = args.per_device_batch_size * world_size
    generation_batch_size = (
        global_micro_batch * args.gradient_accumulation_steps
        if args.generation_batch_size is None
        else args.generation_batch_size
    )
    batch_contract = baseline_batch_contract(
        args.num_generations,
        args.per_device_batch_size,
        generation_batch_size,
        args.gradient_accumulation_steps,
        world_size,
        args.num_iterations,
    )
    if args.reference_mode == "sync":
        if not 0.0 < args.ref_model_mixup_alpha <= 1.0:
            raise ValueError("--ref_model_mixup_alpha must be in (0, 1]")
        if args.ref_model_sync_steps < 1:
            raise ValueError("--ref_model_sync_steps must be positive")
    reference_policy = (
        {
            "mode": "periodic_sync",
            "sync_steps": args.ref_model_sync_steps,
            "mixup_alpha": args.ref_model_mixup_alpha,
        }
        if args.reference_mode == "sync"
        else {"mode": "fixed_sft_checkpoint"}
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": method,
                    "trainer": "trl.GRPOTrainer@0.24.0 + rank-local catalog beam-sampling override",
                    "model": args.model,
                    "train_samples": len(train_rows),
                    "valid_samples": len(valid_rows),
                    "task_counts": task_counts,
                    "valid_task_counts": valid_task_counts,
                    "catalog_items": len(sid_map),
                    "item_metadata": len(item_metadata) if item_metadata is not None else 0,
                    "num_generations": args.num_generations,
                    "generation_mode": "catalog_constrained_beam_sampling",
                    "generation_distribution": "rank_local",
                    "generation": generation_kwargs,
                    "batch": batch_contract,
                    "reference_policy": reference_policy,
                    "use_vllm": False,
                    "train_history": train_history,
                    "valid_history": valid_history,
                },
                indent=2,
            )
        )
        return

    try:
        import torch
        import trl
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - remote GPU dependency
        raise RuntimeError("Baseline RL requires torch, datasets, and trl==0.24.0") from exc
    if getattr(trl, "__version__", None) != "0.24.0":
        raise RuntimeError(
            f"CatalogGRPOTrainer targets trl==0.24.0, found {getattr(trl, '__version__', 'unknown')}"
        )
    if args.max_seq_len <= 4:
        raise ValueError("max_seq_len must leave room for three SID tokens and EOS")

    model, tokenizer, _, _ = load_model_runtime(
        args.model,
        sid_map,
        "disjoint_rows",
        training=True,
        include_interest=False,
    )
    trie = build_sid_trie(tokenizer, sid_map)
    for row in train_rows:
        row["prompt"] = _render_chat_prompt(tokenizer, row["prompt"])
    for row in valid_rows:
        row["prompt"] = _render_chat_prompt(tokenizer, row["prompt"])
    train_dataset = Dataset.from_list(train_rows).shuffle(seed=args.seed)
    valid_dataset = Dataset.from_list(valid_rows)

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        lr_scheduler_type="cosine",
        logging_steps=args.log_every,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to="none",
        remove_unused_columns=False,
        bf16=bf16,
        fp16=bool(torch.cuda.is_available() and not bf16),
        max_prompt_length=args.max_seq_len - 4,
        max_completion_length=4,
        num_generations=args.num_generations,
        generation_batch_size=generation_batch_size,
        temperature=args.temperature,
        beta=args.beta,
        epsilon=args.clip_ratio,
        num_iterations=args.num_iterations,
        loss_type="grpo",
        use_vllm=False,
        sync_ref_model=args.reference_mode == "sync",
        ref_model_sync_steps=args.ref_model_sync_steps,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        seed=args.seed,
    )
    baseline_batch_contract(
        args.num_generations,
        args.per_device_batch_size,
        int(training_args.generation_batch_size),
        args.gradient_accumulation_steps,
        int(training_args.world_size),
        args.num_iterations,
        int(training_args.steps_per_generation),
    )
    CatalogGRPOTrainer = _catalog_trainer_class(GRPOTrainer)
    trainer = CatalogGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        sid_trie=trie,
        reward_funcs=[exact_match_reward, make_rank_aware_reward(args.num_generations)],
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
    )
    if args.training_metrics_file:
        trainer.add_callback(PersistentRLTrainingMetricsCallback(args.training_metrics_file))
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        config = vars(args) | {
            "method": method,
            "trainer": "trl.GRPOTrainer@0.24.0 + rank-local catalog beam-sampling override",
            "task_counts": task_counts,
            "valid_task_counts": valid_task_counts,
            "train_history": train_history,
            "valid_history": valid_history,
            "data_manifest": processed_data_fingerprint(manifest),
            "item_meta_sha256": sha256_file(args.item_meta) if args.item_meta else None,
            "generation_mode": "catalog_constrained_beam_sampling",
            "generation_distribution": "rank_local",
            "generation": generation_kwargs,
            "batch": batch_contract,
            "reference_policy": reference_policy,
            "use_vllm": False,
        }
        Path(args.output_dir, "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    trainer.accelerator.wait_for_everyone()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=RL_METHODS, required=True)
    parser.add_argument("--model", required=True, help="Matching SFT checkpoint")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--sid_index", required=True)
    parser.add_argument("--item_meta")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--training_metrics_file",
        help="Continuously updated JSON log history, kept separately from checkpoints",
    )
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--title_sequence_limit", type=int, default=10_000)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument(
        "--reference_mode",
        choices=("fixed", "sync"),
        default="fixed",
        help="Keep the SFT reference fixed or periodically EMA-sync it from the policy",
    )
    parser.add_argument("--ref_model_sync_steps", type=int, default=512)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.6)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument(
        "--eval_steps",
        type=float,
        default=0.1,
        help="Validation interval; values below 1 are a fraction of total training steps",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=32,
        help="Optimization micro-batch per GPU; G candidates are formed via generation_batch_size",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        help="Global TRL generation batch; defaults to the global effective optimizer batch",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
