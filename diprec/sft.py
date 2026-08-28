"""SFT data encoding and a compact Accelerate training loop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import load_sid_map, read_jsonl, validate_history_records, validate_manifest_sid_index
from .interest import assert_prefix_only_label, diprec_response, interest_tokens_from_history
from .prompts import history_prompt, messages, plan_prompt, sid_prompt
from .runtime import apply_chat_template, load_model_runtime, save_runtime, set_seed, thinking_prompt_ids


def response_for_record(
    record: Mapping[str, Any],
    method: str,
    interest_topk: int,
    interest_strategy: str,
    time_decay: float,
) -> tuple[str, list[str]]:
    target_sid = str(record["target_item_sid"])
    if method == "direct_sid":
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
        raise ValueError("SFT method must be direct_sid, sidreasoner_sft, or diprec_sft")
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
) -> list[dict[str, Any]]:
    response, interest_tokens = response_for_record(
        record, method, interest_topk, interest_strategy, time_decay
    )
    metadata = {"sample_id": record.get("sample_id"), "interest_tokens": interest_tokens}
    if method != "diprec_sft":
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

    plan_response = response[: response.index("</think>") + len("</think>")]
    plan_row = _encode_pair(
        tokenizer,
        plan_prompt(record, max_history_len, interest_topk),
        plan_response,
        max_seq_len,
        metadata | {"stage": "interest_plan"},
        thinking=True,
    )
    sid_row = _encode_pair(
        tokenizer,
        sid_prompt(record, interest_tokens, max_history_len, conditioning),
        str(record["target_item_sid"]),
        max_seq_len,
        metadata | {"stage": "sid_prediction"},
    )
    return [plan_row, sid_row]


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
    )[0]


class _EncodedDataset:
    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


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


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)
    manifest = _manifest_for(train_path)
    train_records = read_jsonl(train_path)
    valid_records = read_jsonl(valid_path)
    train_stats = validate_history_records(train_records, args.max_history_len, manifest)
    valid_stats = validate_history_records(valid_records, args.max_history_len, manifest)
    validate_manifest_sid_index(manifest, args.sid_index)
    sid_map = load_sid_map(args.sid_index)
    if args.dry_run:
        multiplier = 2 if args.method == "diprec_sft" else 1
        print(
            json.dumps(
                {
                    "method": args.method,
                    "train_samples": len(train_records) * multiplier,
                    "valid_samples": len(valid_records) * multiplier,
                    "train_history": train_stats,
                    "valid_history": valid_stats,
                    "catalog_items": len(sid_map),
                    "model": args.model,
                    "interest_parameterization": args.interest_parameterization,
                    "conditioning": args.conditioning,
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
        include_interest=args.method == "diprec_sft",
    )

    train_rows = [
        row
        for record in train_records
        for row in encode_sft_records(
            tokenizer,
            record,
            args.method,
            args.max_history_len,
            args.max_seq_len,
            args.interest_topk,
            args.interest_strategy,
            args.time_decay,
            args.conditioning,
        )
    ]
    valid_rows = [
        row
        for record in valid_records
        for row in encode_sft_records(
            tokenizer,
            record,
            args.method,
            args.max_history_len,
            args.max_seq_len,
            args.interest_topk,
            args.interest_strategy,
            args.time_decay,
            args.conditioning,
        )
    ]
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    collator = CausalCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        _EncodedDataset(train_rows), batch_size=args.micro_batch_size, shuffle=True, collate_fn=collator, generator=generator
    )
    valid_loader = DataLoader(
        _EncodedDataset(valid_rows), batch_size=args.micro_batch_size, shuffle=False, collate_fn=collator
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, update_steps)
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    model.train()
    for epoch in range(args.num_epochs):
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running += loss.detach().float().item()
            if accelerator.is_main_process and step % args.log_every == 0:
                print(f"epoch={epoch + 1} step={step} train_loss={running / step:.6f}")
        validation_loss = _evaluate_loss(model, valid_loader, accelerator)
        if accelerator.is_main_process:
            print(f"epoch={epoch + 1} validation_loss={validation_loss:.6f}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_runtime(unwrapped, tokenizer, router, args.output_dir, args.interest_parameterization)
        training_config = vars(args) | {"train_history": train_stats, "valid_history": valid_stats}
        Path(args.output_dir, "training_config.json").write_text(
            json.dumps(training_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("direct_sid", "sidreasoner_sft", "diprec_sft"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--sid_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--interest_topk", type=int, default=3)
    parser.add_argument("--interest_strategy", choices=("frequency", "time_decay"), default="frequency")
    parser.add_argument("--time_decay", type=float, default=0.1)
    parser.add_argument("--interest_parameterization", choices=("independent_head", "disjoint_rows"), default="independent_head")
    parser.add_argument("--conditioning", choices=("history_visible", "interest_bottleneck"), default="interest_bottleneck")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
