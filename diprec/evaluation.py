"""Unified catalog-constrained evaluator for the seven comparison methods."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constraints import build_sid_trie, sid_prefix_allowed_fn
from .data import (
    joined_sid,
    load_sid_map,
    parse_sid_levels,
    read_jsonl,
    validate_history_records,
    validate_checkpoint_training_contract,
    validate_manifest_sid_index,
)
from .grpo import _generate_plans, _generate_sid_candidates, _sequence_log_probs
from .prompts import history_prompt, messages
from .rewards import aggregate_metric_rows, interest_diversity, ranking_metrics, sid_level_hits
from .runtime import apply_chat_template, load_model_runtime, set_seed, thinking_prompt_ids

BASELINE_METHODS = {"direct_sft", "direct_rl", "minionerec_sft", "minionerec_rl"}
DIPREC_METHODS = {"diprec_sft", "diprec_traj_rl", "diprec_plan_rl"}
ITEM_METADATA_METHODS = {"minionerec_sft", "minionerec_rl", *DIPREC_METHODS}
METHOD_ALIASES = {
    "direct_sid": "direct_sft",
    "diprec_trajectory_grpo": "diprec_traj_rl",
    "diprec_plan_grpo": "diprec_plan_rl",
}


def canonical_evaluation_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def prediction_output_path(metrics_path: str | Path, split: str) -> Path:
    if split not in {"valid", "test"}:
        raise ValueError("split must be valid or test")
    output_path = Path(metrics_path)
    name = "predictions.jsonl" if split == "test" else "valid_predictions.jsonl"
    return output_path.with_name(name)


def per_plan_candidate_budget(total_budget: int, num_plans: int) -> list[int]:
    """Split a fixed raw SID-candidate budget as evenly as possible across plans."""

    if total_budget < 1:
        raise ValueError("total_budget must be positive")
    if num_plans < 1:
        raise ValueError("num_plans must be positive")
    if total_budget < num_plans:
        raise ValueError(
            f"eval_candidate_budget={total_budget} cannot cover num_plans={num_plans}; "
            "increase --eval_candidate_budget or reduce --num_plans"
        )
    quotient, remainder = divmod(total_budget, num_plans)
    return [quotient + int(index < remainder) for index in range(num_plans)]


def unique_top_candidates(
    candidates: Sequence[Sequence[str]], valid: Sequence[bool], limit: int
) -> tuple[list[list[str]], list[bool]]:
    """Keep the first occurrence of each ranked SID without duplicate padding."""

    if len(candidates) != len(valid):
        raise ValueError("Candidate and validity counts differ")
    if limit < 1:
        raise ValueError("Candidate limit must be positive")
    selected: list[list[str]] = []
    selected_valid: list[bool] = []
    seen: set[tuple[str, ...]] = set()
    for candidate, is_valid in zip(candidates, valid):
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        selected.append(list(candidate))
        selected_valid.append(bool(is_valid))
        if len(selected) == limit:
            break
    return selected, selected_valid


def validate_evaluation_checkpoint(
    args: argparse.Namespace, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail fast when a seven-model result points at an incompatible checkpoint."""

    if args.method == "sidreasoner":
        return {}
    item_meta_path = None
    if args.method in ITEM_METADATA_METHODS:
        if not getattr(args, "item_meta", None):
            raise ValueError(f"{args.method} evaluation requires --item_meta")
        item_meta_path = args.item_meta
    expected_config = None
    if args.method in DIPREC_METHODS:
        expected_config = {
            "interest_topk": args.interest_topk,
            "interest_strategy": args.interest_strategy,
            "time_decay": args.time_decay,
            "conditioning": args.conditioning,
            "interest_parameterization": args.interest_parameterization,
        }
        if args.method in {"diprec_traj_rl", "diprec_plan_rl"}:
            expected_config.update(num_plans=args.num_plans, sid_beams=args.sid_beams)
    return validate_checkpoint_training_contract(
        args.model,
        expected_method=args.method,
        manifest=manifest,
        item_meta_path=item_meta_path,
        expected_config=expected_config,
    )


def _device(model: Any):
    return next(model.parameters()).device


def _generate_catalog_beams(
    model: Any,
    tokenizer: Any,
    trie: Any,
    prompt_ids: Sequence[int],
    sid_beams: int,
    max_seq_len: int,
) -> tuple[list[list[int]], list[list[str]], list[bool]]:
    import torch

    if len(prompt_ids) + 4 > max_seq_len:
        raise ValueError(f"Prompt plus SID exceeds max_seq_len={max_seq_len}")
    if len(prompt_ids) + 4 > int(getattr(model.config, "max_position_embeddings", 10**9)):
        raise ValueError("Prompt plus SID exceeds the model context window")
    batch = torch.tensor([list(prompt_ids)], dtype=torch.long, device=_device(model))
    allowed = sid_prefix_allowed_fn(trie, len(prompt_ids), tokenizer.eos_token_id)
    generated = model.generate(
        input_ids=batch,
        attention_mask=torch.ones_like(batch),
        max_new_tokens=4,
        min_new_tokens=3,
        do_sample=False,
        num_beams=sid_beams,
        num_return_sequences=sid_beams,
        early_stopping=True,
        length_penalty=0.0,
        prefix_allowed_tokens_fn=allowed,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    ids = [sequence[len(prompt_ids) : len(prompt_ids) + 3].tolist() for sequence in generated]
    tokens = [tokenizer.convert_ids_to_tokens(sequence) for sequence in ids]
    valid = [trie.contains(sequence) for sequence in ids]
    if len(ids) != sid_beams or not all(valid):
        raise RuntimeError(
            f"Constrained evaluator returned {len(ids)} candidates, valid={valid}"
        )
    return ids, tokens, valid


def _reasoning_context(
    model: Any,
    tokenizer: Any,
    record: Mapping[str, Any],
    max_history_len: int,
    max_reasoning_tokens: int,
    max_seq_len: int,
) -> tuple[list[int], str]:
    import torch

    prompt_ids = thinking_prompt_ids(
        tokenizer, messages(history_prompt(record, max_history_len, reasoning=True))
    )
    if len(prompt_ids) + max_reasoning_tokens + 4 > max_seq_len:
        raise ValueError(
            f"Reasoning prompt budget {len(prompt_ids) + max_reasoning_tokens + 4} exceeds "
            f"max_seq_len={max_seq_len}; lower --max_reasoning_tokens"
        )
    marker = tokenizer.encode("</think>", add_special_tokens=False)

    class StopAtMarker:
        def __init__(self, marker_ids: Sequence[int]):
            self.marker = list(marker_ids)

        def __call__(self, input_ids, scores, **kwargs):
            del scores, kwargs
            return all(row[-len(self.marker) :].tolist() == self.marker for row in input_ids)

    from transformers import StoppingCriteriaList

    batch = torch.tensor([prompt_ids], dtype=torch.long, device=_device(model))
    output = model.generate(
        input_ids=batch,
        attention_mask=torch.ones_like(batch),
        max_new_tokens=max_reasoning_tokens,
        do_sample=False,
        stopping_criteria=StoppingCriteriaList([StopAtMarker(marker)]),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0].tolist()
    continuation = output[len(prompt_ids) :]
    end = next(
        (position + len(marker) for position in range(len(continuation)) if continuation[position : position + len(marker)] == marker),
        None,
    )
    if end is None:
        continuation = continuation + marker
    else:
        continuation = continuation[:end]
    return prompt_ids + continuation, tokenizer.decode(continuation, skip_special_tokens=False)


def _evaluate_record(
    model: Any,
    tokenizer: Any,
    registry: Any,
    trie: Any,
    record: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch

    plans: list[list[str]] = []
    trajectories: list[str] = []
    ranked_candidates: list[list[str]]
    ranked_valid: list[bool]
    plan_budgets: list[int] = []
    reasoning = None
    raw_candidate_count = 0
    unique_candidate_count = 0
    with torch.no_grad():
        if args.method in BASELINE_METHODS:
            prompt_ids = apply_chat_template(
                tokenizer, messages(history_prompt(record, args.max_history_len, reasoning=False)), True
            )
            _, raw_candidates, raw_valid = _generate_catalog_beams(
                model, tokenizer, trie, prompt_ids, args.eval_candidate_budget, args.max_seq_len
            )
            raw_candidate_count = len(raw_candidates)
            unique_candidate_count = len({tuple(candidate) for candidate in raw_candidates})
            ranked_candidates, ranked_valid = unique_top_candidates(
                raw_candidates, raw_valid, args.eval_beams
            )
        elif args.method == "sidreasoner":
            context, reasoning = _reasoning_context(
                model,
                tokenizer,
                record,
                args.max_history_len,
                args.max_reasoning_tokens,
                args.max_seq_len,
            )
            _, raw_candidates, raw_valid = _generate_catalog_beams(
                model, tokenizer, trie, context, args.eval_candidate_budget, args.max_seq_len
            )
            raw_candidate_count = len(raw_candidates)
            unique_candidate_count = len({tuple(candidate) for candidate in raw_candidates})
            ranked_candidates, ranked_valid = unique_top_candidates(
                raw_candidates, raw_valid, args.eval_beams
            )
        elif args.method in DIPREC_METHODS:
            if registry is None:
                raise AssertionError("DIPRec evaluation requires an interest token registry")
            plan_context, plan_ids, plans = _generate_plans(
                model,
                tokenizer,
                registry,
                record,
                args.max_history_len,
                args.interest_topk,
                args.num_plans,
                args.max_seq_len,
                args.plan_temperature,
                args.plan_top_p,
                args.plan_sampling_attempts,
            )
            scored = []
            plan_budgets = per_plan_candidate_budget(args.eval_candidate_budget, len(plans))
            for plan_index, (ids, tokens, candidate_budget) in enumerate(zip(plan_ids, plans, plan_budgets)):
                plan_score = float(_sequence_log_probs(model, plan_context, ids).sum().item())
                sid_context, sid_ids, candidates, valid = _generate_sid_candidates(
                    model,
                    tokenizer,
                    trie,
                    record,
                    tokens,
                    args.max_history_len,
                    args.conditioning,
                    candidate_budget,
                    args.max_seq_len,
                )
                for candidate_index, (candidate_ids, candidate, is_valid) in enumerate(
                    zip(sid_ids, candidates, valid)
                ):
                    sid_score = float(_sequence_log_probs(model, sid_context, candidate_ids).sum().item())
                    scored.append(
                        (plan_score + sid_score, plan_index, candidate_index, candidate, is_valid)
                    )
            raw_candidate_count = len(scored)
            scored.sort(key=lambda row: (-row[0], row[1], row[2]))
            trajectories = [
                f"<think><INT_BEGIN>{''.join(plans[plan_index])}<INT_END></think>{joined_sid(candidate)}"
                for _, plan_index, _, candidate, _ in scored
            ]
            raw_ranked_candidates = [candidate for _, _, _, candidate, _ in scored]
            raw_ranked_valid = [is_valid for _, _, _, _, is_valid in scored]
            unique_candidate_count = len(
                {tuple(candidate) for candidate in raw_ranked_candidates}
            )
            ranked_candidates, ranked_valid = unique_top_candidates(
                raw_ranked_candidates, raw_ranked_valid, args.eval_beams
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")

    target = parse_sid_levels(record["target_sid_levels"])
    rank_metrics = ranking_metrics(ranked_candidates, target)
    best_hits = [
        max((sid_level_hits(candidate, target)[level] for candidate in ranked_candidates), default=0)
        for level in range(3)
    ]
    metrics = {
        **rank_metrics,
        "sid_valid_rate": sum(ranked_valid) / len(ranked_valid) if ranked_valid else 0.0,
        "interest_diversity": interest_diversity(plans),
        "sid_level1_hit": float(best_hits[0]),
        "sid_level2_hit": float(best_hits[1]),
        "sid_level3_hit": float(best_hits[2]),
    }
    prediction = {
        "sample_id": record["sample_id"],
        "target_sid": joined_sid(target),
        "candidate_sids": [joined_sid(candidate) for candidate in ranked_candidates],
        "candidate_sid_levels": ranked_candidates,
        "candidate_valid": ranked_valid,
        "interest_plans": plans,
        "trajectories": trajectories,
        "reasoning": reasoning,
        "raw_candidate_count": raw_candidate_count,
        "unique_candidate_count": unique_candidate_count,
        "returned_candidate_count": len(ranked_candidates),
        "per_plan_candidate_budget": plan_budgets,
        "metrics": metrics,
    }
    return prediction, metrics


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.method = canonical_evaluation_method(args.method)
    if args.method in DIPREC_METHODS and args.num_plans < 1:
        raise ValueError("num_plans must be positive")
    if args.eval_beams < 1:
        raise ValueError("eval_beams must be positive")
    if args.eval_candidate_budget < args.eval_beams:
        raise ValueError(
            f"eval_candidate_budget={args.eval_candidate_budget} is below eval_beams={args.eval_beams}"
        )
    if args.method in DIPREC_METHODS:
        per_plan_candidate_budget(args.eval_candidate_budget, args.num_plans)
    test_path = Path(args.test_file)
    manifest_path = test_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing long-history manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = read_jsonl(test_path)
    history_stats = validate_history_records(records, args.max_history_len, manifest)
    validate_manifest_sid_index(manifest, args.sid_index)
    sid_map = load_sid_map(args.sid_index)
    is_diprec = args.method in DIPREC_METHODS
    checkpoint_training_config = {}
    if not args.dry_run:
        checkpoint_training_config = validate_evaluation_checkpoint(args, manifest)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": args.method,
                    "test_samples": len(records),
                    "history": history_stats,
                    "catalog_items": len(sid_map),
                    "model": args.model,
                    "returned_candidates": args.eval_beams,
                    "raw_candidate_budget": args.eval_candidate_budget,
                    "plans": args.num_plans if is_diprec else 0,
                },
                indent=2,
            )
        )
        return
    model, tokenizer, registry, router = load_model_runtime(
        args.model,
        sid_map,
        args.interest_parameterization,
        training=False,
        include_interest=is_diprec,
    )
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    trie = build_sid_trie(tokenizer, sid_map)
    predictions, rows = [], []
    for index, record in enumerate(records, 1):
        prediction, metrics = _evaluate_record(model, tokenizer, registry, trie, record, args)
        predictions.append(prediction)
        rows.append(metrics)
        if index % args.log_every == 0:
            print(f"evaluated={index}/{len(records)}")
    metrics = aggregate_metric_rows(rows)
    evaluation_config = {
        "max_history_len": args.max_history_len,
        "max_seq_len": args.max_seq_len,
        "interest_topk": args.interest_topk if is_diprec else 0,
        "interest_strategy": args.interest_strategy if is_diprec else None,
        "time_decay": args.time_decay if is_diprec else None,
        "num_plans": args.num_plans if is_diprec else 0,
        "training_sid_beams": args.sid_beams if args.method in {"diprec_traj_rl", "diprec_plan_rl"} else 0,
        "eval_beams": args.eval_beams,
        "eval_candidate_budget": args.eval_candidate_budget,
        "conditioning": args.conditioning if is_diprec else None,
        "interest_parameterization": args.interest_parameterization if is_diprec else None,
        "plan_temperature": args.plan_temperature if is_diprec else None,
        "plan_top_p": args.plan_top_p if is_diprec else None,
        "plan_sampling_attempts": args.plan_sampling_attempts if is_diprec else None,
    }
    result = {
        "schema_version": "diprec.metrics.v1",
        "method": args.method,
        "dataset": manifest["dataset"],
        "seed": args.seed,
        "model": args.base_model or checkpoint_training_config.get("model", args.model),
        "checkpoint": args.model,
        "split": args.split,
        "num_examples": len(records),
        "metrics": metrics,
        "training_config": checkpoint_training_config,
        "evaluation_config": evaluation_config,
        "data_manifest": {
            "source_kind": manifest["source_kind"],
            "source_sha256": manifest.get(
                "source_sha256",
                {"raw": manifest.get("raw_sha256")},
            ),
            "sid_index_sha256": manifest["sid_index_sha256"],
            "split_strategy": manifest["split_strategy"],
        },
        "history_statistics": history_stats,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prediction_path = prediction_output_path(output_path, args.split)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "direct_sft",
            "direct_rl",
            "minionerec_sft",
            "minionerec_rl",
            "diprec_sft",
            "diprec_traj_rl",
            "diprec_plan_rl",
            "direct_sid",
            "sidreasoner",
            "diprec_trajectory_grpo",
            "diprec_plan_grpo",
        ),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--sid_index", required=True)
    parser.add_argument("--item_meta")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--interest_topk", type=int, default=3)
    parser.add_argument(
        "--interest_strategy", choices=("frequency", "time_decay"), default="frequency"
    )
    parser.add_argument("--time_decay", type=float, default=0.1)
    parser.add_argument("--num_plans", type=int, default=8)
    parser.add_argument("--sid_beams", type=int, default=8)
    parser.add_argument("--eval_beams", type=int, default=10)
    parser.add_argument(
        "--eval_candidate_budget",
        type=int,
        default=80,
        help="Total raw constrained SID candidates explored per example for every method",
    )
    parser.add_argument("--conditioning", choices=("history_visible", "interest_bottleneck"), default="interest_bottleneck")
    parser.add_argument("--interest_parameterization", choices=("independent_head", "disjoint_rows"), default="independent_head")
    parser.add_argument("--plan_temperature", type=float, default=1.0)
    parser.add_argument("--plan_top_p", type=float, default=0.95)
    parser.add_argument("--plan_sampling_attempts", type=int, default=8)
    parser.add_argument("--max_reasoning_tokens", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
