"""Hierarchical G-plan x B-candidate GRPO for DIPRec.

This is deliberately additive to the vendored VeRL path. VeRL's stock GRPO
normalizes complete responses under one UID; DIPRec needs a plan group across
G plans and a nested candidate group within each plan, with separate token
masks. A compact Accelerate loop makes that grouping explicit and auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constraints import build_sid_trie, interest_prefix_allowed_fn, sid_prefix_allowed_fn
from .data import (
    load_sid_map,
    parse_sid_levels,
    read_jsonl,
    validate_history_records,
    validate_manifest_sid_index,
)
from .interest import interest_plan_text
from .prompts import messages, plan_prompt, sid_prompt
from .rewards import RewardWeights, hierarchical_advantages, score_plan, select_unique_plans
from .runtime import apply_chat_template, load_model_runtime, save_runtime, set_seed, thinking_prompt_ids


def group_layout(num_prompts: int, num_plans: int, sid_beams: int) -> list[dict[str, int]]:
    if min(num_prompts, num_plans, sid_beams) < 1:
        raise ValueError("num_prompts, num_plans, and sid_beams must be positive")
    return [
        {"prompt_index": prompt, "plan_index": plan, "candidate_index": candidate}
        for prompt in range(num_prompts)
        for plan in range(num_plans)
        for candidate in range(sid_beams)
    ]


def _device(model: Any):
    return next(model.parameters()).device


def _generate_plans(
    model: Any,
    tokenizer: Any,
    registry: Any,
    record: Mapping[str, Any],
    max_history_len: int,
    interest_topk: int,
    num_plans: int,
    max_seq_len: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
) -> tuple[list[int], list[list[int]], list[list[str]]]:
    import torch

    prompt_ids = thinking_prompt_ids(
        tokenizer, messages(plan_prompt(record, max_history_len, interest_topk))
    )
    opening = tokenizer.encode("<INT_BEGIN>", add_special_tokens=False)
    context = prompt_ids + list(opening)
    end_think = tokenizer.encode("</think>", add_special_tokens=False)
    maximum = len(context) + interest_topk + 1 + len(end_think) + 1
    if maximum > max_seq_len:
        raise ValueError(
            f"Plan prompt plus response needs {maximum} tokens (> max_seq_len={max_seq_len})"
        )
    allowed = interest_prefix_allowed_fn(
        registry.interest_token_ids,
        registry.interest_pad_id,
        registry.interest_end_id,
        end_think,
        len(context),
        interest_topk,
        tokenizer.eos_token_id,
    )
    candidates: list[list[int]] = []
    for _ in range(max_attempts):
        needed = num_plans - len({tuple(row) for row in candidates})
        if needed <= 0:
            break
        batch = torch.tensor([context], dtype=torch.long, device=_device(model))
        attention = torch.ones_like(batch)
        generated = model.generate(
            input_ids=batch,
            attention_mask=attention,
            max_new_tokens=interest_topk + 1 + len(end_think) + 1,
            min_new_tokens=interest_topk + 1 + len(end_think),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=needed,
            prefix_allowed_tokens_fn=allowed,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for sequence in generated:
            plan_ids = sequence[len(context) : len(context) + interest_topk].tolist()
            candidates.append(plan_ids)
    try:
        unique_ids = select_unique_plans(candidates, num_plans)
    except RuntimeError as exc:
        unique_count = len({tuple(row) for row in candidates})
        raise RuntimeError(
            f"Sample {record.get('sample_id')} produced only {unique_count}/{num_plans} distinct plans "
            f"after {max_attempts} rounds. Increase --plan_sampling_attempts/temperature or reduce --num_plans."
        ) from exc
    token_lookup = {
        **{token_id: token for token_id, token in zip(registry.interest_token_ids, registry.interest_tokens)},
        registry.interest_pad_id: "<INT_PAD>",
    }
    plans = [[token_lookup[int(token_id)] for token_id in values] for values in unique_ids]
    return context, unique_ids, plans


def _generate_sid_candidates(
    model: Any,
    tokenizer: Any,
    trie: Any,
    record: Mapping[str, Any],
    plan_tokens: Sequence[str],
    max_history_len: int,
    conditioning: str,
    sid_beams: int,
    max_seq_len: int,
) -> tuple[list[int], list[list[int]], list[list[str]], list[bool]]:
    import torch

    prompt = sid_prompt(record, plan_tokens, max_history_len, conditioning)
    prompt_ids = apply_chat_template(tokenizer, messages(prompt), add_generation_prompt=True)
    if len(prompt_ids) + 4 > max_seq_len:
        raise ValueError(
            f"SID prompt plus response needs {len(prompt_ids) + 4} tokens (> max_seq_len={max_seq_len})"
        )
    allowed = sid_prefix_allowed_fn(trie, len(prompt_ids), tokenizer.eos_token_id)
    batch = torch.tensor([prompt_ids], dtype=torch.long, device=_device(model))
    attention = torch.ones_like(batch)
    generated = model.generate(
        input_ids=batch,
        attention_mask=attention,
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
    candidate_ids = [sequence[len(prompt_ids) : len(prompt_ids) + 3].tolist() for sequence in generated]
    candidates = [tokenizer.convert_ids_to_tokens(ids) for ids in candidate_ids]
    valid = [trie.contains(ids) for ids in candidate_ids]
    if len(candidate_ids) != sid_beams or not all(valid):
        raise RuntimeError(f"Constrained decoder returned {len(candidate_ids)} candidates, valid={valid}")
    return prompt_ids, candidate_ids, candidates, valid


def _sequence_log_probs(model: Any, prompt_ids: Sequence[int], generated_ids: Sequence[int]):
    import torch

    full = list(prompt_ids) + list(generated_ids)
    input_ids = torch.tensor([full], dtype=torch.long, device=_device(model))
    attention = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention).logits[0]
    start = len(prompt_ids) - 1
    token_logits = logits[start : start + len(generated_ids)]
    targets = torch.tensor(generated_ids, dtype=torch.long, device=logits.device)
    return torch.log_softmax(token_logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _ppo_objective(current: Any, old: Any, advantage: float, clip_ratio: float):
    import torch

    adv = torch.as_tensor(float(advantage), dtype=current.dtype, device=current.device)
    ratio = torch.exp(current - old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    return -torch.minimum(unclipped, clipped).mean()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_path = Path(args.train_file)
    manifest = json.loads((train_path.parent / "manifest.json").read_text(encoding="utf-8"))
    records = read_jsonl(train_path)
    history_stats = validate_history_records(records, args.max_history_len, manifest)
    validate_manifest_sid_index(manifest, args.sid_index)
    sid_map = load_sid_map(args.sid_index)
    expected_layout = args.num_plans * args.sid_beams
    if len(group_layout(1, args.num_plans, args.sid_beams)) != expected_layout:
        raise AssertionError("G x B rollout layout is malformed")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "records": len(records),
                    "history": history_stats,
                    "catalog_items": len(sid_map),
                    "model": args.model,
                    "group_shape": [args.num_plans, args.sid_beams],
                    "trajectories_per_prompt": expected_layout,
                    "conditioning": args.conditioning,
                    "parameterization": args.interest_parameterization,
                },
                indent=2,
            )
        )
        return
    try:
        import torch
        from accelerate import Accelerator
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("GRPO training requires torch, transformers, and accelerate") from exc
    model, tokenizer, registry, router = load_model_runtime(
        args.model, sid_map, args.interest_parameterization, training=True
    )
    if router is not None:
        router.assert_parameter_isolation(registry.sid_token_ids)
    trie = build_sid_trie(tokenizer, sid_map)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(records, batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda values: values)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    # PPO ratios require old/current log-probabilities under the same dropout
    # state. Evaluation mode still records gradients and makes the rollout
    # policy deterministic except for the explicit sampling parameters.
    model.eval()
    weights = RewardWeights(
        hr=args.reward_hr,
        ndcg=args.reward_ndcg,
        level1=args.reward_level1,
        level2=args.reward_level2,
        level3=args.reward_level3,
        valid=args.reward_valid,
        duplicate=args.reward_duplicate,
    )
    global_step = 0
    for epoch in range(args.num_epochs):
        for records_batch in loader:
            with accelerator.accumulate(model):
                interest_losses = []
                sid_losses = []
                batch_reward = []
                unwrapped = accelerator.unwrap_model(model)
                for record in records_batch:
                    with torch.no_grad():
                        plan_context, plan_ids, plan_tokens = _generate_plans(
                            unwrapped,
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
                        plan_rewards = []
                        candidate_rewards = []
                        candidate_groups = []
                        for tokens in plan_tokens:
                            candidate_prompt, candidate_ids, candidates, valid = _generate_sid_candidates(
                                unwrapped,
                                tokenizer,
                                trie,
                                record,
                                tokens,
                                args.max_history_len,
                                args.conditioning,
                                args.sid_beams,
                                args.max_seq_len,
                            )
                            plan_reward, per_candidate, _ = score_plan(
                                candidates, parse_sid_levels(record["target_sid_levels"]), valid, weights
                            )
                            plan_rewards.append(plan_reward)
                            candidate_rewards.append(per_candidate)
                            candidate_groups.append((candidate_prompt, candidate_ids))
                        plan_advantages, nested_advantages = hierarchical_advantages(
                            plan_rewards, candidate_rewards, args.mode
                        )
                        old_plan_log_probs = [
                            _sequence_log_probs(unwrapped, plan_context, ids).detach() for ids in plan_ids
                        ]
                        old_candidate_log_probs = [
                            [
                                _sequence_log_probs(unwrapped, prompt_ids, ids).detach()
                                for ids in group_ids
                            ]
                            for prompt_ids, group_ids in candidate_groups
                        ]

                    for plan_index, ids in enumerate(plan_ids):
                        current_plan = _sequence_log_probs(model, plan_context, ids)
                        if args.mode == "plan_grpo":
                            plan_loss = _ppo_objective(
                                current_plan,
                                old_plan_log_probs[plan_index],
                                plan_advantages[plan_index],
                                args.clip_ratio,
                            )
                            interest_losses.append(plan_loss)
                        prompt_ids, group_ids = candidate_groups[plan_index]
                        for candidate_index, candidate_ids in enumerate(group_ids):
                            candidate_advantage = nested_advantages[plan_index][candidate_index]
                            if args.mode == "trajectory_grpo":
                                trajectory_plan_loss = _ppo_objective(
                                    current_plan,
                                    old_plan_log_probs[plan_index],
                                    candidate_advantage,
                                    args.clip_ratio,
                                )
                                interest_losses.append(trajectory_plan_loss)
                            current_candidate = _sequence_log_probs(model, prompt_ids, candidate_ids)
                            sid_loss = _ppo_objective(
                                current_candidate,
                                old_candidate_log_probs[plan_index][candidate_index],
                                candidate_advantage,
                                args.clip_ratio,
                            )
                            sid_losses.append(sid_loss)
                    batch_reward.append(sum(plan_rewards) / len(plan_rewards))
                if not interest_losses or not sid_losses:
                    raise RuntimeError("No GRPO trajectories were generated")
                interest_loss = torch.stack(interest_losses).mean()
                sid_loss = torch.stack(sid_losses).mean()
                loss = args.interest_loss_weight * interest_loss + args.sid_loss_weight * sid_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if accelerator.is_main_process and global_step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={global_step} loss={loss.detach().float().item():.6f} "
                    f"interest_loss={interest_loss.detach().float().item():.6f} "
                    f"sid_loss={sid_loss.detach().float().item():.6f} "
                    f"plan_reward={sum(batch_reward) / len(batch_reward):.6f}"
                )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_runtime(unwrapped, tokenizer, router, args.output_dir, args.interest_parameterization)
        config = vars(args) | {"history": history_stats, "group_shape": [args.num_plans, args.sid_beams]}
        Path(args.output_dir, "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("trajectory_grpo", "plan_grpo"), required=True)
    parser.add_argument("--model", required=True, help="DIPRec SFT checkpoint")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--sid_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--interest_topk", type=int, default=3)
    parser.add_argument("--num_plans", type=int, default=8)
    parser.add_argument("--sid_beams", type=int, default=8)
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--conditioning", choices=("history_visible", "interest_bottleneck"), default="interest_bottleneck")
    parser.add_argument("--interest_parameterization", choices=("independent_head", "disjoint_rows"), default="independent_head")
    parser.add_argument("--interest_loss_weight", type=float, default=1.0)
    parser.add_argument("--sid_loss_weight", type=float, default=1.0)
    parser.add_argument("--reward_hr", type=float, default=1.0)
    parser.add_argument("--reward_ndcg", type=float, default=1.0)
    parser.add_argument("--reward_level1", type=float, default=0.1)
    parser.add_argument("--reward_level2", type=float, default=0.2)
    parser.add_argument("--reward_level3", type=float, default=0.4)
    parser.add_argument("--reward_valid", type=float, default=0.1)
    parser.add_argument("--reward_duplicate", type=float, default=0.1)
    parser.add_argument("--plan_temperature", type=float, default=1.0)
    parser.add_argument("--plan_top_p", type=float, default=0.95)
    parser.add_argument("--plan_sampling_attempts", type=int, default=8)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
