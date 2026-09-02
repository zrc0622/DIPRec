"""TRL-backed hierarchical GRPO for DIPRec.

The trainer keeps DIPRec's two-stage rollout (sample G interest plans, then
decode B catalog-constrained SIDs per plan) while delegating optimization,
rollout reuse, checkpointing, and distributed scheduling to TRL 0.24.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constraints import build_sid_trie, interest_prefix_allowed_fn, sid_prefix_allowed_fn
from .data import (
    load_sid_map,
    parse_sid_levels,
    processed_data_fingerprint,
    read_jsonl,
    sha256_file,
    validate_checkpoint_training_contract,
    validate_history_records,
    validate_manifest_sid_index,
)
from .prompts import messages, plan_prompt, sid_prompt
from .rewards import RewardWeights, hierarchical_advantages, score_plan, select_unique_plans
from .runtime import (
    apply_chat_template,
    get_active_interest_router,
    load_model_runtime,
    require_replicated_generation_backend,
    save_runtime,
    set_active_interest_router,
    set_seed,
    thinking_prompt_ids,
)


DIPREC_MODES = ("trajectory_grpo", "plan_grpo")


def group_layout(num_prompts: int, num_plans: int, sid_beams: int) -> list[dict[str, int]]:
    if min(num_prompts, num_plans, sid_beams) < 1:
        raise ValueError("num_prompts, num_plans, and sid_beams must be positive")
    return [
        {"prompt_index": prompt, "plan_index": plan, "candidate_index": candidate}
        for prompt in range(num_prompts)
        for plan in range(num_plans)
        for candidate in range(sid_beams)
    ]


def diprec_batch_contract(
    num_plans: int,
    sid_beams: int,
    per_device_batch_size: int,
    generation_batch_size: int,
    gradient_accumulation_steps: int,
    num_iterations: int,
    world_size: int,
    steps_per_generation: int | None = None,
) -> dict[str, int]:
    """Validate TRL's G-plan grouping and rollout-reuse requirements."""

    values = {
        "num_plans": num_plans,
        "sid_beams": sid_beams,
        "per_device_batch_size": per_device_batch_size,
        "generation_batch_size": generation_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_iterations": num_iterations,
        "world_size": world_size,
    }
    if any(value < 1 for value in values.values()):
        raise ValueError(f"DIPRec RL batch settings must be positive: {values}")
    if num_plans < 2:
        raise ValueError("num_plans must be at least two for group-relative training")
    if sid_beams < 2:
        raise ValueError("sid_beams must be at least two for within-plan advantages")
    if num_iterations < 2:
        raise ValueError("num_iterations must be at least two so PPO clipping can act on reused rollouts")
    global_micro_batch = per_device_batch_size * world_size
    if generation_batch_size % num_plans:
        raise ValueError("generation_batch_size must be divisible by num_plans")
    if generation_batch_size % global_micro_batch:
        raise ValueError(
            "generation_batch_size must be divisible by per_device_batch_size * world_size"
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
            "gradient_accumulation_steps so each rollout iteration maps to one optimizer step"
        )
    effective_update_batch = global_micro_batch * gradient_accumulation_steps
    return {
        **values,
        "global_micro_batch": global_micro_batch,
        "steps_per_generation": steps_per_generation,
        "effective_update_batch": effective_update_batch,
        "unique_prompts_per_generation": generation_batch_size // num_plans,
        "optimizer_updates_per_rollout": num_iterations,
        "sampler_repeat_count": num_iterations * steps_per_generation,
        "trajectories_per_prompt": num_plans * sid_beams,
    }


def stage_advantages(
    plan_rewards: Sequence[float],
    candidate_rewards: Sequence[Sequence[float]],
    mode: str,
) -> tuple[list[list[float]], list[list[float]]]:
    """Return plan-token and SID-token advantages for every G x B rollout."""

    plan_values, candidate_values = hierarchical_advantages(
        plan_rewards, candidate_rewards, mode
    )
    if mode == "plan_grpo":
        plan_token_values = [
            [float(plan_values[index])] * len(candidate_values[index])
            for index in range(len(plan_values))
        ]
    elif mode == "trajectory_grpo":
        # Each complete plan→SID trajectory supplies the advantage to both
        # stages. Keeping the B dimension preserves the original objective.
        plan_token_values = [list(map(float, values)) for values in candidate_values]
    else:  # hierarchical_advantages also checks this; keep the public helper explicit.
        raise ValueError("mode must be plan_grpo or trajectory_grpo")
    return plan_token_values, [list(map(float, values)) for values in candidate_values]


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
            use_cache=True,
        )
        for sequence in generated:
            candidates.append(sequence[len(context) : len(context) + interest_topk].tolist())
    try:
        unique_ids = select_unique_plans(candidates, num_plans)
    except RuntimeError as exc:
        unique_count = len({tuple(row) for row in candidates})
        raise RuntimeError(
            f"Sample {record.get('sample_id')} produced only {unique_count}/{num_plans} distinct plans "
            f"after {max_attempts} rounds. Increase --plan_sampling_attempts/temperature or reduce --num_plans."
        ) from exc
    token_lookup = {
        **{
            token_id: token
            for token_id, token in zip(
                registry.interest_token_ids, registry.interest_tokens
            )
        },
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
    prompt_ids = apply_chat_template(
        tokenizer, messages(prompt), add_generation_prompt=True
    )
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
        use_cache=True,
    )
    candidate_ids = [
        sequence[len(prompt_ids) : len(prompt_ids) + 3].tolist()
        for sequence in generated
    ]
    candidates = [tokenizer.convert_ids_to_tokens(ids) for ids in candidate_ids]
    valid = [trie.contains(ids) for ids in candidate_ids]
    if len(candidate_ids) != sid_beams or not all(valid):
        raise RuntimeError(
            f"Constrained decoder returned {len(candidate_ids)} candidates, valid={valid}"
        )
    return prompt_ids, candidate_ids, candidates, valid


def _sequence_log_probs(
    model: Any, prompt_ids: Sequence[int], generated_ids: Sequence[int]
):
    """Score a generated sequence; retained for evaluator-side reranking."""

    import torch

    full = list(prompt_ids) + list(generated_ids)
    input_ids = torch.tensor([full], dtype=torch.long, device=_device(model))
    attention = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention).logits[0]
    start = len(prompt_ids) - 1
    token_logits = logits[start : start + len(generated_ids)]
    targets = torch.tensor(generated_ids, dtype=torch.long, device=logits.device)
    return torch.log_softmax(token_logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)


def _unused_reward(completions: Sequence[Any], **_: Any) -> list[float]:
    """TRL requires a reward callable; DIPRec scores its nested rollout itself."""

    return [0.0] * len(completions)


def _diprec_trainer_class(base_class: type):
    """Build the TRL 0.24 specialization lazily for dependency-free dry-runs."""

    class DIPRecGRPOTrainer(base_class):
        def __init__(
            self,
            *args: Any,
            sid_trie: Any,
            sid_map: Mapping[str, Sequence[str]],
            token_registry: Any,
            reference_model_path: str,
            mode: str,
            conditioning: str,
            interest_parameterization: str,
            interest_topk: int,
            sid_beams: int,
            max_history_len: int,
            max_seq_len: int,
            plan_temperature: float,
            plan_top_p: float,
            plan_sampling_attempts: int,
            reward_weights: RewardWeights,
            interest_loss_weight: float,
            sid_loss_weight: float,
            logprob_micro_batch_size: int,
            **kwargs: Any,
        ):
            training_args = kwargs.get("args")
            if training_args is None:
                raise ValueError("DIPRecGRPOTrainer requires GRPOConfig through args=")
            requested_beta = float(training_args.beta)
            if requested_beta <= 0:
                raise ValueError("DIPRec GRPO requires beta > 0 for a fixed reference-policy KL")
            if int(training_args.num_iterations) < 2:
                raise ValueError("DIPRec GRPO requires num_iterations >= 2 for rollout reuse")
            if training_args.use_vllm:
                raise ValueError("DIPRec two-stage generation requires use_vllm=False")
            if training_args.use_liger_loss:
                raise ValueError("DIPRec's hierarchical token loss does not support Liger loss")
            if training_args.importance_sampling_level != "token":
                raise ValueError("DIPRec's hierarchical objective requires token-level ratios")
            if getattr(training_args, "gradient_checkpointing", False):
                # compute_loss performs separate policy forwards for the plan
                # and SID stages before one backward pass. Reentrant activation
                # checkpointing marks the same DDP parameters ready from both
                # replay graphs, which distributed autograd rejects. The
                # non-reentrant implementation supports this multi-forward
                # loss while preserving activation-memory savings.
                checkpointing_kwargs = dict(
                    getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
                )
                checkpointing_kwargs["use_reentrant"] = False
                training_args.gradient_checkpointing_kwargs = checkpointing_kwargs

            self.sid_trie = sid_trie
            self.token_registry = token_registry
            self.mode = mode
            self.conditioning = conditioning
            self.interest_parameterization = interest_parameterization
            self.interest_topk = interest_topk
            self.sid_beams = sid_beams
            self.diprec_max_history_len = max_history_len
            self.diprec_max_seq_len = max_seq_len
            self.plan_temperature = plan_temperature
            self.plan_top_p = plan_top_p
            self.plan_sampling_attempts = plan_sampling_attempts
            self.diprec_reward_weights = reward_weights
            self.interest_loss_weight = interest_loss_weight
            self.sid_loss_weight = sid_loss_weight
            self.logprob_micro_batch_size = logprob_micro_batch_size

            # Stock TRL would reload only the base HF weights and omit the
            # separately stored independent interest adapter. Suppress that
            # reference creation and install the complete DIPRec-SFT runtime
            # immediately after the base trainer has initialized.
            training_args.beta = 0.0
            try:
                super().__init__(*args, **kwargs)
            finally:
                training_args.beta = requested_beta
            require_replicated_generation_backend(self, "DIPRec two-stage rollout")
            diprec_batch_contract(
                self.num_generations,
                sid_beams,
                int(self.args.per_device_train_batch_size),
                int(self.args.generation_batch_size),
                int(self.args.gradient_accumulation_steps),
                self.num_iterations,
                int(self.args.world_size),
                int(self.args.steps_per_generation),
            )
            self.beta = requested_beta
            self.interest_router = get_active_interest_router()
            if (
                interest_parameterization == "independent_head"
                and self.interest_router is None
            ):
                raise RuntimeError("TRL initialization lost the current interest-adapter hooks")

            reference, reference_tokenizer, reference_registry, reference_router = (
                load_model_runtime(
                    reference_model_path,
                    sid_map,
                    interest_parameterization,
                    training=False,
                )
            )
            if tuple(reference_registry.interest_token_ids) != tuple(
                token_registry.interest_token_ids
            ):
                raise ValueError("Reference/current interest token IDs differ")
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
            if getattr(reference, "is_gradient_checkpointing", False):
                reference.gradient_checkpointing_disable()
            reference.eval()
            self.reference_tokenizer = reference_tokenizer
            self.reference_router = reference_router
            if interest_parameterization == "independent_head" and reference_router is None:
                raise RuntimeError("Reference DIPRec checkpoint has no interest adapter")
            # The current router was recorded before loading the reference;
            # restore it as the process-wide active runtime for save/eval helpers.
            set_active_interest_router(self.interest_router)
            self.ref_model = self.accelerator.prepare_model(
                reference, evaluation_mode=True
            )

        def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
            """Save reloadable DIPRec checkpoints, including the hook-backed adapter."""

            destination = output_dir or self.args.output_dir
            model = self.accelerator.unwrap_model(
                self.model, keep_torch_compile=False
            )
            save_runtime(
                model,
                self.processing_class,
                self.interest_router,
                destination,
                self.interest_parameterization,
                safe_serialization=bool(self.args.save_safetensors),
            )
            import torch

            torch.save(self.args, Path(destination) / "training_args.bin")

        def _restore_interest_adapter(self, checkpoint_dir: str | Path) -> None:
            """Strictly restore an independent adapter without replacing its Parameters."""

            if self.interest_parameterization != "independent_head":
                return
            if self.interest_router is None:
                raise RuntimeError(
                    "Cannot resume an independent-head DIPRec checkpoint without "
                    "the current interest adapter"
                )

            source = Path(checkpoint_dir)
            config_path = source / "diprec_adapter_config.json"
            state_path = source / "diprec_interest_adapter.pt"
            if not config_path.is_file():
                raise FileNotFoundError(
                    "Missing independent interest adapter configuration in checkpoint: "
                    f"{config_path}"
                )
            if not state_path.is_file():
                raise FileNotFoundError(
                    "Missing independent interest adapter weights in checkpoint: "
                    f"{state_path}"
                )
            try:
                saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid independent interest adapter configuration: {config_path}"
                ) from exc
            if saved_config.get("mode") != "independent_head":
                raise ValueError(
                    "Checkpoint interest adapter mode does not match requested "
                    f"independent_head: {saved_config.get('mode')!r}"
                )
            saved_ids = saved_config.get("interest_token_ids")
            if not isinstance(saved_ids, list) or not all(
                isinstance(token_id, int) for token_id in saved_ids
            ):
                raise ValueError(
                    "Checkpoint independent interest adapter configuration must contain "
                    "an integer interest_token_ids list"
                )
            current_ids = self.interest_router.adapter.global_ids.detach().cpu().tolist()
            if saved_ids != current_ids:
                raise ValueError(
                    "Checkpoint independent interest token IDs do not match the current "
                    f"runtime: saved={saved_ids}, current={current_ids}"
                )

            import torch

            try:
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                self.interest_router.adapter.load_state_dict(state, strict=True)
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError(
                    f"Invalid independent interest adapter weights: {state_path}"
                ) from exc
            self.interest_router.assert_parameter_isolation(
                self.token_registry.sid_token_ids
            )
            set_active_interest_router(self.interest_router)

        def _load_from_checkpoint(
            self, resume_from_checkpoint: str, model: Any = None
        ) -> None:
            """Restore a normal Trainer checkpoint and its DIPRec adapter sidecar.

            ``save_runtime`` deliberately keeps the hook-backed adapter out of
            Hugging Face's model state and writes it as a small sidecar. The
            stock Trainer loader therefore restores only the base model. Run
            that loader first, then copy the sidecar into the already-attached
            adapter: this preserves its Parameter objects for the optimizer
            and ordinary DDP wrapping that happen later in ``Trainer.train``.
            """

            super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            self._restore_interest_adapter(resume_from_checkpoint)

        def _load_best_model(self) -> None:
            """Keep a best-model reload from mixing base and adapter checkpoints."""

            super()._load_best_model()
            if self.state.best_model_checkpoint is None:
                return
            self._restore_interest_adapter(self.state.best_model_checkpoint)

        @staticmethod
        def _pad_sequences(
            sequences: Sequence[Sequence[int]],
            padding_value: int,
            padding_side: str,
            device: Any,
        ) -> tuple[Any, Any]:
            import torch

            if not sequences or any(not sequence for sequence in sequences):
                raise ValueError("DIPRec rollout contains an empty token sequence")
            width = max(len(sequence) for sequence in sequences)
            ids = torch.full(
                (len(sequences), width),
                int(padding_value),
                dtype=torch.long,
                device=device,
            )
            mask = torch.zeros_like(ids)
            for row, sequence in enumerate(sequences):
                values = torch.tensor(sequence, dtype=torch.long, device=device)
                if padding_side == "left":
                    ids[row, -len(sequence) :] = values
                    mask[row, -len(sequence) :] = 1
                else:
                    ids[row, : len(sequence)] = values
                    mask[row, : len(sequence)] = 1
            return ids, mask

        def _stage_logps(
            self,
            model: Any,
            prompt_ids: Any,
            prompt_mask: Any,
            completion_ids: Any,
            completion_mask: Any,
            compute_entropy: bool = False,
        ) -> tuple[Any, Any]:
            import torch

            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            return self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                completion_ids.size(1),
                batch_size=self.logprob_micro_batch_size,
                compute_entropy=compute_entropy,
            )

        def _generate_payload(self, all_inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            from trl.models import unwrap_model_for_generation

            # Every rank enters the context so ordinary DDP stays synchronized;
            # only rank zero performs the expensive generation below.
            payload: list[dict[str, Any]] = []
            generation_model = None
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator
            ) as model:
                generation_model = model
                if self.accelerator.is_main_process:
                    payload = self._generate_payload_with_model(model, all_inputs)

            # TRL's generation context restores checkpointing without passing
            # the configured kwargs, which changes it back to the reentrant
            # implementation. DIPRec combines two policy forwards (plan and
            # SID) into one backward pass; reentrant checkpointing cannot mark
            # the same DDP parameters ready twice. Restore the explicitly
            # supported non-reentrant mode after every generation context.
            if (
                getattr(getattr(self, "args", None), "gradient_checkpointing", False)
                and generation_model is not None
                and getattr(generation_model, "is_gradient_checkpointing", False)
            ):
                generation_model.gradient_checkpointing_disable()
                generation_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            return payload

        def _generate_payload_with_model(
            self, model: Any, all_inputs: Sequence[Mapping[str, Any]]
        ) -> list[dict[str, Any]]:
            group_size = self.num_generations
            if len(all_inputs) % group_size:
                raise ValueError(
                    f"Global generation batch {len(all_inputs)} is not divisible by num_plans={group_size}"
                )
            tokenizer = self.processing_class
            payload: list[dict[str, Any]] = []
            was_training = model.training
            model.eval()
            try:
                for start in range(0, len(all_inputs), group_size):
                    repeated = all_inputs[start : start + group_size]
                    sample_ids = [str(row.get("sample_id", "")) for row in repeated]
                    if len(set(sample_ids)) != 1:
                        raise ValueError(
                            "TRL sampler did not produce contiguous repeated DIPRec prompt groups"
                        )
                    record = repeated[0]
                    plan_context, plan_ids, plan_tokens = _generate_plans(
                        model,
                        tokenizer,
                        self.token_registry,
                        record,
                        self.diprec_max_history_len,
                        self.interest_topk,
                        group_size,
                        self.diprec_max_seq_len,
                        self.plan_temperature,
                        self.plan_top_p,
                        self.plan_sampling_attempts,
                    )
                    plan_rewards: list[float] = []
                    candidate_rewards: list[list[float]] = []
                    candidate_groups: list[tuple[list[int], list[list[int]]]] = []
                    for tokens in plan_tokens:
                        prompt_ids, candidate_ids, candidates, valid = (
                            _generate_sid_candidates(
                                model,
                                tokenizer,
                                self.sid_trie,
                                record,
                                tokens,
                                self.diprec_max_history_len,
                                self.conditioning,
                                self.sid_beams,
                                self.diprec_max_seq_len,
                            )
                        )
                        plan_reward, per_candidate, _ = score_plan(
                            candidates,
                            parse_sid_levels(record["target_sid_levels"]),
                            valid,
                            self.diprec_reward_weights,
                        )
                        plan_rewards.append(plan_reward)
                        candidate_rewards.append(per_candidate)
                        candidate_groups.append((prompt_ids, candidate_ids))
                    plan_advantages, sid_advantages = stage_advantages(
                        plan_rewards, candidate_rewards, self.mode
                    )
                    for plan_index in range(group_size):
                        sid_context, sid_ids = candidate_groups[plan_index]
                        payload.append(
                            {
                                "plan_prompt_ids": plan_context,
                                "plan_completion_ids": plan_ids[plan_index],
                                "sid_prompt_ids": [sid_context] * self.sid_beams,
                                "sid_completion_ids": sid_ids,
                                "plan_advantages": plan_advantages[plan_index],
                                "sid_advantages": sid_advantages[plan_index],
                                "plan_reward": float(plan_rewards[plan_index]),
                            }
                        )
            finally:
                if was_training:
                    model.train()
            return payload

        def _tensorize_payload(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            import torch

            device = self.accelerator.device
            plan_prompt_ids, plan_prompt_mask = self._pad_sequences(
                [row["plan_prompt_ids"] for row in rows],
                self.pad_token_id,
                "left",
                device,
            )
            plan_completion_ids, plan_completion_mask = self._pad_sequences(
                [row["plan_completion_ids"] for row in rows],
                self.pad_token_id,
                "right",
                device,
            )
            flat_sid_prompts = [
                sequence for row in rows for sequence in row["sid_prompt_ids"]
            ]
            flat_sid_completions = [
                sequence for row in rows for sequence in row["sid_completion_ids"]
            ]
            sid_prompt_ids, sid_prompt_mask = self._pad_sequences(
                flat_sid_prompts, self.pad_token_id, "left", device
            )
            sid_completion_ids, sid_completion_mask = self._pad_sequences(
                flat_sid_completions, self.pad_token_id, "right", device
            )
            batch_size = len(rows)
            sid_prompt_ids = sid_prompt_ids.reshape(batch_size, self.sid_beams, -1)
            sid_prompt_mask = sid_prompt_mask.reshape(batch_size, self.sid_beams, -1)
            sid_completion_ids = sid_completion_ids.reshape(
                batch_size, self.sid_beams, -1
            )
            sid_completion_mask = sid_completion_mask.reshape(
                batch_size, self.sid_beams, -1
            )
            return {
                "plan_prompt_ids": plan_prompt_ids,
                "plan_prompt_mask": plan_prompt_mask,
                "plan_completion_ids": plan_completion_ids,
                "plan_completion_mask": plan_completion_mask,
                "sid_prompt_ids": sid_prompt_ids,
                "sid_prompt_mask": sid_prompt_mask,
                "sid_completion_ids": sid_completion_ids,
                "sid_completion_mask": sid_completion_mask,
                "plan_advantages": torch.tensor(
                    [row["plan_advantages"] for row in rows],
                    dtype=torch.float32,
                    device=device,
                ),
                "sid_advantages": torch.tensor(
                    [row["sid_advantages"] for row in rows],
                    dtype=torch.float32,
                    device=device,
                ),
                "plan_rewards": torch.tensor(
                    [row["plan_reward"] for row in rows],
                    dtype=torch.float32,
                    device=device,
                ),
            }

        def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            import torch
            from accelerate.utils import broadcast_object_list, gather_object

            all_inputs = gather_object(inputs)
            if not isinstance(all_inputs, list):
                all_inputs = list(all_inputs)
            generated_payload = self._generate_payload(all_inputs)
            payload = generated_payload if self.accelerator.is_main_process else None
            values = [payload]
            broadcast_object_list(values, from_process=0)
            all_rows = values[0]
            if all_rows is None:
                raise RuntimeError("DIPRec rollout broadcast returned no payload")
            local_count = len(inputs)
            counts = gather_object([local_count])
            if not isinstance(counts, list):
                counts = list(counts)
            start = sum(int(value) for value in counts[: self.accelerator.process_index])
            rows = all_rows[start : start + local_count]
            batch = self._tensorize_payload(rows)

            flat_sid_prompt_ids = batch["sid_prompt_ids"].flatten(0, 1)
            flat_sid_prompt_mask = batch["sid_prompt_mask"].flatten(0, 1)
            flat_sid_completion_ids = batch["sid_completion_ids"].flatten(0, 1)
            flat_sid_completion_mask = batch["sid_completion_mask"].flatten(0, 1)
            with torch.no_grad():
                old_plan_logps, _ = self._stage_logps(
                    self.model,
                    batch["plan_prompt_ids"],
                    batch["plan_prompt_mask"],
                    batch["plan_completion_ids"],
                    batch["plan_completion_mask"],
                )
                old_sid_logps, _ = self._stage_logps(
                    self.model,
                    flat_sid_prompt_ids,
                    flat_sid_prompt_mask,
                    flat_sid_completion_ids,
                    flat_sid_completion_mask,
                )
                ref_plan_logps, _ = self._stage_logps(
                    self.ref_model,
                    batch["plan_prompt_ids"],
                    batch["plan_prompt_mask"],
                    batch["plan_completion_ids"],
                    batch["plan_completion_mask"],
                )
                ref_sid_logps, _ = self._stage_logps(
                    self.ref_model,
                    flat_sid_prompt_ids,
                    flat_sid_prompt_mask,
                    flat_sid_completion_ids,
                    flat_sid_completion_mask,
                )
            batch["old_plan_logps"] = old_plan_logps
            batch["old_sid_logps"] = old_sid_logps.reshape(
                local_count, self.sid_beams, -1
            )
            batch["ref_plan_logps"] = ref_plan_logps
            batch["ref_sid_logps"] = ref_sid_logps.reshape(
                local_count, self.sid_beams, -1
            )
            token_count = (
                batch["plan_completion_mask"].sum()
                + batch["sid_completion_mask"].sum()
            )
            batch["num_items_in_batch"] = token_count

            mode = "train" if self.model.training else "eval"
            mean_reward = batch["plan_rewards"].mean()
            self._metrics[mode]["reward"].append(
                self.accelerator.gather(mean_reward).mean().item()
            )
            self._metrics[mode]["rollout_reuse"].append(float(self.num_iterations))
            return batch

        @staticmethod
        def _masked_mean(values: Any, mask: Any) -> Any:
            expanded_mask = mask.expand_as(values)
            return (values * expanded_mask).sum() / expanded_mask.sum().clamp(min=1.0)

        def _stage_loss(
            self,
            current_logps: Any,
            old_logps: Any,
            ref_logps: Any,
            completion_mask: Any,
            advantages: Any,
        ) -> tuple[Any, Any, Any]:
            import torch

            # current/old/ref/mask: [N,T] or [N,B,T]; advantages: [N,B].
            if current_logps.ndim == 2 and advantages.ndim == 2:
                current_logps = current_logps.unsqueeze(1)
                old_logps = old_logps.unsqueeze(1)
                ref_logps = ref_logps.unsqueeze(1)
                completion_mask = completion_mask.unsqueeze(1)
            advantage = advantages.unsqueeze(-1)
            ratio = (current_logps - old_logps).exp()
            clipped_ratio = ratio.clamp(
                1.0 - self.epsilon_low, 1.0 + self.epsilon_high
            )
            policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
            per_token_kl = (
                (ref_logps - current_logps).exp()
                - (ref_logps - current_logps)
                - 1.0
            )
            total = policy_loss + self.beta * per_token_kl
            loss = self._masked_mean(total, completion_mask)
            kl = self._masked_mean(per_token_kl, completion_mask)
            low = (ratio < 1.0 - self.epsilon_low) & (advantage < 0)
            high = (ratio > 1.0 + self.epsilon_high) & (advantage > 0)
            clip_fraction = self._masked_mean((low | high).float(), completion_mask)
            return loss, kl, clip_fraction

        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            if return_outputs:
                raise ValueError("DIPRecGRPOTrainer does not support returning outputs")
            plan_logps, _ = self._stage_logps(
                model,
                inputs["plan_prompt_ids"],
                inputs["plan_prompt_mask"],
                inputs["plan_completion_ids"],
                inputs["plan_completion_mask"],
            )
            batch_size = inputs["sid_prompt_ids"].shape[0]
            sid_logps, _ = self._stage_logps(
                model,
                inputs["sid_prompt_ids"].flatten(0, 1),
                inputs["sid_prompt_mask"].flatten(0, 1),
                inputs["sid_completion_ids"].flatten(0, 1),
                inputs["sid_completion_mask"].flatten(0, 1),
            )
            sid_logps = sid_logps.reshape(batch_size, self.sid_beams, -1)
            plan_loss, plan_kl, plan_clip = self._stage_loss(
                plan_logps,
                inputs["old_plan_logps"],
                inputs["ref_plan_logps"],
                inputs["plan_completion_mask"],
                inputs["plan_advantages"],
            )
            sid_loss, sid_kl, sid_clip = self._stage_loss(
                sid_logps,
                inputs["old_sid_logps"],
                inputs["ref_sid_logps"],
                inputs["sid_completion_mask"],
                inputs["sid_advantages"],
            )
            loss = (
                self.interest_loss_weight * plan_loss
                + self.sid_loss_weight * sid_loss
            )
            mode = "train" if model.training else "eval"
            for name, value in (
                ("loss/plan", plan_loss),
                ("loss/sid", sid_loss),
                ("kl/plan", plan_kl),
                ("kl/sid", sid_kl),
                ("clip_ratio/plan", plan_clip),
                ("clip_ratio/sid", sid_clip),
            ):
                self._metrics[mode][name].append(
                    self.accelerator.gather(value.detach()).nanmean().item()
                )
            return loss / self.current_gradient_accumulation_steps

    return DIPRecGRPOTrainer


def _manifest_for(split_path: Path) -> dict[str, Any]:
    path = split_path.parent / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing long-history manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _diprec_rl_rows(
    records: Sequence[Mapping[str, Any]], max_history_len: int, interest_topk: int
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = dict(record)
        row["prompt"] = plan_prompt(record, max_history_len, interest_topk)
        rows.append(row)
    return rows


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)
    manifest = _manifest_for(train_path)
    records = read_jsonl(train_path)
    valid_records = read_jsonl(valid_path)
    history_stats = validate_history_records(records, args.max_history_len, manifest)
    valid_history_stats = validate_history_records(
        valid_records, args.max_history_len, manifest
    )
    validate_manifest_sid_index(manifest, args.sid_index)
    sid_map = load_sid_map(args.sid_index)
    if not args.dry_run:
        validate_checkpoint_training_contract(
            args.model,
            expected_method="diprec_sft",
            manifest=manifest,
            item_meta_path=args.item_meta,
            expected_config={
                "interest_topk": args.interest_topk,
                "interest_strategy": args.interest_strategy,
                "time_decay": args.time_decay,
                "conditioning": args.conditioning,
                "interest_parameterization": args.interest_parameterization,
            },
        )
    expected_layout = args.num_plans * args.sid_beams
    if len(group_layout(1, args.num_plans, args.sid_beams)) != expected_layout:
        raise AssertionError("G x B rollout layout is malformed")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    global_micro_batch = args.per_device_batch_size * world_size
    generation_batch_size = (
        global_micro_batch * args.gradient_accumulation_steps
        if args.generation_batch_size is None
        else args.generation_batch_size
    )
    batch_contract = diprec_batch_contract(
        args.num_plans,
        args.sid_beams,
        args.per_device_batch_size,
        generation_batch_size,
        args.gradient_accumulation_steps,
        args.num_iterations,
        world_size,
    )
    if args.beta <= 0:
        raise ValueError("beta must be positive so DIPRec uses a fixed reference-policy KL")
    if args.logprob_micro_batch_size < 1:
        raise ValueError("logprob_micro_batch_size must be positive")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": (
                        "diprec_traj_rl"
                        if args.mode == "trajectory_grpo"
                        else "diprec_plan_rl"
                    ),
                    "mode": args.mode,
                    "trainer": "trl.GRPOTrainer@0.24.0 + DIPRec hierarchical override",
                    "records": len(records),
                    "valid_records": len(valid_records),
                    "history": history_stats,
                    "valid_history": valid_history_stats,
                    "catalog_items": len(sid_map),
                    "model": args.model,
                    "group_shape": [args.num_plans, args.sid_beams],
                    "trajectories_per_prompt": expected_layout,
                    "conditioning": args.conditioning,
                    "parameterization": args.interest_parameterization,
                    "interest_strategy": args.interest_strategy,
                    "time_decay": args.time_decay,
                    "reference_policy": "fixed_diprec_sft_checkpoint",
                    "beta": args.beta,
                    "num_iterations": args.num_iterations,
                    "old_policy": "rollout_snapshot",
                    "ppo_clipping": "active_after_first_reused_update",
                    "advantage_assignment": (
                        "trajectory_to_plan_and_sid"
                        if args.mode == "trajectory_grpo"
                        else "plan_across_G_and_sid_within_B"
                    ),
                    "generation": {
                        "plan": "constrained_sampling",
                        "sid": "catalog_constrained_deterministic_beam",
                    },
                    "batch": batch_contract,
                    "use_vllm": False,
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
        raise RuntimeError(
            "DIPRec RL requires torch, datasets, and trl==0.24.0"
        ) from exc
    if getattr(trl, "__version__", None) != "0.24.0":
        raise RuntimeError(
            f"DIPRecGRPOTrainer targets trl==0.24.0, found {getattr(trl, '__version__', 'unknown')}"
        )

    model, tokenizer, registry, _ = load_model_runtime(
        args.model,
        sid_map,
        args.interest_parameterization,
        training=True,
    )
    if registry is None:
        raise AssertionError("DIPRec runtime did not register interest tokens")
    current_router = get_active_interest_router()
    if current_router is not None:
        current_router.assert_parameter_isolation(registry.sid_token_ids)
    trie = build_sid_trie(tokenizer, sid_map)
    train_dataset = Dataset.from_list(
        _diprec_rl_rows(records, args.max_history_len, args.interest_topk)
    ).shuffle(seed=args.seed)
    valid_dataset = Dataset.from_list(
        _diprec_rl_rows(valid_records, args.max_history_len, args.interest_topk)
    )
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.num_plans,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
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
        max_prompt_length=args.max_seq_len,
        max_completion_length=max(args.interest_topk, 3),
        num_generations=args.num_plans,
        generation_batch_size=generation_batch_size,
        temperature=1.0,
        beta=args.beta,
        epsilon=args.clip_ratio,
        num_iterations=args.num_iterations,
        importance_sampling_level="token",
        loss_type="grpo",
        use_liger_loss=False,
        use_vllm=False,
        disable_dropout=True,
        seed=args.seed,
    )
    diprec_batch_contract(
        args.num_plans,
        args.sid_beams,
        args.per_device_batch_size,
        int(training_args.generation_batch_size),
        args.gradient_accumulation_steps,
        args.num_iterations,
        int(training_args.world_size),
        int(training_args.steps_per_generation),
    )
    TrainerClass = _diprec_trainer_class(GRPOTrainer)
    trainer = TrainerClass(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_unused_reward],
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        sid_trie=trie,
        sid_map=sid_map,
        token_registry=registry,
        reference_model_path=args.model,
        mode=args.mode,
        conditioning=args.conditioning,
        interest_parameterization=args.interest_parameterization,
        interest_topk=args.interest_topk,
        sid_beams=args.sid_beams,
        max_history_len=args.max_history_len,
        max_seq_len=args.max_seq_len,
        plan_temperature=args.plan_temperature,
        plan_top_p=args.plan_top_p,
        plan_sampling_attempts=args.plan_sampling_attempts,
        reward_weights=RewardWeights(
            hr=args.reward_hr,
            ndcg=args.reward_ndcg,
            level1=args.reward_level1,
            level2=args.reward_level2,
            level3=args.reward_level3,
            valid=args.reward_valid,
            duplicate=args.reward_duplicate,
        ),
        interest_loss_weight=args.interest_loss_weight,
        sid_loss_weight=args.sid_loss_weight,
        logprob_micro_batch_size=args.logprob_micro_batch_size,
    )
    trainer.train(resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None))
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        save_runtime(
            unwrapped,
            tokenizer,
            trainer.interest_router,
            args.output_dir,
            args.interest_parameterization,
            safe_serialization=bool(training_args.save_safetensors),
        )
        config = vars(args) | {
            "method": (
                "diprec_traj_rl"
                if args.mode == "trajectory_grpo"
                else "diprec_plan_rl"
            ),
            "trainer": "trl.GRPOTrainer@0.24.0 + DIPRec hierarchical override",
            "history": history_stats,
            "valid_history": valid_history_stats,
            "group_shape": [args.num_plans, args.sid_beams],
            "data_manifest": processed_data_fingerprint(manifest),
            "item_meta_sha256": sha256_file(args.item_meta),
            "batch": batch_contract,
            "reference_policy": "fixed_diprec_sft_checkpoint",
            "old_policy": "rollout_snapshot",
            "use_vllm": False,
        }
        Path(args.output_dir, "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    trainer.accelerator.wait_for_everyone()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=DIPREC_MODES, required=True)
    parser.add_argument("--model", required=True, help="DIPRec SFT checkpoint")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--sid_index", required=True)
    parser.add_argument(
        "--item_meta", required=True, help="Item metadata used by the MiniOneRec-SFT parent"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--interest_topk", type=int, default=3)
    parser.add_argument(
        "--interest_strategy",
        choices=("frequency", "time_decay"),
        default="frequency",
        help="Must match the DIPRec-SFT parent checkpoint",
    )
    parser.add_argument(
        "--time_decay",
        type=float,
        default=0.1,
        help="Must match the DIPRec-SFT parent checkpoint",
    )
    parser.add_argument("--num_plans", type=int, default=8)
    parser.add_argument("--sid_beams", type=int, default=8)
    parser.add_argument("--max_history_len", type=int, default=50, choices=(10, 20, 50))
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--conditioning",
        choices=("history_visible", "interest_bottleneck"),
        default="interest_bottleneck",
    )
    parser.add_argument(
        "--interest_parameterization",
        choices=("independent_head", "disjoint_rows"),
        default="independent_head",
    )
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
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--eval_steps",
        type=float,
        default=0.1,
        help="Validation interval; values below 1 are a fraction of total training steps",
    )
    parser.add_argument(
        "--per_device_batch_size",
        "--train_batch_size",
        dest="per_device_batch_size",
        type=int,
        default=1,
        help="Optimization micro-batch per GPU",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        help="Global TRL generation batch; defaults to the effective optimizer batch",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logprob_micro_batch_size", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume_from_checkpoint",
        help="TRL checkpoint directory to resume, including DIPRec's interest-adapter sidecar",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
