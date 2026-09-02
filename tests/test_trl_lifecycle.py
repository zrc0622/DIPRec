import os
import json
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


class TRLLifecycleTest(unittest.TestCase):
    def test_custom_trainers_reject_sharded_backends_during_initialization(self):
        from diprec.baseline_grpo import _catalog_trainer_class
        from diprec.grpo import _diprec_trainer_class

        class FakeBase:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.use_vllm = False
                self.is_deepspeed_enabled = True
                self.is_fsdp_enabled = False
                self.is_tp_enabled = False

        with self.assertRaisesRegex(RuntimeError, "ordinary replicated DDP"):
            _catalog_trainer_class(FakeBase)(sid_trie=object())

        training_args = type(
            "Args",
            (),
            {
                "beta": 1e-3,
                "num_iterations": 2,
                "use_vllm": False,
                "use_liger_loss": False,
                "importance_sampling_level": "token",
                "per_device_train_batch_size": 1,
                "generation_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "steps_per_generation": 1,
                "world_size": 2,
            },
        )()
        current_model = type(
            "Model", (), {"config": type("Config", (), {"_name_or_path": "original"})()}
        )()
        with (
            mock.patch("diprec.grpo.load_model_runtime") as load_reference,
            self.assertRaisesRegex(RuntimeError, "ordinary replicated DDP"),
        ):
            _diprec_trainer_class(FakeBase)(
                model=current_model,
                args=training_args,
                sid_trie=object(),
                sid_map={},
                token_registry=object(),
                reference_model_path="reference",
                mode="plan_grpo",
                conditioning="interest_bottleneck",
                interest_parameterization="independent_head",
                interest_topk=1,
                sid_beams=2,
                max_history_len=10,
                max_seq_len=128,
                plan_temperature=1.0,
                plan_top_p=1.0,
                plan_sampling_attempts=2,
                reward_weights=object(),
                interest_loss_weight=1.0,
                sid_loss_weight=1.0,
                logprob_micro_batch_size=2,
            )
        load_reference.assert_not_called()
        self.assertEqual(training_args.beta, 1e-3)
        self.assertEqual(current_model.config._name_or_path, "original")

    def test_diprec_generation_uses_trl_unwrap_context(self):
        try:
            import trl.models
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"TRL generation helpers are unavailable: {exc}")
        from diprec.grpo import _diprec_trainer_class

        class FakeBase:
            pass

        trainer = object.__new__(_diprec_trainer_class(FakeBase))
        trainer.model_wrapped = object()
        trainer.accelerator = type("Accelerator", (), {"is_main_process": True})()
        payload = [{"sample_id": "0"}]
        sentinel = object()
        with (
            mock.patch(
                "trl.models.unwrap_model_for_generation",
                return_value=nullcontext(sentinel),
            ) as unwrap,
            mock.patch.object(
                trainer, "_generate_payload_with_model", return_value=[{"ok": True}]
            ) as generate,
        ):
            self.assertEqual(trainer._generate_payload(payload), [{"ok": True}])
        unwrap.assert_called_once_with(trainer.model_wrapped, trainer.accelerator)
        generate.assert_called_once_with(sentinel, payload)

    def test_non_main_rank_enters_unwrap_context_without_generating(self):
        try:
            import trl.models
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"TRL generation helpers are unavailable: {exc}")
        from diprec.grpo import _diprec_trainer_class

        class FakeBase:
            pass

        trainer = object.__new__(_diprec_trainer_class(FakeBase))
        trainer.model_wrapped = object()
        trainer.accelerator = type("Accelerator", (), {"is_main_process": False})()
        with (
            mock.patch(
                "trl.models.unwrap_model_for_generation",
                return_value=nullcontext(object()),
            ) as unwrap,
            mock.patch.object(trainer, "_generate_payload_with_model") as generate,
        ):
            self.assertEqual(trainer._generate_payload([{"sample_id": "0"}]), [])
        unwrap.assert_called_once_with(trainer.model_wrapped, trainer.accelerator)
        generate.assert_not_called()

    def test_diprec_trainer_keeps_adapter_and_fixed_reference(self):
        try:
            import torch
            import trl
            from datasets import Dataset
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import WhitespaceSplit
            from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
            from trl import GRPOConfig, GRPOTrainer
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"pinned TRL runtime is unavailable: {exc}")
        if trl.__version__ != "0.24.0":
            self.skipTest(f"requires trl==0.24.0, found {trl.__version__}")

        from diprec.constraints import build_sid_trie
        from diprec.grpo import _diprec_trainer_class, _unused_reward
        from diprec.rewards import RewardWeights
        from diprec.runtime import (
            get_active_interest_router,
            load_model_runtime,
            save_runtime,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            sft = root / "sft"
            vocabulary = {
                "<unk>": 0,
                "<pad>": 1,
                "<eos>": 2,
                "<think>": 3,
                "</think>": 4,
                "system": 5,
                "user": 6,
                "assistant": 7,
            }
            backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
            backend.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                unk_token="<unk>",
                pad_token="<pad>",
                eos_token="<eos>",
            )
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['role'] }} "
                "{{ message['content'] }} {% endfor %}assistant"
            )
            config = Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=128,
            )
            Qwen3ForCausalLM(config).save_pretrained(base)
            tokenizer.save_pretrained(base)
            sid_map = {
                "0": ("<a_0>", "<b_0>", "<c_0>"),
                "1": ("<a_1>", "<b_1>", "<c_1>"),
            }

            model, tokenizer, registry, router = load_model_runtime(
                str(base), sid_map, "independent_head", training=True
            )
            self.assertIsNotNone(registry)
            self.assertIsNotNone(router)
            save_runtime(model, tokenizer, router, sft, "independent_head")

            model, tokenizer, registry, current_router = load_model_runtime(
                str(sft), sid_map, "independent_head", training=True
            )
            self.assertIsNotNone(registry)
            self.assertIsNotNone(current_router)
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size not in (1, 2):
                self.skipTest(f"lifecycle fixture supports world sizes 1 and 2, found {world_size}")
            gradient_accumulation_steps = 2 // world_size
            training_args = GRPOConfig(
                output_dir=str(root / "output"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_generations=2,
                generation_batch_size=2,
                num_iterations=2,
                max_prompt_length=96,
                max_completion_length=3,
                beta=1e-3,
                importance_sampling_level="token",
                use_liger_loss=False,
                use_vllm=False,
                bf16=False,
                fp16=False,
                max_steps=2,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            )
            Trainer = _diprec_trainer_class(GRPOTrainer)
            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[_unused_reward],
                train_dataset=Dataset.from_list([{"prompt": "unused", "sample_id": "0"}]),
                args=training_args,
                sid_trie=build_sid_trie(tokenizer, sid_map),
                sid_map=sid_map,
                token_registry=registry,
                reference_model_path=str(sft),
                mode="plan_grpo",
                conditioning="interest_bottleneck",
                interest_parameterization="independent_head",
                interest_topk=1,
                sid_beams=2,
                max_history_len=10,
                max_seq_len=128,
                plan_temperature=1.0,
                plan_top_p=1.0,
                plan_sampling_attempts=2,
                reward_weights=RewardWeights(),
                interest_loss_weight=1.0,
                sid_loss_weight=1.0,
                logprob_micro_batch_size=2,
            )

            self.assertEqual(training_args.beta, 1e-3)
            self.assertEqual(training_args.world_size, world_size)
            self.assertEqual(
                training_args.steps_per_generation, gradient_accumulation_steps
            )
            self.assertEqual(
                training_args.gradient_checkpointing_kwargs,
                {"use_reentrant": False},
            )
            self.assertEqual(trainer.beta, 1e-3)
            self.assertIsNotNone(trainer.ref_model)
            self.assertIs(trainer.interest_router, current_router)
            self.assertIs(get_active_interest_router(), current_router)
            self.assertIsNot(trainer.reference_router, current_router)
            self.assertTrue(all(not parameter.requires_grad for parameter in trainer.ref_model.parameters()))
            self.assertFalse(trainer.ref_model.is_gradient_checkpointing)

            adapter_ids = {id(parameter) for parameter in current_router.adapter.parameters()}
            model_ids = {id(parameter) for parameter in trainer.model.parameters()}
            self.assertTrue(adapter_ids <= model_ids)
            optimizer = trainer.create_optimizer()
            optimizer_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            self.assertTrue(adapter_ids <= optimizer_ids)
            self.assertTrue(all(parameter.requires_grad for parameter in current_router.adapter.parameters()))
            sampler = trainer._get_train_sampler()
            self.assertEqual(sampler.mini_repeat_count, 2)
            self.assertEqual(sampler.batch_size, 1)
            self.assertEqual(
                sampler.repeat_count,
                training_args.num_iterations * training_args.steps_per_generation,
            )
            checkpoint = root / "checkpoint-test"
            trainer._save(str(checkpoint))
            self.assertTrue((checkpoint / "diprec_adapter_config.json").is_file())
            self.assertTrue((checkpoint / "diprec_interest_adapter.pt").is_file())
            self.assertTrue((checkpoint / "training_args.bin").is_file())

            saved_adapter = {
                name: value.detach().clone()
                for name, value in current_router.adapter.state_dict().items()
            }
            adapter_parameters = list(current_router.adapter.parameters())
            with torch.no_grad():
                for parameter in adapter_parameters:
                    parameter.add_(1.0)
            self.assertTrue(
                any(
                    not torch.equal(saved_adapter[name], value)
                    for name, value in current_router.adapter.state_dict().items()
                )
            )
            trainer._load_from_checkpoint(str(checkpoint))
            self.assertEqual(
                [id(parameter) for parameter in adapter_parameters],
                [id(parameter) for parameter in current_router.adapter.parameters()],
            )
            for name, value in current_router.adapter.state_dict().items():
                self.assertTrue(torch.equal(saved_adapter[name], value), name)
            self.assertTrue(adapter_ids <= optimizer_ids)

            # Honor GRPOConfig's binary checkpoint option as well: stock
            # Trainer must find the base weights and then load the sidecar.
            trainer.args.save_safetensors = False
            binary_checkpoint = root / "checkpoint-binary"
            trainer._save(str(binary_checkpoint))
            self.assertTrue((binary_checkpoint / "pytorch_model.bin").is_file())
            with torch.no_grad():
                for parameter in adapter_parameters:
                    parameter.add_(3.0)
            trainer._load_from_checkpoint(str(binary_checkpoint))
            for name, value in current_router.adapter.state_dict().items():
                self.assertTrue(torch.equal(saved_adapter[name], value), name)
            trainer.args.save_safetensors = True

            # Transformers' load_best_model_at_end uses a separate loader,
            # so it must restore the same adapter sidecar rather than leaving
            # the final-step adapter paired with the best base-model weights.
            with torch.no_grad():
                for parameter in adapter_parameters:
                    parameter.add_(2.0)
            trainer.state.best_model_checkpoint = str(checkpoint)
            trainer._load_best_model()
            self.assertEqual(
                [id(parameter) for parameter in adapter_parameters],
                [id(parameter) for parameter in current_router.adapter.parameters()],
            )
            for name, value in current_router.adapter.state_dict().items():
                self.assertTrue(torch.equal(saved_adapter[name], value), name)

            missing_sidecar = root / "checkpoint-missing-adapter"
            shutil.copytree(checkpoint, missing_sidecar)
            (missing_sidecar / "diprec_interest_adapter.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "interest adapter weights"):
                trainer._load_from_checkpoint(str(missing_sidecar))

            bad_config = root / "checkpoint-bad-adapter-config"
            shutil.copytree(checkpoint, bad_config)
            config_path = bad_config / "diprec_adapter_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["interest_token_ids"] = [-1]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "interest token IDs"):
                trainer._load_from_checkpoint(str(bad_config))

            target_tokens = list(sid_map["0"])
            other_tokens = list(sid_map["1"])
            target_ids = [tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            other_ids = [tokenizer.convert_tokens_to_ids(token) for token in other_tokens]
            interest_ids = list(registry.interest_token_ids)
            before = [parameter.detach().clone() for parameter in current_router.adapter.parameters()]

            def fake_plans(*_args, **_kwargs):
                return [5], [[interest_ids[0]], [interest_ids[1]]], [
                    [registry.interest_tokens[0]],
                    [registry.interest_tokens[1]],
                ]

            def fake_candidates(*call_args, **_kwargs):
                plan = call_args[4]
                sid_context = [6, tokenizer.convert_tokens_to_ids(plan[0])]
                if plan == [registry.interest_tokens[0]]:
                    return sid_context, [target_ids, other_ids], [target_tokens, other_tokens], [True, True]
                return sid_context, [other_ids, other_ids], [other_tokens, other_tokens], [True, True]

            trainer.train_dataset = Dataset.from_list(
                [
                    {
                        "prompt": "unused",
                        "sample_id": "0",
                        "target_sid_levels": target_tokens,
                    }
                ]
            )
            with (
                mock.patch(
                    "diprec.grpo._generate_plans", side_effect=fake_plans
                ) as generate_plans,
                mock.patch(
                    "diprec.grpo._generate_sid_candidates", side_effect=fake_candidates
                ) as generate_candidates,
            ):
                result = trainer.train()

            self.assertEqual(result.global_step, 2)
            after = list(current_router.adapter.parameters())
            changed = [
                not torch.equal(previous, current.detach())
                for previous, current in zip(before, after)
            ]
            self.assertTrue(all(changed), changed)
            expected_generation_calls = 1 if trainer.accelerator.is_main_process else 0
            self.assertEqual(generate_plans.call_count, expected_generation_calls)
            self.assertEqual(generate_candidates.call_count, 2 * expected_generation_calls)
            if world_size == 1:
                with (
                    mock.patch("diprec.grpo._generate_plans", side_effect=fake_plans),
                    mock.patch(
                        "diprec.grpo._generate_sid_candidates",
                        side_effect=fake_candidates,
                    ),
                ):
                    metrics = trainer.evaluate(
                        Dataset.from_list(
                            [
                                {
                                    "prompt": "unused",
                                    "sample_id": "valid-0",
                                    "target_sid_levels": target_tokens,
                                }
                            ]
                        )
                    )
                self.assertIn("eval_loss", metrics)
            if world_size == 2:
                import torch.distributed as dist

                local_batch = trainer._buffered_inputs[0]
                local_plan = int(local_batch["plan_completion_ids"][0, 0].item())
                local_reward = float(local_batch["plan_rewards"][0].item())
                gathered: list[tuple[int, float] | None] = [None, None]
                dist.all_gather_object(gathered, (local_plan, local_reward))
                self.assertEqual([value[0] for value in gathered], interest_ids)
                self.assertGreater(gathered[0][1], gathered[1][1])

    def test_catalog_trainer_distributes_beams_and_rewards_by_rank(self):
        try:
            import torch
            import trl
            from datasets import Dataset
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import WhitespaceSplit
            from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
            from trl import GRPOConfig, GRPOTrainer
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"pinned TRL runtime is unavailable: {exc}")
        if trl.__version__ != "0.24.0":
            self.skipTest(f"requires trl==0.24.0, found {trl.__version__}")

        from diprec.baseline_grpo import (
            _catalog_trainer_class,
            exact_match_reward,
            make_rank_aware_reward,
        )
        from diprec.constraints import build_sid_trie
        from diprec.runtime import load_model_runtime

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size not in (1, 2):
            self.skipTest(f"lifecycle fixture supports world sizes 1 and 2, found {world_size}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            sft = root / "sft"
            vocabulary = {
                "<unk>": 0,
                "<pad>": 1,
                "<eos>": 2,
                "system": 3,
                "user": 4,
                "assistant": 5,
                "history": 6,
            }
            backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
            backend.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                unk_token="<unk>",
                pad_token="<pad>",
                eos_token="<eos>",
            )
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['role'] }} "
                "{{ message['content'] }} {% endfor %}assistant"
            )
            config = Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=128,
            )
            Qwen3ForCausalLM(config).save_pretrained(base)
            tokenizer.save_pretrained(base)
            sid_map = {
                "0": ("<a_0>", "<b_0>", "<c_0>"),
                "1": ("<a_1>", "<b_1>", "<c_1>"),
            }
            model, tokenizer, _, _ = load_model_runtime(
                str(base), sid_map, "disjoint_rows", training=True, include_interest=False
            )
            model.save_pretrained(sft)
            tokenizer.save_pretrained(sft)
            model, tokenizer, _, _ = load_model_runtime(
                str(sft), sid_map, "disjoint_rows", training=True, include_interest=False
            )

            gradient_accumulation_steps = 2 // world_size
            training_args = GRPOConfig(
                output_dir=str(root / "output"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_generations=2,
                generation_batch_size=2,
                num_iterations=1,
                max_prompt_length=96,
                max_completion_length=4,
                beta=1e-3,
                sync_ref_model=True,
                ref_model_sync_steps=1,
                ref_model_mixup_alpha=1.0,
                use_vllm=False,
                bf16=False,
                fp16=False,
                max_steps=1,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            )
            target = "".join(sid_map["0"])
            trainer = _catalog_trainer_class(GRPOTrainer)(
                model=model,
                processing_class=tokenizer,
                sid_trie=build_sid_trie(tokenizer, sid_map),
                reward_funcs=[exact_match_reward, make_rank_aware_reward(2)],
                train_dataset=Dataset.from_list(
                    [{"prompt": "history", "target_sid": target, "sample_id": "0"}]
                ),
                args=training_args,
            )
            sampler = trainer._get_train_sampler()
            self.assertTrue(
                any(
                    callback.__class__.__name__ == "SyncRefModelCallback"
                    for callback in trainer.callback_handler.callbacks
                )
            )
            self.assertEqual(training_args.steps_per_generation, gradient_accumulation_steps)
            self.assertEqual(sampler.mini_repeat_count, 2)
            self.assertEqual(sampler.batch_size, 1)
            self.assertEqual(
                sampler.repeat_count,
                training_args.num_iterations * training_args.steps_per_generation,
            )

            target_ids = [tokenizer.convert_tokens_to_ids(token) for token in sid_map["0"]]
            other_ids = [tokenizer.convert_tokens_to_ids(token) for token in sid_map["1"]]
            generate_calls = 0

            def fake_generate(_model, input_ids, **_kwargs):
                nonlocal generate_calls
                generate_calls += 1
                rows = []
                completions = (target_ids, other_ids)
                for prompt_ids in input_ids:
                    for completion in completions:
                        suffix = torch.tensor(
                            [*completion, tokenizer.eos_token_id],
                            dtype=prompt_ids.dtype,
                            device=prompt_ids.device,
                        )
                        rows.append(torch.cat([prompt_ids, suffix]))
                return torch.stack(rows)

            with mock.patch.object(type(model), "generate", new=fake_generate):
                result = trainer.train()

            self.assertEqual(result.global_step, 1)
            policy = trainer.accelerator.unwrap_model(trainer.model)
            reference = trainer.accelerator.unwrap_model(trainer.ref_model)
            self.assertTrue(
                all(
                    torch.equal(policy_parameter, reference_parameter)
                    for policy_parameter, reference_parameter in zip(
                        policy.parameters(), reference.parameters()
                    )
                )
            )
            expected_generation_calls = 1 if trainer.accelerator.is_main_process else 0
            self.assertEqual(generate_calls, expected_generation_calls)
            self.assertEqual(
                list(trainer._logs["rewards"]["exact_match_reward"]), [1.0, 0.0]
            )
            if world_size == 1:
                with mock.patch.object(type(model), "generate", new=fake_generate):
                    metrics = trainer.evaluate(
                        Dataset.from_list(
                            [{"prompt": "history", "target_sid": target, "sample_id": "valid-0"}]
                        )
                    )
                self.assertIn("eval_loss", metrics)
            if world_size == 2:
                import torch.distributed as dist

                local_advantage = float(
                    trainer._buffered_inputs[0]["advantages"][0].item()
                )
                gathered: list[float | None] = [None, None]
                dist.all_gather_object(gathered, local_advantage)
                self.assertGreater(gathered[0], gathered[1])


if __name__ == "__main__":
    unittest.main()
