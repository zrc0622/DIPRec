import json
import math
import re
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from diprec.prefix_reward import RL_TASKS, prefix_group_signal, validate_prefix_reward

TARGET = "<a_0><b_0><c_0>"
MISS = ["<a_0><b_0><c_1>", "<a_0><b_1><c_0>", "<a_1><b_0><c_0>", "<a_1><b_1><c_1>"]


class PrefixRewardTest(unittest.TestCase):
    def test_only_main_all_miss_groups_gain_bounded_centered_signal(self):
        completions = MISS + [TARGET, *MISS[:3]] + MISS * 3
        tasks = [task for task in [RL_TASKS[0], *RL_TASKS] for _ in range(4)]
        delta, groups = prefix_group_signal(completions, [TARGET]*20, tasks, 4, .1)
        for actual, expected in zip(delta[:4], [.0625, .0125, -.0375, -.0375]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(delta[4:], [0.]*16)
        self.assertAlmostEqual(sum(delta[:4]), 0.)
        self.assertLessEqual(max(map(abs, delta)), .1)
        self.assertEqual(sum(g['aux_active'] for g in groups), 1)
        smaller, _ = prefix_group_signal(MISS, [TARGET]*4, [RL_TASKS[0]]*4, 4, .01)
        for a, b in zip(smaller, delta):
            self.assertAlmostEqual(a, b/10)

    def test_zero_strength_uniform_scores_and_prefix_contiguity(self):
        for strength, candidates in [(0., MISS), (.1, [MISS[0]]*4), (.1, [MISS[2], MISS[3]]*2)]:
            signal, _ = prefix_group_signal(candidates, [TARGET]*4, [RL_TASKS[0]]*4, 4, strength)
            self.assertEqual(signal, [0.]*4)
        # Equal second-level code under a different first-level code earns nothing.
        _, groups = prefix_group_signal([MISS[2]]*4, [TARGET]*4, [RL_TASKS[0]]*4, 4, .1)
        self.assertEqual(groups[0]['prefix_mean'], 0.)

    def test_invalid_inputs_fail_before_silent_group_mixing(self):
        for bad in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_prefix_reward('main_miss_prefix', bad)
        for candidates, targets, tasks in [
            (MISS[:3], [TARGET]*3, [RL_TASKS[0]]*3),
            (MISS, [TARGET]*3 + [MISS[0]], [RL_TASKS[0]]*4),
            (MISS, [TARGET]*4, [RL_TASKS[0]]*3 + [RL_TASKS[1]]),
            (["<a_0><b_0>", *MISS[1:]], [TARGET]*4, [RL_TASKS[0]]*4),
            (MISS, [TARGET]*4, ['unknown']*4),
        ]:
            with self.assertRaises(ValueError):
                prefix_group_signal(candidates, targets, tasks, 4, .1)

    def test_g16_all_miss_signal_is_zero_mean_and_scales_linearly(self):
        candidates = [f'<a_0><b_0><c_{i}>' for i in range(1, 5)] + [
            f'<a_1><b_0><c_{i}>' for i in range(12)]
        signal, groups = prefix_group_signal(candidates, [TARGET]*16, [RL_TASKS[0]]*16, 16, .1)
        self.assertAlmostEqual(sum(signal), 0.)
        self.assertAlmostEqual(signal[0], .075)
        self.assertAlmostEqual(signal[-1], -.025)
        self.assertTrue(groups[0]['aux_active'])

    def test_pilot_stop_preserves_schedule_and_diagnostics_append(self):
        from diprec.rl_logging import RLDiagnosticsCallback, StopAfterStepsCallback
        callback = StopAfterStepsCallback(2)
        state = SimpleNamespace(global_step=1, max_steps=100, epoch=.01, is_world_process_zero=True)
        control = SimpleNamespace(should_training_stop=False)
        callback.on_step_end(None, state, control)
        self.assertFalse(control.should_training_stop)
        state.global_step = 2
        callback.on_step_end(None, state, control)
        self.assertTrue(control.should_training_stop)
        self.assertEqual(state.max_steps, 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'diagnostics.jsonl'
            writer = RLDiagnosticsCallback(path)
            writer.on_log(None, state, control, logs={'prefix_aux/history_sid_to_sid/aux_active': .5})
            writer.on_log(None, state, control, logs={'eval_loss': 1.})
            state.is_world_process_zero = False
            writer.on_log(None, state, control, logs={'loss': 2.})
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]['step'], 2)
            self.assertEqual(entries[0]['prefix_aux/history_sid_to_sid/aux_active'], .5)


class PrefixTRLLifecycleTest(unittest.TestCase):
    def test_prefix_signal_survives_group_split_and_backward(self):
        import torch
        import transformers.utils.import_utils as import_utils
        import_utils._mlx_available = False
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from trl import GRPOConfig, GRPOTrainer
        from diprec.baseline_grpo import _catalog_trainer_class, exact_match_reward, make_rank_aware_reward
        from diprec.constraints import build_sid_trie
        from diprec.rl_logging import StopAfterStepsCallback

        torch.set_num_threads(1)
        torch.manual_seed(42)
        world = int(os.environ.get('WORLD_SIZE', '1'))
        vocab = {s: i for i, s in enumerate(['<unk>', '<pad>', '<eos>', 'miss', 'hit', 'title', 'uniform',
            '<a_0>', '<a_1>', '<b_0>', '<b_1>', '<c_0>', '<c_1>'])}
        backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>', eos_token='<eos>')
        # Added SID tokens ensure concatenated SID encoding is atomic.
        tokenizer.add_tokens([token for token in vocab if token.startswith(('<a_', '<b_', '<c_'))])
        sid_map = {str(i): tuple(re.findall(r'<[^>]+>', sid)) for i, sid in enumerate([TARGET, *MISS])}
        with tempfile.TemporaryDirectory(prefix='prefix-lifecycle-') as directory:
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, attention_dropout=0., use_cache=False))
            model.save_pretrained(directory)
            model = Qwen3ForCausalLM.from_pretrained(directory)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            config = GRPOConfig(output_dir=directory+'/output', use_cpu=True, bf16=False,
                per_device_train_batch_size=2, gradient_accumulation_steps=8,
                num_generations=4, generation_batch_size=16*world, num_iterations=1,
                max_prompt_length=8, max_completion_length=4, temperature=1.,
                beta=.01, max_steps=2, learning_rate=1e-3, warmup_ratio=0.,
                save_strategy='no', report_to='none', logging_steps=1, gradient_checkpointing=False)
            catalog_class = _catalog_trainer_class(GRPOTrainer)
            captured = []

            class CapturingTrainer(catalog_class):
                def _generate_and_score_completions(self, inputs):
                    out = super()._generate_and_score_completions(inputs)
                    captured.append((inputs, out))
                    return out

            trainer = CapturingTrainer(model=model, processing_class=tokenizer,
                sid_trie=build_sid_trie(tokenizer, sid_map), reward_mode='main_miss_prefix',
                prefix_reward_strength=.1, reward_diagnostics=True,
                reward_funcs=[exact_match_reward, make_rank_aware_reward(4)], args=config,
                train_dataset=Dataset.from_list([dict(prompt=prompt, target_sid=TARGET,
                    task='title_to_sid' if prompt == 'title' else 'history_sid_to_sid',
                    sample_id=f'{i}:{prompt}') for i in range(world) for prompt in ('miss','hit','title','uniform')]))
            reference_before = [p.detach().clone() for p in trainer.ref_model.parameters()]
            calls = []

            def fake_generate(_model, input_ids, **kwargs):
                calls.append(len(input_ids))
                rows = []
                for ids in input_ids:
                    prompt = tokenizer.decode(ids, skip_special_tokens=True)
                    candidates = [TARGET, *MISS[:3]] if prompt == 'hit' else MISS
                    if prompt == 'uniform':
                        candidates = [MISS[2], MISS[3]]*2
                    for sid in candidates:
                        suffix = tokenizer.encode(sid, add_special_tokens=False) + [tokenizer.eos_token_id]
                        rows.append(torch.cat([ids, ids.new_tensor(suffix)]))
                return torch.stack(rows)

            trainer.add_callback(StopAfterStepsCallback(1))
            with mock.patch.object(type(model), 'generate', new=fake_generate):
                result = trainer.train()
            self.assertEqual(result.global_step, 1)
            self.assertEqual(trainer.state.max_steps, 2)
            self.assertEqual(calls, [4])  # one rollout per rank for eight microsteps
            self.assertEqual(len(captured), 1)
            inputs, out = captured[0]
            for start in range(0, len(inputs), 4):
                prompt = inputs[start]['prompt']
                actual = out['advantages'][start:start+4].cpu()
                if prompt == 'miss':
                    torch.testing.assert_close(actual, torch.tensor([.0625,.0125,-.0375,-.0375]))
                elif prompt in ('title', 'uniform'):
                    self.assertEqual(actual.tolist(), [0.]*4)
                else:
                    discounts = [1 / math.log2(i+2) for i in range(4)]
                    raw = torch.tensor([1., *[-d/sum(discounts) for d in discounts[1:]]])
                    torch.testing.assert_close(actual, (raw-raw.mean())/(raw.std()+1e-4))
            self.assertTrue(torch.all(out['completion_mask'] == 1))
            self.assertEqual(out['completion_ids'].shape[1], 4)  # EOS remains in loss
            # Every shuffled accumulation slice must retain the advantage
            # attached to its own prompt/completion, including the new signal.
            def scored_rows(batch):
                return sorted((tuple(p.tolist()), tuple(c.tolist()), float(a)) for p,c,a in
                    zip(batch['prompt_ids'], batch['completion_ids'], batch['advantages']))
            buffered_rows = sorted(row for batch in trainer._buffered_inputs for row in scored_rows(batch))
            self.assertEqual(buffered_rows, scored_rows(out))
            actual_all = trainer.accelerator.gather(out['advantages']).tolist()
            self.assertEqual(list(trainer._logs['advantages']), actual_all)
            policy = trainer.accelerator.unwrap_model(trainer.model)
            self.assertTrue(any(not torch.equal(p, before[n]) for n,p in policy.named_parameters()))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(reference_before, trainer.ref_model.parameters())))
            self.assertIn('prefix_aux/history_sid_to_sid/aux_active', trainer.state.log_history[0])
            # Evaluate same trajectories at lambda=0 and official mode: exact
            # equality of advantages, reference logps and loss at fixed policy.
            trainer.model.eval()
            trainer.prefix_reward_strength = 0.
            with mock.patch.object(type(model), 'generate', new=fake_generate):
                zero = trainer._generate_and_score_completions(inputs)
                trainer.reward_mode = 'official'
                original = trainer._generate_and_score_completions(inputs)
            for key in ('advantages', 'ref_per_token_logps', 'completion_ids', 'completion_mask'):
                self.assertTrue(torch.equal(zero[key], original[key]), key)
            trainer.reward_diagnostics = False
            with mock.patch.object(type(model), 'generate', new=fake_generate):
                default = trainer._generate_and_score_completions(inputs)
            self.assertTrue(torch.equal(default['advantages'], original['advantages']))


if __name__ == '__main__':
    unittest.main()
