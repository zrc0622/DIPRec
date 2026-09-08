import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch


class SnapshotLifecycleTest(unittest.TestCase):
    def test_budget_guard_fails_before_updates(self):
        from types import SimpleNamespace
        from diprec.rl_logging import MilestoneSnapshotCallback
        callback = MilestoneSnapshotCallback(None, [], None, expected_steps=3455)
        with self.assertRaisesRegex(ValueError,'Expected 3455 optimizer steps'):
            callback.on_train_begin(SimpleNamespace(output_dir='unused'),SimpleNamespace(max_steps=3000),None)

    def test_snapshots_preserve_grpo_rng_updates_and_sync_reference(self):
        import numpy as np
        import torch
        from accelerate import PartialState
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainerCallback, set_seed
        from trl import GRPOConfig, GRPOTrainer
        from diprec.baseline_grpo import _catalog_trainer_class, exact_match_reward, make_rank_aware_reward
        from diprec.constraints import build_sid_trie
        from diprec.rl_logging import MilestoneSnapshotCallback

        distributed = PartialState(cpu=True)
        world = distributed.num_processes
        temporary = tempfile.TemporaryDirectory() if distributed.is_main_process else None
        paths = [temporary.name if temporary else None]
        if world > 1:
            torch.distributed.broadcast_object_list(paths, src=0)
        root = Path(paths[0])
        try:
            vocabulary = {'<unk>':0,'<pad>':1,'<eos>':2,'history':3}
            backend = Tokenizer(WordLevel(vocabulary,unk_token='<unk>'))
            backend.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',pad_token='<pad>',eos_token='<eos>')
            tokenizer.model_input_names = ['input_ids', 'attention_mask']
            sid_map = {'0':('<a_0>','<b_0>','<c_0>'),'1':('<a_1>','<b_1>','<c_1>')}
            tokenizer.add_tokens([s for sid in sid_map.values() for s in sid])
            set_seed(42)
            if distributed.is_main_process:
                model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,
                    num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
                    max_position_embeddings=128,pad_token_id=1,eos_token_id=2))
                model.save_pretrained(root / 'sft')
                tokenizer.save_pretrained(root / 'sft')
            distributed.wait_for_everyone()
            trajectories = []
            for snapshots in (False,True):
                set_seed(42)
                model = Qwen3ForCausalLM.from_pretrained(root / 'sft')
                args = GRPOConfig(output_dir=str(root / str(snapshots) / 'final_checkpoint'),use_cpu=True,
                    per_device_train_batch_size=2,generation_batch_size=2*world,num_generations=2,num_iterations=1,
                    max_prompt_length=32,max_completion_length=4,beta=.01,learning_rate=1e-3,
                    sync_ref_model=True,ref_model_sync_steps=2,ref_model_mixup_alpha=.6,
                    max_steps=3,logging_steps=1,save_strategy='no',report_to='none',bf16=False,fp16=False,
                    disable_tqdm=True,gradient_checkpointing=False)
                trainer = _catalog_trainer_class(GRPOTrainer)(model=model,processing_class=tokenizer,
                    sid_trie=build_sid_trie(tokenizer,sid_map),args=args,
                    reward_funcs=[exact_match_reward,make_rank_aware_reward(2)],
                    train_dataset=Dataset.from_list([{'prompt':'history','target_sid':''.join(sid_map['0']),
                        'sample_id':str(i)} for i in range(8*world)]))
                trajectory = []

                def copy_model(model):
                    return {k:v.detach().cpu().clone() for k,v in trainer.accelerator.unwrap_model(model).state_dict().items()}

                class Observer(TrainerCallback):
                    def on_step_end(self, args, state, control, **kwargs):
                        trajectory.append(dict(policy=copy_model(trainer.model),reference=copy_model(trainer.ref_model),
                            torch_rng=torch.get_rng_state().clone(),python_rng=random.getstate(),numpy_rng=np.random.get_state()))

                if snapshots:
                    trainer.add_callback(MilestoneSnapshotCallback(trainer,[1,2],
                        lambda completed,total:dict(completed_optimizer_steps=completed,scheduler_total_steps=total)))
                trainer.add_callback(Observer())
                save = trainer.save_model

                def save_with_rng(*args,**kwargs):
                    # A future serializer may consume random values. The snapshot
                    # callback must isolate all three generators on every rank.
                    random.random()
                    np.random.random()
                    torch.rand(2)
                    return save(*args,**kwargs)

                with patch.object(trainer,'save_model',side_effect=save_with_rng):
                    trainer.train()
                self.assertEqual(trainer.state.global_step,3)
                self.assertEqual(trainer.state.max_steps,3)
                self.assertTrue(any(row.get('grad_norm',0) > 0 for row in trainer.state.log_history))
                trajectories.append(trajectory)
                if snapshots:
                    for step in (1,2):
                        path = root / 'True' / f'step_{step}'
                        saved = Qwen3ForCausalLM.from_pretrained(path).state_dict()
                        for key,value in saved.items():
                            self.assertTrue(torch.equal(value,trajectory[step-1]['policy'][key]),key)
                        self.assertTrue((path / 'tokenizer.json').is_file())
                        metadata = json.loads((path / 'training_config.json').read_text())
                        self.assertEqual(metadata['completed_optimizer_steps'],step)
                        self.assertTrue(metadata['snapshot_model_only'])
                        self.assertFalse((path / 'optimizer.pt').exists())
            for before,after in zip(*trajectories):
                for kind in ('policy','reference'):
                    for key in before[kind]:
                        self.assertTrue(torch.equal(before[kind][key],after[kind][key]),f'{kind}:{key}')
                self.assertTrue(torch.equal(before['torch_rng'],after['torch_rng']))
                self.assertEqual(before['python_rng'],after['python_rng'])
                self.assertTrue(np.array_equal(before['numpy_rng'][1],after['numpy_rng'][1]))
                self.assertEqual(before['numpy_rng'][2:],after['numpy_rng'][2:])
            reference_states = [row['reference'] for row in trajectories[0]]
            self.assertTrue(any(not torch.equal(reference_states[0][k],reference_states[1][k]) for k in reference_states[0]))
            self.assertTrue(all(torch.equal(reference_states[1][k],reference_states[2][k]) for k in reference_states[0]))
        finally:
            distributed.wait_for_everyone()
            if temporary:
                temporary.cleanup()


if __name__ == '__main__':
    unittest.main()
