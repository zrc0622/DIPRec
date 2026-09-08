import math
import unittest

from diprec.rl_probe import probe_statistics, score_sequences, select_records


class ProbeStatisticsTest(unittest.TestCase):
    def test_fixed_margin_and_kl(self):
        baseline = [-4., -3., -5.]
        identical = probe_statistics(baseline, baseline)
        self.assertAlmostEqual(identical['candidate_set_kl_sft_to_policy'], 0.)
        self.assertEqual(identical['target_margin'], -1.)
        improved = probe_statistics([-2.,-3.,-5.],baseline)
        self.assertEqual(improved['delta_target_logp'],2.)
        self.assertEqual(improved['delta_target_margin'],2.)
        self.assertEqual(improved['fixed_candidate_rank'],1)
        self.assertAlmostEqual(improved['target_probability'],math.exp(-2.))
        # Uniform score shifts leave normalized candidate KL unchanged but
        # change the full-vocabulary target likelihood and absolute drift.
        shifted = probe_statistics([x-1 for x in baseline],baseline)
        self.assertAlmostEqual(shifted['candidate_set_kl_sft_to_policy'],0.)
        self.assertEqual(shifted['candidate_logp_abs_change'],1.)

    def test_sampling_is_stable_under_input_reordering(self):
        records = [dict(sample_id=str(i)) for i in range(50)]
        selected = select_records(records,8,42,'train')
        self.assertEqual(selected,select_records(records[::-1],8,42,'train'))
        self.assertNotEqual(selected,select_records(records,8,42,'valid'))
        with self.assertRaises(ValueError):
            select_records(records+records[:1],8,42,'train')


class ProbeScoringTest(unittest.TestCase):
    def test_prepare_and_evaluate_with_real_checkpoint_and_frozen_negatives(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from diprec.data import processed_data_fingerprint
        from diprec.rl_probe import digest, evaluate, prepare

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            sid = {str(i):[f'<a_{i}>',f'<b_{i}>',f'<c_{i}>'] for i in range(80)}
            index, items = root / 'index.json', root / 'items.json'
            index.write_text(json.dumps(sid))
            items.write_text('{}')
            manifest = dict(source_kind='raw_event_interactions', max_history_len=50,
                            sid_index_sha256=digest(index), dataset='Office_Products')
            (data / 'manifest.json').write_text(json.dumps(manifest))
            record = dict(sample_id='sample',dataset='Office_Products',history_item_id=['1'],
                          history_item_sid=[''.join(sid['1'])],history_len=1,history_len_before_truncation=1,
                          target_item_sid=''.join(sid['0']))
            for split in ('train','valid'):
                (data / f'{split}.jsonl').write_text(json.dumps(record)+'\n')
            backend = Tokenizer(WordLevel({'<unk>':0,'<pad>':1,'<eos>':2},unk_token='<unk>'))
            backend.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',pad_token='<pad>',eos_token='<eos>')
            tokenizer.model_input_names = ['input_ids','attention_mask']
            tokenizer.add_tokens([s for values in sid.values() for s in values])
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,
                     num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
                     eos_token_id=2,pad_token_id=1))
            sft = root / 'sft'
            model.save_pretrained(sft)
            tokenizer.save_pretrained(sft)
            (sft / 'training_config.json').write_text(json.dumps(dict(method='minionerec_sft',
                data_manifest=processed_data_fingerprint(manifest),item_meta_sha256=digest(items))))
            args = SimpleNamespace(data_dir=str(data),sid_index=str(index),item_meta=str(items),samples=1,
                                   sft=str(sft),output=str(root / 'probe.json'))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
                frozen = json.loads(Path(args.output).read_text())
                for row in frozen['rows']:
                    self.assertEqual(len(set(row['candidate_sids'])),6)
                    self.assertEqual(row['candidate_sids'][0],record['target_item_sid'])
                args.probe, args.model, args.output = args.output, str(sft), str(root / 'metrics.json')
                evaluate(args)
            result = json.loads(Path(args.output).read_text())
            for split in ('train','valid'):
                self.assertEqual(result['summary'][split]['samples'],1)
                self.assertAlmostEqual(result['summary'][split]['delta_target_logp'],0.)
                self.assertAlmostEqual(result['summary'][split]['candidate_set_kl_sft_to_policy'],0.)

    def test_qwen_scores_match_teacher_forced_sid_and_eos_loss(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=20,hidden_size=16,intermediate_size=32,
                    num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
        prompt, continuations = [4,5,6], [[10,11,12,2],[13,14,15,2]]
        actual = score_sequences(model,prompt,continuations)
        for score, continuation in zip(actual,continuations):
            ids = torch.tensor([prompt+continuation])
            labels = ids.clone()
            labels[:,:len(prompt)] = -100
            with torch.inference_mode():
                output = model(input_ids=ids,labels=labels)
            self.assertAlmostEqual(score,-output.loss.item()*len(continuation),places=5)
        self.assertFalse(model.training)


if __name__ == '__main__':
    unittest.main()
