import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_rl_optimization_sweep as sweep


class OptimizationSweepTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        root_patch = patch.object(sweep, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.directory = self.root / 'outputs' / sweep.BASE / 'rl_optimization_sweeps' / 'test'
        self.args = ['--sweep_tag', 'test', '--probe_samples', '2']
        self.save_checkpoint(self.root / sweep.SFT, dict(method='minionerec_sft'))
        for path in (sweep.DATA / 'train.jsonl', sweep.DATA / 'valid.jsonl', sweep.DATA / 'manifest.json', sweep.INDEX, sweep.ITEMS):
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text('{}')

    def save_checkpoint(self, path, cfg):
        path.mkdir(parents=True, exist_ok=True)
        sweep.write_json(path / 'training_config.json', cfg)
        for name in ('config.json', 'model.safetensors'):
            (path / name).write_text('{}')

    def call(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()):
            sweep.main(self.args + list(extra))

    def fake_execute(self, cmd, env, log):
        def value(flag):
            return cmd[cmd.index(flag)+1]
        if cmd[0] == 'bash':
            self.assertIn('--require_existing_sft', cmd)
            self.assertEqual(value('--baseline_rl_snapshot_steps'), '1000,2000')
            self.assertEqual(value('--baseline_rl_stop_after_steps'), '0')
            self.assertEqual(value('--baseline_rl_reward_mode'), 'official')
            self.assertEqual(env['DIPREC_NUM_PROCESSES'], '4')
            tag = value('--run_tag')
            arm = next(a for a in sweep.ARMS if f"_{a['label']}_a" in tag) | dict(run_tag=tag)
            for step in sweep.STEPS:
                cfg = dict(method='minionerec_rl', model=str(sweep.SFT), seed=42, reward_mode='official',
                           completed_optimizer_steps=step, stop_after_steps=0, scheduler_total_steps=3455,
                           snapshot_steps=[1000,2000], num_epochs=1, task_scope='official_mixed',
                           reference_mode=arm['reference_mode'], learning_rate=arm['learning_rate'], beta=arm['beta'],
                           ref_model_sync_steps=512, ref_model_mixup_alpha=.6,
                           num_generations=16, num_iterations=1, temperature=1.0,
                           batch=dict(world_size=4, effective_update_batch=256, generation_batch_size=256,
                                      per_device_batch_size=16, gradient_accumulation_steps=4))
                if step != sweep.STEPS[-1]:
                    cfg.update(snapshot_model_only=True, checkpoint_role='rl_milestone')
                self.save_checkpoint(sweep.checkpoint(arm, step), cfg)
            out = sweep.run_dir(tag)
            out.mkdir(parents=True)
            row = {'grad_norm':1., 'kl':.01}
            for task in sweep.TASKS:
                row[f'prefix_aux/{task}/groups'] = 4
                row[f'prefix_aux/{task}/exact_hit'] = .25
            (out / 'rl_diagnostics.jsonl').write_text(''.join(json.dumps(row | dict(step=i))+'\n' for i in range(1,3456)))
            return
        dest = self.root / value('--output')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if 'prepare' in cmd:
            sweep.write_json(dest, dict(schema='diprec.rl_probe.v1', samples_per_split=2))
            return
        model = value('--model')
        cfg = sweep.read_json(self.root / model / 'training_config.json')
        if cmd[1].endswith('probe_baseline_rl.py'):
            metrics = dict(samples=2, target_logp=-2., target_margin=-.2, candidate_set_kl_sft_to_policy=.01)
            sweep.write_json(dest, dict(checkpoint=model, training_config=cfg,
                probe_sha256=sweep.digest(self.root / value('--probe')), summary=dict(train=metrics, valid=metrics)))
        else:
            data = dict(checkpoint=model, training_config=cfg, split='valid', num_examples=2,
                        data_manifest={'same':'source'}, metrics={key:.2 for key in sweep.METRICS},
                        evaluation_config=dict(eval_beams=10, eval_candidate_budget=80, max_history_len=50, max_seq_len=2048))
            sweep.write_json(dest, data)
            (dest.parent / 'valid_predictions.jsonl').write_text(''.join(json.dumps({'sample_id':str(i)})+'\n' for i in range(2)))

    def test_five_commands_fixed_protocol_and_no_dry_run_writes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(sweep, 'execute') as execute:
            sweep.main(['--sweep_tag','dry','--dry_run'])
        execute.assert_not_called()
        self.assertFalse((self.root / 'outputs').exists())
        commands = [line for line in out.getvalue().splitlines() if line.startswith('bash ')]
        self.assertEqual(len(commands), 5)
        for arm, cmd in zip(sweep.ARMS, commands):
            self.assertIn('--baseline_rl_reference_mode '+arm['reference_mode'], cmd)
            self.assertIn('--baseline_rl_beta '+str(arm['beta']), cmd)
            self.assertIn('--baseline_rl_learning_rate '+str(arm['learning_rate']), cmd)
            self.assertIn('--baseline_rl_snapshot_steps 1000,2000', cmd)
            self.assertIn('--baseline_rl_reward_mode official', cmd)
            self.assertNotIn('similarity', cmd)

    def test_training_complete_evaluation_failure_resumes_without_retraining(self):
        def fail_eval(cmd, env, log):
            if cmd[1].endswith('evaluate_diprec.py') and 'step_1000' in str(log):
                raise RuntimeError('evaluation failed')
            self.fake_execute(cmd, env, log)
        with patch.object(sweep, 'execute', side_effect=fail_eval), self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
            self.call()
        with patch.object(sweep, 'execute', side_effect=self.fake_execute) as execute:
            self.call(['--resume'])
        commands = [c.args[0] for c in execute.call_args_list if c.args[0][0]=='bash']
        self.assertEqual(len(commands), 4)
        state = sweep.read_json(self.directory / 'state.json')
        self.assertTrue(all(a['status']=='complete' for a in state['arms']))
        self.assertTrue(all(len(a['attempts'])==1 for a in state['arms']))
        summary = sweep.read_json(self.directory / 'summary.json')
        self.assertEqual(len(summary['results']), 15)
        self.assertEqual(len(summary['comparisons']), 15)
        self.assertEqual(summary['results'][0]['history_sid_to_sid_exact_hit_group_fraction'], .25)
        with patch.object(sweep, 'execute') as execute:
            self.call(['--resume'])
            self.call(['--summarize_only'])
        execute.assert_not_called()

    def test_training_failure_creates_new_attempt(self):
        def fail_train(cmd, env, log):
            if cmd[0]=='bash':
                raise RuntimeError('training failed')
            self.fake_execute(cmd, env, log)
        with patch.object(sweep, 'execute', side_effect=fail_train), self.assertRaisesRegex(RuntimeError,'training failed'):
            self.call()
        with patch.object(sweep, 'execute', side_effect=self.fake_execute):
            self.call(['--resume'])
        state = sweep.read_json(self.directory / 'state.json')
        self.assertEqual(len(state['arms'][0]['attempts']), 2)
        self.assertTrue(state['arms'][0]['run_tag'].endswith('_a2'))

    def test_changed_parent_and_completed_metrics_rejected_before_launch(self):
        with patch.object(sweep, 'execute', side_effect=self.fake_execute):
            self.call()
        weights = self.root / sweep.SFT / 'model.safetensors'
        weights.write_text('changed')
        with patch.object(sweep,'execute') as execute, self.assertRaisesRegex(ValueError,'input data changed'):
            self.call(['--resume'])
        execute.assert_not_called()
        weights.write_text('{}')
        state = sweep.read_json(self.directory / 'state.json')
        dest = sweep.result_dir(state['arms'][0],1000) / 'valid_metrics.json'
        data = sweep.read_json(dest)
        data['training_config']['completed_optimizer_steps']=999
        sweep.write_json(dest,data)
        with patch.object(sweep,'execute') as execute, self.assertRaises(ValueError):
            self.call(['--resume'])
        execute.assert_not_called()

    def test_partial_predictions_are_reevaluated(self):
        with patch.object(sweep, 'execute', side_effect=self.fake_execute):
            self.call()
        state = sweep.read_json(self.directory / 'state.json')
        dest = sweep.result_dir(state['arms'][0],1000)
        (dest / 'evaluation_complete.json').unlink()
        (dest / 'valid_predictions.jsonl').write_text('{"sample_id":"0"}\n')
        # An unmarked incomplete evaluation must not be included in summaries.
        with patch.object(sweep, 'execute', side_effect=self.fake_execute) as execute:
            self.call(['--resume'])
        self.assertEqual(execute.call_count, 1)

    def test_summary_works_after_copying_only_lightweight_outputs(self):
        import shutil
        with patch.object(sweep, 'execute', side_effect=self.fake_execute):
            self.call()
        shutil.rmtree(self.root / 'output_dir')
        shutil.rmtree(self.root / 'data')
        with patch.object(sweep, 'execute') as execute:
            self.call(['--summarize_only', '--gpus', '0'])
        execute.assert_not_called()
        self.assertEqual(len(sweep.read_json(self.directory / 'summary.json')['results']),15)


if __name__ == '__main__':
    unittest.main()
