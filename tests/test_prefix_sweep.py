import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_prefix_sweep as sweep


class PrefixSweepTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch_root = patch.object(sweep, 'ROOT', self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.directory = self.root / 'outputs' / sweep.BASE / 'prefix_sweeps' / 'test'
        self.args = ['--sweep_tag', 'test', '--gpus', '0,1,2,3', '--strengths', '.3', '.5']
        parent = self.root / sweep.SFT
        parent.mkdir(parents=True)
        for name in ('config.json', 'training_config.json', 'model.safetensors'):
            (parent / name).write_text('{}')

    def call(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()):
            sweep.main(self.args + list(extra))

    def emit_result(self, cmd, env, log):
        def value(flag):
            return cmd[cmd.index(flag) + 1]

        self.assertIn('--require_existing_sft', cmd)
        self.assertEqual(value('--eval_split'), 'valid')
        self.assertEqual(value('--sft_run_tag'), sweep.SFT_TAG)
        self.assertEqual(env['DIPREC_NUM_PROCESSES'], '4')
        tag = value('--run_tag')
        cfg = dict(method='minionerec_rl', model=str(sweep.SFT), seed=42,
                   reward_mode=value('--baseline_rl_reward_mode'),
                   prefix_reward_strength=float(value('--baseline_rl_prefix_reward_strength')),
                   completed_optimizer_steps=1000, stop_after_steps=1000, scheduler_total_steps=3455,
                   learning_rate=2e-6, beta=.01, num_epochs=1, task_scope='official_mixed',
                   reference_mode='fixed', num_generations=16, num_iterations=1, temperature=1.0,
                   batch=dict(world_size=4, effective_update_batch=256, generation_batch_size=256,
                              per_device_batch_size=16, gradient_accumulation_steps=4))
        data = dict(split='valid', num_examples=50, training_config=cfg, data_manifest={'valid': 'same'},
                    evaluation_config={'eval_beams': 10}, metrics={key: .2 for key in sweep.METRICS})
        destination = sweep.run_dir(tag)
        destination.mkdir(parents=True)
        sweep.write_json(destination / 'valid_metrics.json', data)
        prefix = 'prefix_aux/history_sid_to_sid/'
        row = {'grad_norm': 1, 'kl': .01, prefix + 'groups': 16,
               prefix + 'exact_hit': .5, prefix + 'aux_active': .25,
               prefix + 'aux_abs_mean': .03}
        (destination / 'rl_diagnostics.jsonl').write_text(''.join(
            json.dumps(row | {'step': i}) + '\n' for i in range(1, 1001)))

    def test_dry_run_is_four_fixed_commands_without_writes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(sweep, 'execute') as execute:
            sweep.main(['--sweep_tag', 'dry', '--gpus', '0,1,2,3', '--dry_run'])
        execute.assert_not_called()
        self.assertFalse((self.root / 'outputs').exists())
        commands = [line for line in out.getvalue().splitlines() if line.startswith('bash ')]
        self.assertEqual(len(commands), 4)
        self.assertIn('--baseline_rl_reward_mode official', commands[0])
        for strength, cmd in zip(('0.3', '0.5', '1.0'), commands[1:]):
            self.assertIn('--baseline_rl_prefix_reward_strength ' + strength, cmd)
            self.assertIn('--baseline_rl_stop_after_steps 1000', cmd)
            self.assertIn('--require_existing_sft', cmd)

    def test_failure_stops_then_resume_skips_complete_and_keeps_old_attempt(self):
        calls = []

        def initial(cmd, env, log):
            calls.append(cmd)
            if len(calls) == 2:
                self.assertTrue((self.directory / 'summary.csv').is_file())
                raise RuntimeError('simulated training failure')
            self.emit_result(cmd, env, log)

        with patch.object(sweep, 'execute', side_effect=initial):
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                self.call()
        state = sweep.read_json(self.directory / 'state.json')
        self.assertEqual([x['status'] for x in state['arms']], ['complete', 'failed', 'pending'])
        official_path = sweep.run_dir(state['arms'][0]['run_tag']) / 'valid_metrics.json'
        official_bytes = official_path.read_bytes()
        with patch.object(sweep, 'execute', side_effect=self.emit_result) as execute:
            self.call(['--resume'])
        self.assertEqual(execute.call_count, 2)
        state = sweep.read_json(self.directory / 'state.json')
        self.assertTrue(all(x['status'] == 'complete' for x in state['arms']))
        self.assertTrue(state['arms'][1]['run_tag'].endswith('_a2'))
        self.assertEqual(len(state['arms'][1]['attempts']), 2)
        self.assertEqual(official_bytes, official_path.read_bytes())
        summary = sweep.read_json(self.directory / 'summary.json')['results']
        self.assertEqual(len(summary), 3)
        self.assertAlmostEqual(summary[-1]['task_advantage_coverage'], .75)
        with patch.object(sweep, 'execute') as execute:
            self.call(['--resume'])
            self.call(['--summarize_only'])
        execute.assert_not_called()
        with self.assertRaises(FileExistsError):
            self.call()  # Reusing the tag without --resume never overwrites it.

    def test_reject_changed_completed_budget_before_launching_more(self):
        with patch.object(sweep, 'execute', side_effect=self.emit_result):
            self.call()
        state = sweep.read_json(self.directory / 'state.json')
        path = sweep.run_dir(state['arms'][0]['run_tag']) / 'valid_metrics.json'
        data = sweep.read_json(path)
        data['training_config']['completed_optimizer_steps'] = 999
        sweep.write_json(path, data)
        with patch.object(sweep, 'execute') as execute, self.assertRaisesRegex(ValueError, 'fixed pilot protocol'):
            self.call(['--resume'])
        execute.assert_not_called()

    def test_missing_parent_never_launches_runner(self):
        (self.root / sweep.SFT / 'model.safetensors').unlink()
        with patch.object(sweep, 'execute') as execute, self.assertRaisesRegex(ValueError, 'Existing SFT'):
            self.call()
        execute.assert_not_called()

    def test_required_parent_flag_blocks_shell_sft_fallback(self):
        # Minimal isolated checkout makes the real shell reach missing-parent logic.
        scripts = self.root / 'scripts'
        scripts.mkdir()
        real_root = Path(__file__).resolve().parents[1]
        (scripts / 'run_experiment.sh').write_bytes((real_root / 'scripts/run_experiment.sh').read_bytes())
        result = subprocess.run(['bash', str(scripts / 'run_experiment.sh'), '--method', 'minionerec_rl',
                                 '--dataset', 'Office', '--sft_run_tag', 'absent',
                                 '--require_existing_sft', '--dry_run'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SFT will not be trained', result.stderr)
        self.assertNotIn('scripts/train_diprec_sft.sh', result.stdout)

    def test_real_subprocess_error_is_logged_and_propagated(self):
        log = self.root / 'failure.log'
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'exited 7'):
            sweep.execute([sys.executable, '-c', "print('failed rollout'); raise SystemExit(7)"], {}, log)
        self.assertIn('failed rollout', log.read_text())


if __name__ == '__main__':
    unittest.main()
