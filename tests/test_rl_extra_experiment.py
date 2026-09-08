import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

from scripts import run_rl_extra_experiment as extra
from scripts import run_rl_optimization_sweep as sweep
from tests import test_rl_optimization_sweep as fixtures


class ExtraExperimentTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OptimizationSweepTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.source = self.root / 'outputs' / sweep.BASE / 'rl_optimization_sweeps/rl_opt_v1'
        self.directory = self.source.with_name('rl_opt_extra_v1')
        self.source.mkdir(parents=True)
        frozen = dict(samples_per_split=2, sft=str(sweep.SFT), source_sha256={},
                      sft_training_config=sweep.read_json(self.root/sweep.SFT/'training_config.json'))
        sweep.write_json(self.source/'probe.json', frozen)
        state = dict(samples=2, input_sha256=sweep.input_identity(), probe_sha256=sweep.digest(self.source/'probe.json'),
                     arms=[a | dict(status='pending', attempts=[]) for a in sweep.ARMS])
        with patch.object(sweep, 'execute', side_effect=self.fake_execute):
            sweep.ensure_evaluation(self.source/'baseline', self.source, sweep.SFT, {}, 2)
            arm = state['arms'][0]
            arm.update(status='complete', run_tag='rl_rl_opt_v1_A_a1')
            self.fake_execute(sweep.training_command(arm), {'DIPREC_NUM_PROCESSES':'4'}, None)
            for step in sweep.STEPS:
                sweep.ensure_evaluation(sweep.result_dir(arm, step), self.source, sweep.checkpoint(arm, step), {}, 2, arm, step)
        sweep.write_json(self.source/'state.json', state)

    def fake_execute(self, cmd, env, log):
        with patch.object(sweep, 'ARMS', (*sweep.ARMS, extra.ARM)):
            self.fixture.fake_execute(cmd, env, log)
        if cmd[0] == 'bash' and cmd[cmd.index('--baseline_rl_beta')+1] == '0.0':
            tag = cmd[cmd.index('--run_tag')+1]
            path = sweep.run_dir(tag)/'rl_diagnostics.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                del row['kl']
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def call(self, args=()):
        with contextlib.redirect_stdout(io.StringIO()):
            extra.main(list(args))

    def test_extra_run_isolated_from_source_with_no_kl_and_matched_comparison(self):
        before = {str(p.relative_to(self.root)):p.read_bytes() for p in (self.root/'outputs').rglob('*') if p.is_file()}
        with patch.object(sweep, 'execute', side_effect=self.fake_execute) as execute:
            self.call()
        training = [c for c in execute.call_args_list if c.args[0][0]=='bash']
        self.assertEqual(len(training),1)
        cmd, env, _ = training[0].args
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'4,5,6,7')
        self.assertEqual(env['DIPREC_MAIN_PROCESS_PORT'],'29517')
        self.assertIn('--skip_preprocess',cmd)
        self.assertEqual(cmd[cmd.index('--baseline_rl_beta')+1],'0.0')
        for c in execute.call_args_list:
            self.assertEqual(c.args[1]['CUDA_VISIBLE_DEVICES'],'4,5,6,7')
        for name, content in before.items():
            self.assertEqual((self.root/name).read_bytes(),content,name)
        self.assertEqual((self.directory/'probe.json').read_bytes(),(self.source/'probe.json').read_bytes())
        summary = sweep.read_json(self.directory/'summary.json')
        self.assertEqual(len(summary['results']),3)
        self.assertTrue(all(r['kl_mean'] is None and r['kl_max'] is None and not r['training_kl_available'] for r in summary['results']))
        comparisons = sweep.read_json(self.directory/'comparisons_with_source.json')['comparisons']
        self.assertEqual([(r['comparison'],r['step']) for r in comparisons],[('F-A',s) for s in sweep.STEPS])
        state = sweep.read_json(self.directory/'state.json')
        self.assertEqual(state['arms'][0]['status'],'complete')
        with patch.object(sweep,'execute') as execute:
            self.call(['--resume'])
        execute.assert_not_called()
        # C may finish on the other GPUs after F: summary adds it without rerunning F.
        source_state=sweep.read_json(self.source/'state.json')
        control=source_state['arms'][2]
        control.update(status='complete',run_tag='rl_rl_opt_v1_C_a2')
        with patch.object(sweep,'execute',side_effect=self.fake_execute):
            self.fake_execute(sweep.training_command(control),{'DIPREC_NUM_PROCESSES':'4'},None)
            for step in sweep.STEPS:
                sweep.ensure_evaluation(sweep.result_dir(control,step),self.source,sweep.checkpoint(control,step),{},2,control,step)
        sweep.write_json(self.source/'state.json',source_state)
        with patch.object(sweep,'execute') as execute:
            self.call(['--summarize_only'])
        execute.assert_not_called()
        comparisons=sweep.read_json(self.directory/'comparisons_with_source.json')['comparisons']
        self.assertEqual(len(comparisons),6)
        self.assertEqual(sum(r['comparison']=='F-C' for r in comparisons),3)
        shutil.rmtree(self.root/'output_dir')
        shutil.rmtree(self.root/'data')
        with patch.object(sweep,'execute') as execute:
            self.call(['--summarize_only'])
        execute.assert_not_called()

    def test_evaluation_failure_reuses_training_and_training_failure_gets_new_attempt(self):
        def fail_train(cmd, env, log):
            if cmd[0]=='bash':
                raise RuntimeError('training interrupted')
            self.fake_execute(cmd,env,log)
        with patch.object(sweep,'execute',side_effect=fail_train), self.assertRaisesRegex(RuntimeError,'training interrupted'):
            self.call()
        def fail_eval(cmd, env, log):
            if cmd[1].endswith('evaluate_diprec.py'):
                raise RuntimeError('evaluation interrupted')
            self.fake_execute(cmd,env,log)
        with patch.object(sweep,'execute',side_effect=fail_eval), self.assertRaisesRegex(RuntimeError,'evaluation interrupted'):
            self.call(['--resume'])
        with patch.object(sweep,'execute',side_effect=self.fake_execute) as execute:
            self.call(['--resume'])
        self.assertTrue(all(c.args[0][0]!='bash' for c in execute.call_args_list))
        arm=sweep.read_json(self.directory/'state.json')['arms'][0]
        self.assertEqual(arm['status'],'complete')
        self.assertTrue(arm['run_tag'].endswith('_a2'))
        self.assertEqual(len(arm['attempts']),2)

    def test_changed_inputs_or_probe_rejected_without_launch(self):
        with patch.object(sweep,'execute',side_effect=self.fake_execute):
            self.call()
        weights=self.root/sweep.SFT/'model.safetensors'
        original=weights.read_bytes()
        weights.write_text('changed')
        with patch.object(sweep,'execute') as execute, self.assertRaisesRegex(ValueError,'input data changed'):
            self.call(['--resume'])
        execute.assert_not_called()
        weights.write_bytes(original)
        (self.directory/'probe.json').write_text('{}')
        with patch.object(sweep,'execute') as execute, self.assertRaisesRegex(ValueError,'Frozen probe changed'):
            self.call(['--resume'])
        execute.assert_not_called()

    def test_dry_run_and_invalid_isolation_have_no_writes(self):
        before=set(self.root.rglob('*'))
        out=io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(sweep,'execute') as execute:
            extra.main(['--dry_run'])
        execute.assert_not_called()
        self.assertEqual(set(self.root.rglob('*')),before)
        self.assertIn('CUDA_VISIBLE_DEVICES=4,5,6,7',out.getvalue())
        self.assertEqual(sum(line.startswith('bash ') for line in out.getvalue().splitlines()),1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.call(['--experiment_tag','rl_opt_v1'])
        self.assertEqual(set(self.root.rglob('*')),before)

    def test_shell_forwards_explicit_port_and_preserves_default(self):
        executable=self.root/'accelerate'
        executable.write_text('#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps(dict(args=sys.argv[1:],gpus=os.environ["CUDA_VISIBLE_DEVICES"])))\n')
        executable.chmod(0o755)
        env=os.environ | dict(PATH=str(self.root)+os.pathsep+os.environ['PATH'],
                              DIPREC_DDP='1',DIPREC_NUM_PROCESSES='4',CUDA_VISIBLE_DEVICES='4,5,6,7')
        repo=Path(__file__).resolve().parents[1]
        for port in (None,'29517'):
            selected=env.copy()
            selected.pop('DIPREC_MAIN_PROCESS_PORT',None)
            if port is not None:
                selected['DIPREC_MAIN_PROCESS_PORT']=port
            result=subprocess.run(['bash','scripts/train_baseline_grpo.sh','--beta','0'],cwd=repo,env=selected,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            output=json.loads(result.stdout)
            self.assertEqual(output['gpus'],'4,5,6,7')
            if port is None:
                self.assertNotIn('--main_process_port',output['args'])
            else:
                self.assertEqual(output['args'][output['args'].index('--main_process_port')+1],port)
            self.assertEqual(output['args'][-3:],['scripts/train_baseline_grpo.py','--beta','0'])


if __name__ == '__main__':
    unittest.main()
