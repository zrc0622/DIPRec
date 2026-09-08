#!/usr/bin/env python3
"""Independent beta=0 arm on GPUs 4–7, reusing the frozen A–E sweep baseline."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_rl_optimization_sweep as sweep

ARM = dict(label='F', reference_mode='fixed', beta=0.0, learning_rate=2e-6)
BASELINE_FILES = ('valid_metrics.json', 'valid_predictions.jsonl', 'probe_metrics.json', 'evaluation_complete.json')


def training_command(arm):
    # Source identity is checked first; parallel jobs never build shared input data.
    return sweep.training_command(arm) + ['--skip_preprocess']


def validate_baseline(directory, state):
    if sweep.digest(directory / 'probe.json') != state['probe_sha256']:
        raise ValueError('Frozen probe changed')
    frozen = sweep.read_json(directory / 'probe.json')
    if frozen['samples_per_split'] != state['samples'] or frozen['sft'] != str(sweep.SFT):
        raise ValueError('Unexpected frozen probe sample count or SFT parent')
    if any(state['input_sha256'].get(k) != v for k,v in frozen['source_sha256'].items()):
        raise ValueError('Probe input identity differs from saved sweep')
    dest = directory / 'baseline'
    sweep.validate_marker(dest)
    ranking = sweep.validate_evaluation(dest, sweep.SFT, check_checkpoint=False)
    probe = sweep.validate_probe(dest / 'probe_metrics.json', directory / 'probe.json',
                                 sweep.SFT, state['samples'], check_checkpoint=False)
    if ranking['training_config'] != probe['training_config'] or ranking['training_config'] != frozen['sft_training_config']:
        raise ValueError('Frozen baseline checkpoint metadata differs')


def summarize(directory, state):
    sweep.summarize(directory, state)
    # Read completed A/C evaluations only. Never acquire or update the active source state.
    source = sweep.ROOT / 'outputs' / sweep.BASE / 'rl_optimization_sweeps' / state['source_sweep']
    source_state = sweep.read_json(source / 'state.json')
    if source_state['input_sha256'] != state['input_sha256'] or source_state['probe_sha256'] != state['probe_sha256']:
        raise ValueError('Source sweep identity differs from the extra experiment')
    comparisons = []
    arm = state['arms'][0]
    if arm.get('run_tag'):
        for step in sweep.STEPS:
            extra_dir = sweep.result_dir(arm, step)
            if not (extra_dir / 'evaluation_complete.json').exists():
                continue
            extra = sweep.read_json(extra_dir / 'valid_metrics.json')
            for control in source_state['arms']:
                if control['label'] not in ('A', 'C') or not control.get('run_tag'):
                    continue
                dest = sweep.result_dir(control, step)
                if not (dest / 'evaluation_complete.json').exists():
                    continue
                sweep.validate_marker(dest)
                reference = sweep.validate_evaluation(dest, sweep.checkpoint(control, step), control, step, check_checkpoint=False)
                probe = sweep.validate_probe(dest / 'probe_metrics.json', source / 'probe.json',
                                             sweep.checkpoint(control, step), state['samples'], check_checkpoint=False)
                if probe['probe_sha256'] != state['probe_sha256'] or not sweep.compatible_eval(reference, extra):
                    raise ValueError('Source and extra evaluations are not comparable')
                fixed = lambda d: {k:v for k,v in d['training_config'].items() if k not in sweep.VARIABLE_CONFIG}
                if fixed(reference) != fixed(extra):
                    raise ValueError('Source and extra training configurations differ outside declared variables')
                comparisons.append(dict(comparison=f"F-{control['label']}", step=step,
                                        **{key:extra['metrics'][key]-reference['metrics'][key] for key in sweep.METRICS}))
    sweep.write_json(directory / 'comparisons_with_source.json', dict(comparisons=comparisons,
                     note='Matched-step Valid point estimates, not significance tests. Missing source evaluations are omitted.'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_sweep', default='rl_opt_v1')
    parser.add_argument('--experiment_tag', default='rl_opt_extra_v1')
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--main_process_port', type=int, default=29517)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', tag) for tag in (args.source_sweep,args.experiment_tag)):
        parser.error('Sweep and experiment tags must be safe directory names')
    if args.source_sweep == args.experiment_tag:
        parser.error('The extra experiment must use a separate directory from A–E')
    gpu_ids = [x.strip() for x in args.gpus.split(',')]
    if not args.summarize_only and (len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or not all(gpu_ids)):
        parser.error('Exactly four distinct GPUs are required')
    if not 1024 <= args.main_process_port <= 65535:
        parser.error('Use a main process port between 1024 and 65535')
    env = os.environ | dict(CUDA_VISIBLE_DEVICES=','.join(gpu_ids), DIPREC_DDP='1',
                           DIPREC_NUM_PROCESSES='4', DIPREC_MAIN_PROCESS_PORT=str(args.main_process_port),
                           PYTHONUNBUFFERED='1')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    eval_env = env | dict(DIPREC_DDP='0')
    base = sweep.ROOT / 'outputs' / sweep.BASE / 'rl_optimization_sweeps'
    source, directory = base / args.source_sweep, base / args.experiment_tag
    arm = ARM | dict(status='pending', attempts=[], base_tag=f'rl_{args.experiment_tag}_F')
    if args.dry_run:
        arm['run_tag'] = arm['base_tag'] + '_a1'
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} DIPREC_DDP=1 DIPREC_NUM_PROCESSES=4 DIPREC_MAIN_PROCESS_PORT={args.main_process_port}")
        print(f'Reuse frozen SFT baseline/probe from {source}; write only {directory}')
        print(shlex.join(training_command(arm)))
        for step in sweep.STEPS:
            dest = sweep.result_dir(arm, step)
            print(shlex.join(sweep.evaluation_command(sweep.checkpoint(arm, step), dest)))
            print(shlex.join(sweep.probe_command(directory, 128, sweep.checkpoint(arm, step), dest)))
        return
    state_path = directory / 'state.json'
    if args.resume or args.summarize_only:
        if not state_path.is_file():
            parser.error(f'No saved extra experiment: {directory}')
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume or args.summarize_only:
            state = sweep.read_json(state_path)
            if state['source_sweep'] != args.source_sweep or len(state['arms']) != 1 or any(state['arms'][0][k] != v for k,v in ARM.items()):
                raise ValueError('Saved extra experiment definition differs')
        else:
            original = sweep.read_json(source / 'state.json')
            validate_baseline(source, original)
            state = dict(schema='diprec.rl_extra.v1', source_sweep=args.source_sweep,
                         samples=original['samples'], input_sha256=original['input_sha256'],
                         probe_sha256=original['probe_sha256'], arms=[arm])
            if sweep.input_identity() != state['input_sha256']:
                raise ValueError('SFT checkpoint or input data changed since the source sweep')
            (directory / 'baseline').mkdir()
            shutil.copyfile(source / 'probe.json', directory / 'probe.json')
            for name in BASELINE_FILES:
                shutil.copyfile(source / 'baseline' / name, directory / 'baseline' / name)
            sweep.write_json(state_path, state)
        validate_baseline(directory, state)
        summarize(directory, state)
        if args.summarize_only:
            return
        if sweep.input_identity() != state['input_sha256']:
            raise ValueError('SFT checkpoint or input data changed since the source sweep')
        arm = state['arms'][0]
        try:
            if not sweep.training_complete(arm):
                if arm['status'] == 'complete':
                    raise ValueError('Completed extra arm lost its checkpoint')
                tag = f"{arm['base_tag']}_a{len(arm['attempts'])+1}"
                if any(sweep.run_dir(tag, kind).exists() for kind in ('outputs','output_dir')):
                    raise ValueError(f'Attempt output already exists: {tag}')
                arm.update(run_tag=tag, status='training')
                arm['attempts'].append(dict(run_tag=tag, gpus=args.gpus, main_process_port=args.main_process_port))
                sweep.write_json(state_path, state)
                sweep.execute(training_command(arm), env, directory / f'{tag}.log')
                if not sweep.training_complete(arm):
                    raise ValueError('Runner returned without complete training checkpoints')
            arm['status'] = 'evaluating'
            sweep.write_json(state_path, state)
            for step in sweep.STEPS:
                sweep.ensure_evaluation(sweep.result_dir(arm, step), directory, sweep.checkpoint(arm, step),
                                        eval_env, state['samples'], arm, step)
                summarize(directory, state)
            arm['status'] = 'complete'
            arm.pop('error', None)
            sweep.write_json(state_path, state)
        except BaseException as exc:
            arm.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            sweep.write_json(state_path, state)
            raise


if __name__ == '__main__':
    main()
