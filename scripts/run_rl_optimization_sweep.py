#!/usr/bin/env python3
"""Five serial MiniOneRec RL runs, with frozen probes and offline milestone evaluation."""

import argparse
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diprec.rl_probe import digest
from scripts.run_prefix_sweep import BASE, SFT, SFT_TAG, METRICS, compatible_eval, execute, read_json, write_json

STEPS = (1000, 2000, 3455)
ARMS = (
    dict(label='A', reference_mode='fixed', beta=.01, learning_rate=2e-6),
    dict(label='B', reference_mode='sync', beta=.01, learning_rate=2e-6),
    dict(label='C', reference_mode='fixed', beta=.001, learning_rate=2e-6),
    dict(label='D', reference_mode='sync', beta=.001, learning_rate=2e-6),
    dict(label='E', reference_mode='fixed', beta=.01, learning_rate=5e-6),
)
DATA = Path('data/processed/Office_Products/history_50')
INDEX = Path('data/Amazon/index/Office_Products.index.json')
ITEMS = Path('data/Amazon/index/Office_Products.item.json')
TASKS = ('history_sid_to_sid', 'title_to_sid', 'description_to_sid', 'title_history_to_sid')
VARIABLE_CONFIG = {'reference_mode', 'reference_policy', 'beta', 'learning_rate', 'output_dir',
                   'training_metrics_file', 'diagnostics_file'}


def relative(path):
    return str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else str(path)


def run_dir(tag, kind='outputs'):
    return ROOT / kind / BASE / 'minionerec_rl' / f'seed_42_{tag}'


def checkpoint(arm, step):
    return run_dir(arm['run_tag'], 'output_dir') / ('final_checkpoint' if step == STEPS[-1] else f'step_{step}')


def result_dir(arm, step):
    directory = run_dir(arm['run_tag'])
    return directory if step == STEPS[-1] else directory / f'step_{step}'


def training_command(arm):
    return ['bash', 'scripts/run_experiment.sh',
            '--method', 'minionerec_rl', '--dataset', 'Office_Products',
            '--model', 'Qwen/Qwen3-0.6B', '--max_history_len', '50', '--seed', '42',
            '--sft_run_tag', SFT_TAG, '--require_existing_sft', '--run_tag', arm['run_tag'],
            '--baseline_rl_task_scope', 'official_mixed',
            '--baseline_rl_reference_mode', arm['reference_mode'],
            '--baseline_rl_ref_model_sync_steps', '512', '--baseline_rl_ref_model_mixup_alpha', '0.6',
            '--baseline_rl_per_device_batch_size', '16', '--baseline_rl_gradient_accumulation_steps', '4',
            '--baseline_rl_generation_batch_size', '256',
            '--baseline_rl_learning_rate', str(arm['learning_rate']), '--baseline_rl_beta', str(arm['beta']),
            '--baseline_rl_num_epochs', '1', '--baseline_rl_stop_after_steps', '0',
            '--baseline_rl_reward_mode', 'official', '--baseline_rl_snapshot_steps', '1000,2000',
            '--baseline_rl_expected_optimizer_steps', '3455',
            '--baseline_rl_diagnostics', '--eval_beams', '10', '--eval_candidate_budget', '80',
            '--eval_split', 'valid']


def evaluation_command(model, directory, method='minionerec_rl'):
    return [sys.executable, 'scripts/evaluate_diprec.py', '--method', method,
            '--model', relative(model), '--base_model', 'Qwen/Qwen3-0.6B',
            '--test_file', str(DATA / 'valid.jsonl'), '--sid_index', str(INDEX), '--item_meta', str(ITEMS),
            '--output', relative(directory / 'valid_metrics.json'), '--split', 'valid',
            '--eval_beams', '10', '--eval_candidate_budget', '80',
            '--max_history_len', '50', '--max_seq_len', '2048', '--seed', '42']


def probe_command(directory, samples, model=None, destination=None):
    command = [sys.executable, 'scripts/probe_baseline_rl.py']
    if model is None:
        return command + ['prepare', '--sft', str(SFT), '--samples', str(samples),
                          '--output', relative(directory / 'probe.json')]
    return command + ['evaluate', '--model', relative(model), '--probe', relative(directory / 'probe.json'),
                      '--output', relative(destination / 'probe_metrics.json')]


def require_weights(path):
    if not all((path / name).is_file() for name in ('config.json', 'training_config.json')) or not (
            list(path.glob('model*.safetensors')) or list(path.glob('pytorch_model*.bin'))):
        raise ValueError(f'Existing checkpoint weights/config required: {path}')
    for index in path.glob('*.index.json'):
        if any(not (path / shard).is_file() for shard in read_json(index).get('weight_map', {}).values()):
            raise ValueError(f'Missing checkpoint shard: {index}')


def input_identity():
    parent = ROOT / SFT
    require_weights(parent)
    config = read_json(parent / 'training_config.json')
    if config.get('method') != 'minionerec_sft':
        raise ValueError('The existing parent must be MiniOneRec SFT')
    files = [ROOT / DATA / name for name in ('train.jsonl', 'valid.jsonl', 'manifest.json')]
    files += [ROOT / INDEX, ROOT / ITEMS]
    # Hash only top-level model/tokenizer files, never nested optimizer checkpoints.
    files += sorted(p for p in parent.iterdir() if p.is_file())
    return {relative(p): digest(p) for p in files}


def validate_training(config, arm, step):
    expected = dict(method='minionerec_rl', model=str(SFT), seed=42, reward_mode='official',
                    completed_optimizer_steps=step, stop_after_steps=0, scheduler_total_steps=3455,
                    snapshot_steps=[1000, 2000], num_epochs=1, task_scope='official_mixed',
                    reference_mode=arm['reference_mode'], learning_rate=arm['learning_rate'], beta=arm['beta'],
                    ref_model_sync_steps=512, ref_model_mixup_alpha=.6,
                    num_generations=16, num_iterations=1, temperature=1.0)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Arm {arm['label']} step {step}: {key}={config.get(key)!r}, expected {value!r}")
    batch = dict(world_size=4, effective_update_batch=256, generation_batch_size=256,
                 per_device_batch_size=16, gradient_accumulation_steps=4)
    if any(config.get('batch', {}).get(k) != v for k, v in batch.items()):
        raise ValueError('Unexpected four-GPU batch contract')
    if step != STEPS[-1] and not (config.get('snapshot_model_only') and config.get('checkpoint_role') == 'rl_milestone'):
        raise ValueError('Expected a model-only milestone checkpoint')


def validate_checkpoint(arm, step):
    path = checkpoint(arm, step)
    require_weights(path)
    config = read_json(path / 'training_config.json')
    validate_training(config, arm, step)
    return config


def training_complete(arm):
    if not arm.get('run_tag') or not (checkpoint(arm, STEPS[-1]) / 'training_config.json').exists():
        return False
    for step in STEPS:
        validate_checkpoint(arm, step)
    return True


def validate_evaluation(directory, model, arm=None, step=None, check_checkpoint=True):
    data = read_json(directory / 'valid_metrics.json')
    if data['split'] != 'valid' or data['num_examples'] <= 0 or data['checkpoint'] != relative(model):
        raise ValueError(f'Unexpected validation checkpoint/split: {directory}')
    if any(not math.isfinite(data['metrics'][key]) for key in METRICS):
        raise ValueError(f'Nonfinite validation metrics: {directory}')
    cfg = data['evaluation_config']
    if any(cfg.get(k) != v for k, v in dict(eval_beams=10, eval_candidate_budget=80, max_history_len=50, max_seq_len=2048).items()):
        raise ValueError('Unexpected evaluation budget')
    if check_checkpoint and data['training_config'] != read_json(ROOT / model / 'training_config.json'):
        raise ValueError('Evaluation training metadata differs from checkpoint')
    if arm:
        validate_training(data['training_config'], arm, step)
    with (directory / 'valid_predictions.jsonl').open() as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    ids = [r['sample_id'] for r in predictions]
    if len(predictions) != data['num_examples'] or len(set(ids)) != len(ids):
        raise ValueError('Incomplete or duplicate validation predictions')
    return data


def validate_probe(path, frozen, model, samples, check_checkpoint=True):
    data = read_json(path)
    if data['probe_sha256'] != digest(frozen) or data['checkpoint'] != relative(model):
        raise ValueError('Probe checkpoint or frozen candidates differ')
    if check_checkpoint and data['training_config'] != read_json(ROOT / model / 'training_config.json'):
        raise ValueError('Probe training metadata differs from checkpoint')
    for split in ('train', 'valid'):
        if data['summary'][split]['samples'] != samples:
            raise ValueError('Probe sample count differs')
        if any(not math.isfinite(v) for v in data['summary'][split].values()):
            raise ValueError('Nonfinite probe metrics')
    return data


def validate_marker(directory):
    paths = [directory / n for n in ('valid_metrics.json', 'valid_predictions.jsonl', 'probe_metrics.json')]
    if read_json(directory / 'evaluation_complete.json') != {p.name:digest(p) for p in paths}:
        raise ValueError(f'Completed evaluation files changed: {directory}')


def ensure_evaluation(directory, sweep_dir, model, env, samples, arm=None, step=None):
    directory.mkdir(parents=True, exist_ok=True)
    # A marker is written only after both evaluator outputs and the probe validate.
    marker = directory / 'evaluation_complete.json'
    paths = [directory / n for n in ('valid_metrics.json', 'valid_predictions.jsonl', 'probe_metrics.json')]
    if marker.exists():
        validate_marker(directory)
    else:
        try:
            validate_evaluation(directory, model, arm, step)
        except (OSError, ValueError, KeyError):
            execute(evaluation_command(model, directory, 'minionerec_rl' if arm else 'minionerec_sft'),
                    env, directory / 'evaluation.log')
        if not (directory / 'probe_metrics.json').exists():
            execute(probe_command(sweep_dir, samples, model, directory), env, directory / 'probe.log')
    result = validate_evaluation(directory, model, arm, step)
    probe = validate_probe(directory / 'probe_metrics.json', sweep_dir / 'probe.json', model, samples)
    if not marker.exists():
        write_json(marker, {p.name: digest(p) for p in paths})
    return result, probe


def diagnostics(path, beta=None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if 'grad_norm' in r]
    if [r['step'] for r in rows] != list(range(1, STEPS[-1] + 1)):
        raise ValueError(f'Expected {STEPS[-1]} ordered training records: {path}')
    result = {}
    start = 0
    for end in STEPS:
        window = rows[start:end]
        # TRL omits reference scoring/KL logging when beta=0. Missing KL is not zero drift.
        summary = dict(window_start=start+1, window_end=end,
                       training_kl_available=beta != 0,
                       kl_mean=sum(r['kl'] for r in window)/len(window) if beta != 0 else None,
                       kl_max=max(r['kl'] for r in window) if beta != 0 else None)
        for task in TASKS:
            prefix = f'prefix_aux/{task}/'
            count = sum(r.get(prefix+'groups', 0) for r in window)
            summary[task+'_groups'] = count
            summary[task+'_exact_hit_group_fraction'] = (
                sum(r.get(prefix+'groups', 0)*r.get(prefix+'exact_hit', 0) for r in window)/count if count else None)
        result[end] = summary
        start = end
    return result


def summarize(directory, state):
    results = []
    baseline = directory / 'baseline' / 'valid_metrics.json'
    if not baseline.exists():
        return
    reference = read_json(baseline)
    validate_marker(directory / 'baseline')
    controls = {}
    for arm in state['arms']:
        if not arm.get('run_tag'):
            continue
        if not any((result_dir(arm, step) / 'evaluation_complete.json').exists() for step in STEPS):
            continue
        diag = diagnostics(run_dir(arm['run_tag']) / 'rl_diagnostics.jsonl', beta=arm['beta'])
        for step in STEPS:
            dest = result_dir(arm, step)
            if not (dest / 'evaluation_complete.json').exists():
                continue
            validate_marker(dest)
            data = validate_evaluation(dest, checkpoint(arm, step), arm, step, check_checkpoint=False)
            if not compatible_eval(reference, data):
                raise ValueError('Evaluation data/budget differs from the SFT baseline')
            fixed = {k:v for k,v in data['training_config'].items() if k not in VARIABLE_CONFIG}
            if controls.setdefault(step, fixed) != fixed:
                raise ValueError('Arms differ outside reference mode, beta, and learning rate')
            probe = validate_probe(dest / 'probe_metrics.json', directory / 'probe.json', checkpoint(arm, step), state['samples'], check_checkpoint=False)
            if probe['training_config'] != data['training_config']:
                raise ValueError('Probe and ranking metrics describe different checkpoints')
            row = dict(arm=arm['label'], step=step, reference_mode=arm['reference_mode'], beta=arm['beta'],
                       learning_rate=arm['learning_rate'], metrics_file=relative(dest / 'valid_metrics.json'),
                       **{key:data['metrics'][key] for key in METRICS}, **diag[step])
            row.update({f'delta_{key}_vs_SFT': data['metrics'][key]-reference['metrics'][key] for key in METRICS})
            for split, metrics in probe['summary'].items():
                row.update({f'probe_{split}_{key}':value for key,value in metrics.items()})
            results.append(row)
    comparisons = []
    for step in STEPS:
        by_label = {row['arm']:row for row in results if row['step'] == step}
        for control, variant in (('A','B'), ('A','C'), ('C','D'), ('B','D'), ('A','E')):
            if control in by_label and variant in by_label:
                comparisons.append(dict(step=step, comparison=f'{variant}-{control}', **{
                    key:by_label[variant][key]-by_label[control][key] for key in METRICS}))
    write_json(directory / 'summary.json', dict(results=results, comparisons=comparisons,
               sft_metrics=reference['metrics'], primary_step=3455,
               note='Validation point estimates, not significance tests. Training KL uses the arm reference; probe KL uses six frozen candidates, not the full policy.'))
    columns = list(dict.fromkeys(k for row in results for k in row))
    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    for row in results:
        print(f"{row['arm']} step={row['step']} R@10={row['Recall@10']:.6f} NDCG@10={row['NDCG@10']:.6f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep_tag', default=datetime.now().strftime('rl_optimization_%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'))
    parser.add_argument('--probe_samples', type=int, default=128, help='Frozen main-task records per train/valid split')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.sweep_tag):
        parser.error('Invalid sweep tag')
    gpu_ids = [x.strip() for x in args.gpus.split(',')]
    if not args.summarize_only and (len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or not all(gpu_ids)):
        parser.error('Exactly four distinct GPUs are required')
    if args.probe_samples < 1:
        parser.error('--probe_samples must be positive')
    env = os.environ | dict(CUDA_VISIBLE_DEVICES=','.join(gpu_ids), DIPREC_DDP='1', DIPREC_NUM_PROCESSES='4', PYTHONUNBUFFERED='1')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    # Evaluation/probing use a single process on the first visible GPU.
    eval_env = env | dict(DIPREC_DDP='0')
    directory = ROOT / 'outputs' / BASE / 'rl_optimization_sweeps' / args.sweep_tag
    arms = [arm | dict(status='pending', attempts=[], base_tag=f"rl_{args.sweep_tag}_{arm['label']}") for arm in ARMS]
    if args.dry_run:
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} DIPREC_DDP=1 DIPREC_NUM_PROCESSES=4")
        print(shlex.join(probe_command(directory, args.probe_samples)))
        print(shlex.join(evaluation_command(SFT, directory / 'baseline', 'minionerec_sft')))
        print(shlex.join(probe_command(directory, args.probe_samples, SFT, directory / 'baseline')))
        for arm in arms:
            arm['run_tag'] = arm['base_tag'] + '_a1'
            print(shlex.join(training_command(arm)))
            for step in STEPS:
                # The shell already evaluates the final checkpoint; this command is also used on evaluation resume.
                print(shlex.join(evaluation_command(checkpoint(arm, step), result_dir(arm, step))))
                print(shlex.join(probe_command(directory, args.probe_samples, checkpoint(arm, step), result_dir(arm, step))))
        return
    state_path = directory / 'state.json'
    if args.resume or args.summarize_only:
        if not state_path.is_file():
            parser.error(f'No saved sweep: {directory}')
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume or args.summarize_only:
            state = read_json(state_path)
            if state['samples'] != args.probe_samples:
                parser.error('Use the saved --probe_samples value when resuming/summarizing')
            if [{k:arm[k] for k in template} for arm,template in zip(state['arms'], ARMS)] != list(ARMS) or len(state['arms']) != 5:
                raise ValueError('Saved experiment definitions differ')
        else:
            state = dict(schema='diprec.rl_optimization.v1', samples=args.probe_samples, arms=arms)
            write_json(state_path, state)
        if args.summarize_only:
            summarize(directory, state)
            return
        identity = input_identity()
        if state.get('input_sha256', identity) != identity:
            raise ValueError('SFT checkpoint or input data changed since the sweep started')
        state['input_sha256'] = identity
        write_json(state_path, state)
        frozen = directory / 'probe.json'
        if not frozen.exists():
            execute(probe_command(directory, args.probe_samples), eval_env, directory / 'prepare_probe.log')
        if state.get('probe_sha256', digest(frozen)) != digest(frozen):
            raise ValueError('Frozen probe changed')
        state['probe_sha256'] = digest(frozen)
        write_json(state_path, state)
        ensure_evaluation(directory / 'baseline', directory, SFT, eval_env, args.probe_samples)
        summarize(directory, state)  # Verify completed runs before launching more work.
        for arm in state['arms']:
            try:
                if not training_complete(arm):
                    if arm['status'] == 'complete':
                        raise ValueError('Completed arm lost its checkpoint')
                    tag = f"{arm['base_tag']}_a{len(arm['attempts'])+1}"
                    if any(run_dir(tag, kind).exists() for kind in ('outputs', 'output_dir')):
                        raise ValueError(f'Attempt output already exists: {tag}')
                    arm.update(run_tag=tag, status='training')
                    arm['attempts'].append(dict(run_tag=tag))
                    write_json(state_path, state)
                    execute(training_command(arm), env, directory / f'{tag}.log')
                    if not training_complete(arm):
                        raise ValueError('Runner returned without complete training checkpoints')
                arm['status'] = 'evaluating'
                write_json(state_path, state)
                for step in STEPS:
                    ensure_evaluation(result_dir(arm, step), directory, checkpoint(arm, step), eval_env,
                                      args.probe_samples, arm, step)
                    summarize(directory, state)
                arm['status'] = 'complete'
                arm.pop('error', None)
                write_json(state_path, state)
            except BaseException as exc:
                arm.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                write_json(state_path, state)
                raise
        summarize(directory, state)


if __name__ == '__main__':
    main()
