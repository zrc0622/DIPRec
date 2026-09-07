#!/usr/bin/env python3
"""Serial, fixed-budget MiniOneRec prefix-strength experiments (stdlib only)."""

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
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('Office_Products/history_50/Qwen_Qwen3-0.6B')
SFT_TAG = 'sft6e_lr1e-4_best'
SFT = Path('output_dir') / BASE / 'minionerec_sft' / f'seed_42_{SFT_TAG}/best_checkpoint'
METRICS = ('Recall@10', 'NDCG@10', 'Recall@5', 'NDCG@5', 'sid_level1_hit', 'sid_level2_hit')
VARIABLES = {'reward_mode', 'prefix_reward_strength', 'output_dir',
             'diagnostics_file', 'training_metrics_file'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def run_dir(tag, kind='outputs'):
    return ROOT / kind / BASE / 'minionerec_rl' / f'seed_42_{tag}'


def command(tag, mode, strength):
    return ['bash', 'scripts/run_experiment.sh',
            '--method', 'minionerec_rl', '--dataset', 'Office_Products',
            '--model', 'Qwen/Qwen3-0.6B', '--max_history_len', '50', '--seed', '42',
            '--sft_run_tag', SFT_TAG, '--require_existing_sft', '--run_tag', tag,
            '--baseline_rl_task_scope', 'official_mixed', '--baseline_rl_reference_mode', 'fixed',
            '--baseline_rl_per_device_batch_size', '16', '--baseline_rl_gradient_accumulation_steps', '4',
            '--baseline_rl_generation_batch_size', '256', '--baseline_rl_learning_rate', '2e-6',
            '--baseline_rl_beta', '1e-2', '--baseline_rl_num_epochs', '1',
            '--baseline_rl_reward_mode', mode, '--baseline_rl_prefix_reward_strength', str(strength),
            '--baseline_rl_stop_after_steps', '1000', '--baseline_rl_diagnostics',
            '--eval_beams', '10', '--eval_candidate_budget', '80', '--eval_split', 'valid']


def validate_metrics(path, mode=None, strength=None):
    data = read_json(path)
    if data['split'] != 'valid' or data['num_examples'] <= 0:
        raise ValueError(f'Expected nonempty validation results: {path}')
    for key in METRICS:
        if not math.isfinite(data['metrics'][key]):
            raise ValueError(f'Nonfinite {key}: {path}')
    if mode is not None:
        cfg = data['training_config']
        expected = dict(method='minionerec_rl', model=str(SFT), seed=42,
                        reward_mode=mode, prefix_reward_strength=strength,
                        completed_optimizer_steps=1000, stop_after_steps=1000,
                        scheduler_total_steps=3455, learning_rate=2e-6, beta=.01,
                        num_epochs=1, task_scope='official_mixed', reference_mode='fixed',
                        num_generations=16, num_iterations=1, temperature=1.0)
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Run does not match the fixed pilot protocol: {path}')
        batch = cfg['batch']
        if any(batch.get(k) != v for k, v in dict(world_size=4, effective_update_batch=256,
                generation_batch_size=256, per_device_batch_size=16, gradient_accumulation_steps=4).items()):
            raise ValueError(f'Unexpected batch contract: {path}')
    return data


def compatible_eval(a, b):
    normalize = lambda d: {'sft_num_plans': 0, 'sft_objective': None, 'sft_plan_mode': None} | d['evaluation_config']
    return (a['data_manifest'] == b['data_manifest'] and
            a['num_examples'] == b['num_examples'] and normalize(a) == normalize(b))


def fixed_training(data):
    return {k: v for k, v in data['training_config'].items() if k not in VARIABLES}


def diagnostics(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if 'grad_norm' in r]
    if [r['step'] for r in rows] != list(range(1, 1001)):
        raise ValueError(f'Expected exactly 1000 ordered training records: {path}')
    tasks = ('history_sid_to_sid', 'title_to_sid', 'description_to_sid', 'title_history_to_sid')
    total = hit = aux = main = main_aux = aux_mass = 0
    for row in rows:
        for task in tasks:
            p = f'prefix_aux/{task}/'
            n = row.get(p + 'groups', 0)
            total += n
            hit += n * row.get(p + 'exact_hit', 0)
            active = n * row.get(p + 'aux_active', 0)
            aux += active
            aux_mass += n * row.get(p + 'aux_abs_mean', 0)
            if task == tasks[0]:
                main += n
                main_aux += active
    if total != 16000 or not main:
        raise ValueError(f'Unexpected training group count: {path}')
    return dict(main_aux_active=main_aux / main, task_advantage_coverage=(hit + aux) / total,
                global_aux_abs_mean=aux_mass / total,
                kl_mean=sum(r['kl'] for r in rows) / len(rows), kl_max=max(r['kl'] for r in rows))


def summarize(directory, state):
    records = []
    control = None
    for arm in state['arms']:
        if arm['status'] != 'complete':
            continue
        path = run_dir(arm['run_tag']) / 'valid_metrics.json'
        data = validate_metrics(path, arm['mode'], arm['strength'])
        if control is None:
            control = data
        if not compatible_eval(control, data) or fixed_training(control) != fixed_training(data):
            raise ValueError(f'Incomparable sweep arm: {path}')
        records.append((arm['label'], 'sweep', path, data, diagnostics(path.with_name('rl_diagnostics.jsonl'))))

    # Historical rows are references only, never silently reused as new sweep arms.
    references = [('SFT', ROOT / 'outputs' / BASE / 'minionerec_sft' / f'seed_42_{SFT_TAG}/valid_metrics.json', None)]
    for mode in ('official', 'main_miss_prefix'):
        tag = f'rl_prefix_ab_{mode}_s1000_lr2e-6_beta1e-2_4gpu_eb256'
        references.append((f'previous_{mode}', run_dir(tag) / 'valid_metrics.json', mode))
    warnings = []
    for label, path, mode in references:
        try:
            data = validate_metrics(path, mode, .1 if mode else None)
            if control and (not compatible_eval(control, data) or
                            (mode and fixed_training(control) != fixed_training(data))):
                raise ValueError('configuration differs from the current sweep')
            if mode is None and control and control['training_config']['model'] != data['checkpoint']:
                raise ValueError('SFT checkpoint path differs from the recorded parent')
            records.append((label, 'reference', path, data, {}))
        except (OSError, ValueError, KeyError) as exc:
            warnings.append(f'{label}: {exc}')
    sft = next((d for label, _, _, d, _ in records if label == 'SFT'), None)
    results = []
    for label, source, path, data, diag in records:
        row = dict(arm=label, source=source, metrics_file=str(path.relative_to(ROOT)),
                   **{k: data['metrics'][k] for k in METRICS}, **diag)
        for name, ref in [('SFT', sft), ('official', control)]:
            for key in METRICS[:2]:
                row[f'delta_{key}_vs_{name}'] = data['metrics'][key] - ref['metrics'][key] if ref else None
        results.append(row)
    write_json(directory / 'summary.json', dict(results=results, warnings=warnings,
               note='Validation point estimates only; historical rows are references, not new trials.'))
    columns = list(dict.fromkeys(k for r in results for k in r))
    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    print('\narm                         Valid R@10   NDCG@10   R@5', flush=True)
    for r in results:
        print(f"{r['arm']:<28} {r['Recall@10']:.6f}   {r['NDCG@10']:.6f}   {r['Recall@5']:.6f}", flush=True)
    for warning in warnings:
        print(f'Reference omitted: {warning}', flush=True)


def execute(cmd, env, log):
    with log.open('w', encoding='utf-8') as f:
        f.write(shlex.join(cmd) + '\n')
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors='replace', start_new_session=True)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
        finally:
            proc.stdout.close()
    if code:
        raise RuntimeError(f'Experiment exited {code}; see {log}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep_tag', default=datetime.now().strftime('prefix_strength_%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'))
    parser.add_argument('--strengths', nargs='+', type=float, default=[.3, .5, 1.0],
                        help='Two or three distinct prefix strengths in (0,1]; official is always first.')
    parser.add_argument('--dry_run', action='store_true', help='Print commands only; no files or training.')
    parser.add_argument('--resume', action='store_true', help='Skip verified completed arms; retry incomplete arms with fresh tags.')
    parser.add_argument('--summarize_only', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.sweep_tag):
        parser.error('Invalid sweep tag')
    if len(args.strengths) not in (2, 3) or len(set(args.strengths)) != len(args.strengths) or any(
            not math.isfinite(x) or not 0 < x <= 1 for x in args.strengths):
        parser.error('--strengths requires two or three distinct finite values in (0,1]')
    gpu_ids = args.gpus.split(',')
    if len(gpu_ids) != 4 or any(not g.strip() for g in gpu_ids) or len(set(gpu_ids)) != 4:
        parser.error('This fixed batch protocol requires exactly four distinct visible GPUs')
    env = os.environ | dict(CUDA_VISIBLE_DEVICES=args.gpus, DIPREC_DDP='1', DIPREC_NUM_PROCESSES='4',
                           PYTHONUNBUFFERED='1')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    arms = [dict(label='official', mode='official', strength=.1)] + [
        dict(label='prefix_' + str(x), mode='main_miss_prefix', strength=x) for x in args.strengths]
    for arm in arms:
        arm.update(status='pending', attempts=[])
        arm['base_tag'] = f"rl_{args.sweep_tag}_{arm['label'].replace('.', 'p')}_s1000"
    if args.dry_run:
        print(f'CUDA_VISIBLE_DEVICES={args.gpus} DIPREC_DDP=1 DIPREC_NUM_PROCESSES=4')
        for arm in arms:
            print(shlex.join(command(arm['base_tag'] + '_a1', arm['mode'], arm['strength'])))
        return
    directory = ROOT / 'outputs' / BASE / 'prefix_sweeps' / args.sweep_tag
    if args.summarize_only or args.resume:
        if not (directory / 'state.json').is_file():
            parser.error(f'No saved sweep: {directory}')
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory / 'state.json'
        if args.resume or args.summarize_only:
            state = read_json(state_path)
            if not args.summarize_only and state['strengths'] != args.strengths:
                parser.error('Resume with the same --strengths as the saved sweep')
        else:
            state = dict(strengths=args.strengths, gpus=args.gpus, arms=arms)
            write_json(state_path, state)
        if args.summarize_only:
            summarize(directory, state)
            return
        # The inner runner checks parent compatibility; this preflight also prevents
        # accidental SFT creation before the first expensive experiment.
        parent = ROOT / SFT
        if not all((parent / f).is_file() for f in ('config.json', 'training_config.json')) or not (
                list(parent.glob('model*.safetensors')) or list(parent.glob('pytorch_model*.bin'))):
            raise ValueError(f'Existing SFT weights/config required: {parent}')
        summarize(directory, state)  # Verify completed arms before launching anything.
        for arm in state['arms']:
            if arm['status'] == 'complete':
                print(f"Skipping verified completed arm: {arm['label']}", flush=True)
                continue
            attempt = len(arm['attempts']) + 1
            tag = f"{arm['base_tag']}_a{attempt}"
            if any(run_dir(tag, kind).exists() for kind in ('outputs', 'output_dir')):
                raise ValueError(f'Refusing existing run tag: {tag}')
            cmd = command(tag, arm['mode'], arm['strength'])
            log = directory / f"{arm['label']}_a{attempt}.log"
            arm.update(status='running', run_tag=tag)
            entry = dict(run_tag=tag, command=cmd, log=log.name, gpus=args.gpus, status='running')
            arm['attempts'].append(entry)
            write_json(state_path, state)
            print(f"\nStarting {arm['label']}: {tag}\nLog: {log}", flush=True)
            try:
                execute(cmd, env, log)
                validate_metrics(run_dir(tag) / 'valid_metrics.json', arm['mode'], arm['strength'])
                diagnostics(run_dir(tag) / 'rl_diagnostics.jsonl')
                arm['status'] = 'complete'
                summarize(directory, state)
            except BaseException as exc:
                arm['status'] = entry['status'] = 'failed'
                entry['error'] = str(exc)
                write_json(state_path, state)
                raise
            entry['status'] = 'complete'
            write_json(state_path, state)
        print(f'\nSweep complete. Results: {directory / "summary.csv"}', flush=True)


if __name__ == '__main__':
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'Sweep stopped: {exc}', file=sys.stderr)
        sys.exit(1)
