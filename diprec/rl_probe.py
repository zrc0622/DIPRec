"""Frozen main-task probes: target likelihood and fixed SFT-negative margins."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def select_records(records, count, seed, split):
    if count < 1 or len(records) < count:
        raise ValueError(f'Need at least {count} {split} records')
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError(f'Duplicate sample IDs in {split}')
    return sorted(records, key=lambda r: hashlib.sha256(
        f"{seed}:{split}:{r['sample_id']}".encode()).digest())[:count]


def score_sequences(model, prompt_ids, sequences):
    """Sum full-vocabulary log probabilities of SID + EOS, with no prompt loss."""
    import torch

    device = next(model.parameters()).device
    scores = []
    supports_tail = 'logits_to_keep' in inspect.signature(model.forward).parameters
    with torch.inference_mode():
        for continuation in sequences:
            if not prompt_ids or not continuation:
                raise ValueError('Empty prompt or continuation')
            full = list(prompt_ids) + list(continuation)
            ids = torch.tensor([full], device=device)
            extra = {'logits_to_keep': len(continuation) + 1} if supports_tail else {}
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **extra).logits
            # Last logit predicts the token after EOS and is excluded.
            logits = logits[0, -len(continuation)-1:-1].float()
            targets = torch.tensor(continuation, device=device)
            score = torch.log_softmax(logits, -1).gather(1, targets[:, None]).sum().item()
            if not math.isfinite(score):
                raise ValueError('Nonfinite probe sequence score')
            scores.append(score)
    return scores


def probe_statistics(scores, baseline):
    if len(scores) != len(baseline) or len(scores) < 2:
        raise ValueError('Probe needs the same target and at least one fixed negative')

    def normalized(values):
        maximum = max(values)
        z = maximum + math.log(sum(math.exp(v-maximum) for v in values))
        return [v-z for v in values]

    p, q = normalized(baseline), normalized(scores)
    return dict(target_logp=scores[0], target_probability=math.exp(scores[0]),
                target_margin=scores[0]-max(scores[1:]),
                fixed_candidate_rank=1+sum(s >= scores[0] for s in scores[1:]),
                delta_target_logp=scores[0]-baseline[0],
                delta_target_margin=(scores[0]-max(scores[1:]))-(baseline[0]-max(baseline[1:])),
                candidate_set_kl_sft_to_policy=sum(math.exp(x)*(x-y) for x,y in zip(p,q)),
                candidate_logp_abs_change=sum(abs(x-y) for x,y in zip(scores,baseline))/len(scores))


def source_files(args):
    return [Path(args.data_dir)/f'{split}.jsonl' for split in ('train','valid')] + [
        Path(args.data_dir)/'manifest.json', Path(args.sid_index), Path(args.item_meta)]


def load_checked_model(checkpoint, args, manifest):
    from .data import load_sid_map, validate_checkpoint_training_contract
    from .runtime import load_model_runtime
    import torch

    config = json.loads((Path(checkpoint)/'training_config.json').read_text())
    if config['method'] not in ('minionerec_sft', 'minionerec_rl'):
        raise ValueError('Probe only supports MiniOneRec SFT/RL')
    validate_checkpoint_training_contract(checkpoint, expected_method=config['method'],
                                          manifest=manifest, item_meta_path=args.item_meta)
    sid_map = load_sid_map(args.sid_index)
    model, tokenizer, _, _ = load_model_runtime(checkpoint, sid_map, 'disjoint_rows', training=False, include_interest=False)
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return model, tokenizer, sid_map, config


def prepare(args):
    from .data import read_jsonl, validate_history_records, validate_manifest_sid_index
    from .constraints import build_sid_trie
    from .evaluation import _generate_catalog_beams
    from .prompts import history_prompt, messages
    from .runtime import apply_chat_template

    output = Path(args.output)
    if output.exists():
        raise ValueError(f'Frozen probe already exists: {output}')
    manifest = json.loads((Path(args.data_dir)/'manifest.json').read_text())
    validate_manifest_sid_index(manifest, args.sid_index)
    model, tokenizer, sid_map, config = load_checked_model(args.sft, args, manifest)
    if config['method'] != 'minionerec_sft':
        raise ValueError('Probe negatives must be built from the initial SFT')
    trie = build_sid_trie(tokenizer, sid_map)
    rows = []
    for split in ('train','valid'):
        records = read_jsonl(Path(args.data_dir)/f'{split}.jsonl')
        validate_history_records(records, 50, manifest)
        for record in select_records(records, args.samples, 42, split):
            text = history_prompt(record, 50, reasoning=False)
            prompt = apply_chat_template(tokenizer, messages(text), True)
            _, candidates, _ = _generate_catalog_beams(model, tokenizer, trie, prompt, 80, 2048)
            target = record['target_item_sid']
            negative = list(dict.fromkeys(''.join(c) for c in candidates if ''.join(c) != target))[:5]
            if len(negative) != 5:
                raise ValueError('SFT beam search must provide five distinct wrong candidates')
            texts = [target] + negative
            sequences = [tokenizer.encode(s, add_special_tokens=False)+[tokenizer.eos_token_id] for s in texts]
            if any(len(s) != 4 for s in sequences):
                raise ValueError('Each probe SID must be three tokens plus EOS')
            rows.append(dict(split=split, sample_id=record['sample_id'], prompt=text,
                             prompt_ids=prompt, candidate_sids=texts, candidate_ids=sequences,
                             sft_scores=score_sequences(model, prompt, sequences)))
            if len(rows) % 16 == 0:
                print(f'prepared probe records={len(rows)}/{2*args.samples}', flush=True)
    write_json(output, dict(schema='diprec.rl_probe.v1', sft=str(args.sft), samples_per_split=args.samples,
                            source_sha256={str(p):digest(p) for p in source_files(args)},
                            sft_training_config=config, rows=rows,
                            note='Frozen SFT top-five wrong candidates plus target; full-vocabulary SID+EOS logp. Candidate-set KL is NOT full-policy KL.'))


def evaluate(args):
    from .prompts import messages
    from .runtime import apply_chat_template

    frozen = json.loads(Path(args.probe).read_text())
    if frozen['schema'] != 'diprec.rl_probe.v1':
        raise ValueError('Unknown probe schema')
    if {str(p):digest(p) for p in source_files(args)} != frozen['source_sha256']:
        raise ValueError('Probe source files changed')
    manifest = json.loads((Path(args.data_dir)/'manifest.json').read_text())
    model, tokenizer, _, config = load_checked_model(args.model, args, manifest)
    if config['method'] == 'minionerec_rl' and config['model'] != frozen['sft']:
        raise ValueError('RL checkpoint does not use the frozen probe SFT parent')
    if config['method'] == 'minionerec_sft' and str(args.model) != frozen['sft']:
        raise ValueError('Wrong SFT baseline for probe')
    rows = []
    for row in frozen['rows']:
        prompt = apply_chat_template(tokenizer, messages(row['prompt']), True)
        candidates = [tokenizer.encode(s, add_special_tokens=False)+[tokenizer.eos_token_id] for s in row['candidate_sids']]
        if prompt != row['prompt_ids'] or candidates != row['candidate_ids']:
            raise ValueError('Checkpoint tokenizer does not match the frozen probe')
        scores = score_sequences(model, prompt, candidates)
        rows.append(dict(split=row['split'],sample_id=row['sample_id'],scores=scores,
                         **probe_statistics(scores,row['sft_scores'])))
    summaries = {}
    for split in ('train','valid'):
        selected = [r for r in rows if r['split'] == split]
        keys = probe_statistics([0.,-1.],[0.,-1.]).keys()
        summaries[split] = {key:sum(r[key] for r in selected)/len(selected) for key in keys}
        summaries[split]['samples'] = len(selected)
    write_json(args.output,dict(checkpoint=str(args.model),probe_sha256=digest(args.probe),
                                training_config=config,summary=summaries,rows=rows,
                                note=frozen['note']))
    print(json.dumps(summaries,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','evaluate'))
    parser.add_argument('--data_dir', default='data/processed/Office_Products/history_50')
    parser.add_argument('--sid_index', default='data/Amazon/index/Office_Products.index.json')
    parser.add_argument('--item_meta', default='data/Amazon/index/Office_Products.item.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sft')
    parser.add_argument('--model')
    parser.add_argument('--probe')
    parser.add_argument('--samples', type=int, default=128)
    args = parser.parse_args()
    if args.action == 'prepare' and not args.sft:
        parser.error('prepare requires --sft')
    if args.action == 'evaluate' and (not args.model or not args.probe):
        parser.error('evaluate requires --model and --probe')
    from .runtime import set_seed
    set_seed(42)
    (prepare if args.action == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()
