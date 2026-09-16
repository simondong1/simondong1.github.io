"""Audit completed runs on CPU, leaving the training queue untouched."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from rewards import score


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def audit(run, root, output):
    config = json.loads((run / 'config.json').read_text())
    labels = {}
    for split in ['train', 'test']:
        for row in records(root / 'data' / f"{config['dataset']}_{split}.jsonl"):
            label = json.loads(row['label']) if isinstance(row['label'], str) else row['label']
            labels[row['metadata']['uid']] = label['answer']
    checked = 0
    for filename in ['rollouts.jsonl', 'eval.jsonl']:
        for collection in records(run / 'measurements' / filename):
            for sample in collection['samples']:
                actual = score(sample['response'], labels[sample['uid']])
                assert actual == sample['reward'], (run.name, sample['uid'], 'reward differs on replay')
                checked += 1
    differences, response_means, outliers = [], [], []
    paths = sorted((run / 'train_data').glob('*.pt'))
    assert len(paths) == 2 * config['rollouts'], (run.name, 'incomplete training dumps')
    for path in paths:
        dump = torch.load(path, map_location='cpu', weights_only=False)
        data = dump['rollout_data']
        for index, (p, q, mask) in enumerate(zip(data['log_probs'], data['rollout_log_probs'], data['loss_masks'])):
            p, q, mask = p.float(), q.float(), mask.bool()
            assert p.shape == q.shape == mask.shape
            absolute = (p - q)[mask].abs().numpy()
            assert np.isfinite(absolute).all() and len(absolute)
            differences.append(absolute)
            response_means.append(float(absolute.mean()))
            if absolute.max() > 1:
                position = int(torch.where(mask)[0][int(absolute.argmax())])
                outliers.append({'collection': dump['rollout_id'], 'rank': dump['rank'],
                                 'sample_index': int(data['sample_indices'][index]),
                                 'response_position': position, 'trainer_logp': float(p[position]),
                                 'inference_logp': float(q[position]), 'abs_diff': float(absolute.max())})
    values = np.concatenate(differences)
    result = {'run': run.name, 'rewards_replayed': checked, 'scored_tokens': len(values),
              'response_mean_abs_logprob_difference': float(np.mean(response_means)),
              'token_mean_abs_logprob_difference': float(values.mean()),
              'token_quantiles': dict(zip(['median', 'p99', 'p999', 'p9999', 'max'],
                                         np.quantile(values, [.5, .99, .999, .9999, 1]).tolist())),
              'tokens_above': {str(t): int((values > t).sum()) for t in [1, 5, 20]},
              'largest_response_outliers': sorted(outliers, key=lambda x: x['abs_diff'], reverse=True)[:10]}
    output.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/rl-surrogate-2026-v2'))
    parser.add_argument('--model', type=Path, default=Path('/tmp/rl-surrogate-2026/model'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = args.root / 'audits'
    directory.mkdir(exist_ok=True)
    state = json.loads((args.root / 'study-status.json').read_text())
    for job in state['runs']:
        if job['state'] != 'complete':
            continue
        run = args.root / 'runs' / job['name']
        checkpoint_audit = directory / (job['name'] + '-checkpoint.json')
        if not checkpoint_audit.exists():
            iteration = int((run / 'checkpoints/latest_checkpointed_iteration.txt').read_text())
            subprocess.run([sys.executable, str(Path(__file__).with_name('audit_checkpoint.py')),
                            '--checkpoint', str(run / 'checkpoints' / f'iter_{iteration:07d}'),
                            '--model', str(args.model), '--output', str(checkpoint_audit)], check=True,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2'})
        measurement_audit = directory / (job['name'] + '-measurements.json')
        if not measurement_audit.exists():
            result = audit(run, args.root, measurement_audit)
            print(json.dumps({k: v for k, v in result.items() if k != 'largest_response_outliers'}), flush=True)


if __name__ == '__main__':
    main()
