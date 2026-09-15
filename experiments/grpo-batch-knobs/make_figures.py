"""Create standalone research figures from completed, audited numeric records."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#64748b', '#4f46e5', '#d97706']


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def label(config):
    s = config['shape']
    result = f"P{s['prompts']} · G{s['group']} · B{s['batch']} · S{s['steps']}"
    if config.get('lr_multiplier', 1) != 1:
        result += f" · LR×{config['lr_multiplier']:.2g}"
    return result


def paired_prompt_interval(candidate, baseline, repeats=10000):
    """Resample task IDs once across seeds, preserving repeated evaluation IDs."""
    seeds = sorted(candidate)
    assert seeds == sorted(baseline)
    ids = sorted(candidate[seeds[0]])
    assert all(set(candidate[s]) == set(baseline[s]) == set(ids) for s in seeds)
    delta = np.array([[candidate[s][i] - baseline[s][i] for i in ids] for s in seeds])
    per_prompt = delta.mean(axis=0)
    rng = np.random.default_rng(1729)
    boots = np.concatenate([per_prompt[rng.integers(len(ids), size=(1000, len(ids)))].mean(axis=1)
                            for _ in range(repeats // 1000)])
    return {'mean_pp': float(per_prompt.mean()*100),
            'conditional_prompt_ci95_pp': (np.quantile(boots, [0.025, 0.975])*100).tolist(),
            'per_seed_difference_pp': {str(s): float(delta[i].mean()*100) for i,s in enumerate(seeds)}}


def curves(runs, task, selection, destination, x_key, mobile=False):
    fig, ax = plt.subplots(figsize=(3.5, 4.1) if mobile else (7.1, 4.5))
    for color, arm in zip(COLORS, selection):
        group = sorted([r for r in runs if r['config']['task'] == task and r['config']['arm'] == arm],
                       key=lambda r: r['config']['seed'])
        assert len(group) == 3
        sample_counts = [[p['samples'] for p in r['curves']] for r in group]
        assert sample_counts[0] == sample_counts[1] == sample_counts[2]
        x = np.array([[p[x_key] for p in r['curves']] for r in group])
        y = np.array([[p['accuracy']*100 for p in r['curves']] for r in group])
        if x_key.endswith('seconds'):
            x = x / 60
        elif x_key == 'output_tokens':
            x = x / 1e6
        elif x_key == 'samples':
            x = x / 1000
        ax.plot(x.mean(axis=0), y.mean(axis=0), color=color, marker='o', ms=3.6, lw=1.5,
                label=label(group[0]['config']), zorder=3)
        # These are observed seed ranges, deliberately not confidence bands.
        ax.vlines(x.mean(axis=0), y.min(axis=0), y.max(axis=0), color=color, alpha=.38, lw=1)
    ax.set_title({'gsm8k': 'GSM8K accuracy', 'countdown': 'Countdown accuracy'}[task], loc='left', weight='bold', pad=13)
    labels = {'active_phase_seconds': 'Active phase time (minutes)',
              'job_elapsed_seconds': 'Elapsed job time (minutes)', 'samples': 'Training responses (thousands)',
              'output_tokens': 'Generated output tokens (millions)', 'optimizer_steps': 'Optimizer updates'}
    ax.set_xlabel(labels[x_key])
    ax.set_ylabel('Validation accuracy (%)')
    ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#cbd5e1')
    ax.set_xlim(left=0)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.21), frameon=False, ncol=1, fontsize=8.5 if mobile else 10)
    fig.subplots_adjust(left=.19 if mobile else .1, right=.98, top=.89, bottom=.34)
    suffix = '-mobile' if mobile else ''
    axis = {'active_phase_seconds': 'active-time', 'job_elapsed_seconds': 'time', 'samples': 'samples',
            'output_tokens': 'tokens', 'optimizer_steps': 'updates'}[x_key]
    name = f'grpo-batch-knobs-{task}-{axis}{suffix}'
    for extension in ['svg', 'png']:
        fig.savefig(destination / f'{name}.{extension}', dpi=180, facecolor='#fbfbfa')
        if extension == 'svg':
            path = destination / f'{name}.{extension}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    runs = [r for r in load(args.results) if r['config']['phase'] == 'confirm']
    selection = json.loads(args.selection.read_text())['tasks']
    args.assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.facecolor': '#fbfbfa',
                         'text.color': '#20252b', 'axes.labelcolor': '#515963',
                         'xtick.color': '#515963', 'ytick.color': '#515963', 'svg.fonttype': 'none'})
    for task in ['gsm8k', 'countdown']:
        assert selection[task][0] == 'onpolicy-small'
        for x_key in ['job_elapsed_seconds', 'active_phase_seconds', 'samples', 'output_tokens', 'optimizer_steps']:
            for mobile in [False, True]:
                curves(runs, task, selection[task], args.assets, x_key, mobile)


if __name__ == '__main__':
    main()
