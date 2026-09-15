"""Single-seed generation-volume screen; no uncertainty inferred from one run."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_figures import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    runs = [r for r in load(args.results) if r['config']['phase'] == 'screen']
    assert len(runs) == 22
    arms = ['onpolicy-small', 'split-2', 'split-4', 'split-8']
    args.assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.facecolor': '#fbfbfa',
                         'text.color': '#20252b', 'axes.labelcolor': '#515963', 'svg.fonttype': 'none'})
    for mobile in [False, True]:
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.2) if mobile else (7.1, 5.2), sharex=True)
        for ax, task, color in zip(axes, ['gsm8k', 'countdown'], ['#4f46e5', '#d97706']):
            selected = [next(r for r in runs if r['config']['task'] == task and r['config']['arm'] == a) for a in arms]
            x = [r['config']['shape']['prompts'] * 8 for r in selected]
            y = [r['summary']['job_seconds'] / 60 for r in selected]
            ax.plot(x, y, 'o-', color=color, ms=4, lw=1.4)
            ax.set_title('GSM8K' if task == 'gsm8k' else 'Countdown', loc='left', weight='bold', fontsize=11)
            ax.set_ylabel('Job time (min)')
            ax.set_xscale('log', base=2)
            ax.set_xticks(x, [str(v) for v in x])
            ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['bottom', 'left']].set_color('#cbd5e1')
        axes[-1].set_xlabel('Responses per collection')
        fig.subplots_adjust(left=.2 if mobile else .12, right=.97, top=.93, bottom=.11, hspace=.45)
        name = 'grpo-batch-knobs-screen-time' + ('-mobile' if mobile else '')
        for extension in ['svg', 'png']:
            fig.savefig(args.assets / f'{name}.{extension}', dpi=180, facecolor='#fbfbfa')
        plt.close(fig)


if __name__ == '__main__':
    main()
