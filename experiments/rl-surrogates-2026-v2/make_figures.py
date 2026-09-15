"""Paper-style curves from recorded measurements only; SVG/PNG/PDF outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'ppo':'GRPO (PPO clip)','cispo':'CISPO','glm5':'GLM-5 gate','dppo':'DPPO','sapo':'SAPO','gspo':'GSPO'}
COLORS={'ppo':'#444444','cispo':'#D89000','glm5':'#D55E00','dppo':'#009E73','sapo':'#CC79A7','gspo':'#0072B2'}


def read(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    state=json.loads((root/'study-status.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'figure.dpi':120})
    for task in ['gsm8k','dapo']:
        fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
        used=[]
        for entry in state['runs']:
            if entry['dataset']!=task:continue
            alg=entry['algorithm'];run=root/'runs'/entry['name'];cfg=json.loads((run/'config.json').read_text())
            k=cfg['prompts_per_rollout']*cfg['group_size']//cfg['global_batch_size']
            train=read(run/'measurements/rollouts.jsonl');ev=read(run/'measurements/eval.jsonl')
            if train:
                x=np.array([r['rollout_id']*k for r in train]);y=np.array([r['reward_mean']*100 for r in train])
                axes[0].plot(x,y,color=COLORS[alg],alpha=.25,lw=1)
                smoothed=np.array([y[max(0,i-4):i+1].mean() for i in range(len(y))])
                axes[0].plot(x,smoothed,color=COLORS[alg],label=LABELS[alg],lw=2)
            if ev:
                axes[1].plot([r['optimizer_updates'] for r in ev],[np.mean(r['rewards'])*100 for r in ev],
                             color=COLORS[alg],label=LABELS[alg],lw=2,marker='o',ms=3)
            used.append(alg)
        if not used:plt.close(fig);continue
        for ax,title in zip(axes,['Training accuracy','Held-out accuracy']):
            ax.set(title=title,xlabel='Optimizer updates',ylabel='Correct answers (%)')
            ax.grid(axis='y',alpha=.2);ax.set_ylim(0,100)
        handles,labels=axes[0].get_legend_handles_labels()
        if not handles:handles,labels=axes[1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle('GSM8K' if task=='gsm8k' else 'DAPO-Math-17K',fontweight='bold')
        for suffix in ['svg','png','pdf']:fig.savefig(out/f'{task}-accuracy.{suffix}',dpi=220)
        plt.close(fig)
    (out/'CAPTION.txt').write_text('Qwen3.5-9B, full-parameter training in Miles on two B200s. '
        'Training curves show raw collection accuracy faintly and a trailing five-collection mean. '
        'Training reward is placed at the optimizer version that generated the responses. '
        'Held-out curves show recorded greedy accuracy on the fixed 512-question evaluation split. '
        'One training seed; no seed-variance bands are implied. Missing measurements are not filled in.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.root,a.out)
