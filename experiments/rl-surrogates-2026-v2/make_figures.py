"""Paper-style curves from recorded measurements only; SVG/PNG/PDF outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LABELS={'ppo':'GRPO (PPO clip)','cispo':'CISPO','glm5':'GLM-5 gate','dppo':'DPPO','sapo':'SAPO','gspo':'GSPO'}
COLORS={'ppo':'#444444','cispo':'#D89000','glm5':'#D55E00','dppo':'#009E73','sapo':'#CC79A7','gspo':'#0072B2'}
MARKERS={'ppo':'o','cispo':'s','glm5':'^','dppo':'D','sapo':'P','gspo':'v'}


def read(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    state=json.loads((root/'study-status.json').read_text())
    plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'figure.dpi':120})
    for task in ['gsm8k','dapo']:
        rows=[]
        for entry in state['runs']:
            if entry['dataset']!=task:continue
            alg=entry['algorithm'];run=root/'runs'/entry['name']
            if not (run/'config.json').exists():continue
            cfg=json.loads((run/'config.json').read_text())
            k=cfg['prompts_per_rollout']*cfg['group_size']//cfg['global_batch_size']
            train=read(run/'measurements/rollouts.jsonl');ev=read(run/'measurements/eval.jsonl')
            if train or ev:rows.append((alg,k,train,ev))
        if not rows:continue
        for mobile in [False,True]:
            for diagnostic in [False,True]:
                fig,axes=plt.subplots(2 if mobile else 1,1 if mobile else 2,
                                     figsize=(4.8,7.2) if mobile else (11.5,4.5),layout='constrained')
                axes=np.asarray(axes).ravel();all_values=[[],[]]
                for alg,k,train,ev in rows:
                    x=np.array([r['rollout_id']*k for r in train])
                    if diagnostic:
                        for ax,key in zip(axes,['truncation_fraction','zero_variance_group_fraction']):
                            ax.plot(x,[100*r[key] for r in train],color=COLORS[alg],lw=1.8,
                                    marker=MARKERS[alg],ms=3,markevery=2)
                    else:
                        y=np.array([r['reward_mean']*100 for r in train])
                        if train:
                            axes[0].plot(x,y,color=COLORS[alg],alpha=.2,lw=1)
                            smoothed=np.array([y[max(0,i-4):i+1].mean() for i in range(len(y))])
                            axes[0].plot(x,smoothed,color=COLORS[alg],lw=2,
                                         marker=MARKERS[alg],ms=3,markevery=3)
                            all_values[0].extend(y)
                        if ev:
                            values=[np.mean(r['rewards'])*100 for r in ev]
                            axes[1].plot([r['optimizer_updates'] for r in ev],values,
                                         color=COLORS[alg],lw=1.8,marker=MARKERS[alg],ms=4)
                            all_values[1].extend(values)
                titles=['Responses reaching the cap','Groups with no reward variance'] if diagnostic else ['Training accuracy','Held-out accuracy']
                for index,(ax,title) in enumerate(zip(axes,titles)):
                    ax.set(title=title,xlabel='Optimizer updates',ylabel='Percent (%)' if diagnostic else 'Correct answers (%)')
                    ax.grid(axis='y',alpha=.2,lw=.7)
                    ax.set_xticks([0,64,128,192,256] if task=='gsm8k' else [0,64,128,192,256,320])
                    if diagnostic:ax.set_ylim(0,100)
                    elif all_values[index]:
                        low=max(0,5*np.floor((min(all_values[index])-3)/5))
                        high=min(100,5*np.ceil((max(all_values[index])+3)/5))
                        ax.set_ylim(low,high)
                handles=[Line2D([0],[0],color=COLORS[a],marker=MARKERS[a],lw=1.8,ms=4,label=LABELS[a]) for a,_,_,_ in rows]
                fig.legend(handles=handles,loc='outside lower center',ncol=2 if mobile else 3,
                           frameon=False,fontsize=12 if mobile else 13)
                stem=f"{task}-{'diagnostics' if diagnostic else 'accuracy'}"+('-mobile' if mobile else '')
                for suffix in ['svg','png','pdf']:
                    destination=out/f'{stem}.{suffix}'
                    fig.savefig(destination,dpi=220)
                    if suffix=='svg':
                        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
                plt.close(fig)
    (out/'CAPTION.txt').write_text('Qwen3.5-9B, full-parameter training in Miles on two B200s. '
        'Training curves show raw collection accuracy faintly and a trailing five-collection mean. '
        'Training reward is placed at the optimizer version that generated the responses. '
        'Held-out curves show recorded greedy accuracy on the fixed 512-question evaluation split. '
        'Accuracy-axis ranges differ between panels to show changes near the GSM8K ceiling. '
        'One training seed; no seed-variance bands are implied. Missing measurements are not filled in.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.root,a.out)
