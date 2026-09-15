"""Publication plots and seed-explicit endpoint summaries from measured runs."""
import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'ppo':'#4f46e5','dppo':'#16865c','glm5':'#e5484d'}
LABELS={'ppo':'PPO','dppo':'DPPO','glm5':'GLM-5'}
TASKS={'countdown':'Countdown','deepmath':'DeepMath · integer answers'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    args=p.parse_args()
    opener=gzip.open if args.results.suffix=='.gz' else open
    with opener(args.results,'rt') as f:data=json.load(f)
    assert len(data['runs'])==12, 'All predeclared arms are required for the final figures'
    args.assets.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Liberation Sans','sans-serif'],'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#bbc0c5','text.color':'#1f2328',
                         'axes.labelcolor':'#475059','xtick.color':'#475059','ytick.color':'#475059',
                         'axes.facecolor':'#fbfbfa','figure.facecolor':'#fbfbfa'})
    summary={'datasets':{},'uncertainty':'Paired task-bootstrap intervals condition on the two realized trained models and decoded outputs; they do not measure training-seed or inference-runtime uncertainty.'}
    rows=[]
    gate_rows=[]
    for task,title in TASKS.items():
        task_runs=[r for r in data['runs'] if r['config']['dataset']==task]
        by_alg={a:sorted([r for r in task_runs if r['config']['algorithm']==a],key=lambda r:r['config']['seed']) for a in COLORS}
        assert all([r['config']['seed'] for r in rs]==[42,43] for rs in by_alg.values())
        endpoint=task_runs[0]['evaluation'][-1]['updates']
        assert all(r['evaluation'][-1]['updates']==endpoint for r in task_runs)
        entry={'optimizer_updates':endpoint,'algorithms':{},'paired_vs_ppo':{}}
        baseline_maps=[{s['uid']:s['reward'] for s in r['evaluation'][0]['examples']} for r in task_runs]
        assert all(m.keys()==baseline_maps[0].keys() for m in baseline_maps)
        entry['same_checkpoint_baselines']={
            'run_accuracy':{r['name']:r['evaluation'][0]['accuracy'] for r in task_runs},
            'mean_accuracy':statistics.mean(r['evaluation'][0]['accuracy'] for r in task_runs),
            'min_accuracy':min(r['evaluation'][0]['accuracy'] for r in task_runs),
            'max_accuracy':max(r['evaluation'][0]['accuracy'] for r in task_runs),
            'mean_pairwise_reward_disagreement':statistics.mean(
                statistics.mean(a[k]!=b[k] for k in a) for a,b in itertools.combinations(baseline_maps,2)),
        }
        final_by_alg={}
        for a,runs in by_alg.items():
            final=np.array([r['evaluation'][-1]['accuracy'] for r in runs])
            steps=[s for r in runs for s in r['steps']]
            active_weight=sum(s['train/nonzero_adv_fraction'] for s in steps)
            gate_audit={name:{
                'conditional_gated_fraction':sum(s[f'train/audit_{name}_gated'] for s in steps)/active_weight,
                'conditional_mean_gradient_weight':sum(s[f'train/audit_{name}_weight'] for s in steps)/active_weight,
            } for name in ('ppo','dapo','cispo','glm5','sapo','dppo')}
            entry['algorithms'][a]={
                'seed_accuracy':{str(r['config']['seed']):r['evaluation'][-1]['accuracy'] for r in runs},
                'seed_baseline_accuracy':{str(r['config']['seed']):r['evaluation'][0]['accuracy'] for r in runs},
                'mean_accuracy':float(final.mean()),'min_accuracy':float(final.min()),'max_accuracy':float(final.max()),
                'mean_response_tokens':float(np.mean([r['evaluation'][-1]['mean_response_tokens'] for r in runs])),
                'baseline_mean_response_tokens':float(np.mean([r['evaluation'][0]['mean_response_tokens'] for r in runs])),
                'mean_truncation_fraction':float(np.mean([r['evaluation'][-1]['truncation_fraction'] for r in runs])),
                'baseline_mean_truncation_fraction':float(np.mean([r['evaluation'][0]['truncation_fraction'] for r in runs])),
                'wall_seconds':sum(r['wall_seconds'] for r in runs),
                'mean_same_weight_logprob_abs_diff':float(np.mean([r['mean_engine_logprob_abs_diff'] for r in runs])),
                'gate_audit_on_this_policy':gate_audit,
                'mean_zero_variance_group_fraction':statistics.mean(c['zero_variance_group_fraction'] for r in runs for c in r['rollout']),
                'training_response_tokens':sum(r['training_response_tokens'] for r in runs),
                'training_seconds':sum(r['train_seconds'] for r in runs),
                'generation_seconds':sum(r['generation_seconds'] for r in runs),
                'zero_gradient_step_fraction':statistics.mean(r['zero_gradient_step_fraction'] for r in runs),
                'mean_preclip_gradient_norm':statistics.mean(s['train/grad_norm'] for s in steps),
                'p95_preclip_gradient_norm':float(np.quantile([s['train/grad_norm'] for s in steps],.95)),
                'gradient_norm_clip_step_fraction':statistics.mean(s['train/grad_norm']>1 for s in steps),
            }
            examples=[{x['uid']:x['reward'] for x in r['evaluation'][-1]['examples']} for r in runs]
            ids=sorted(examples[0]);assert all(sorted(x)==ids for x in examples)
            final_by_alg[a]=(ids,np.array([[x[i] for i in ids] for x in examples]))
            for r in runs:
                end=r['evaluation'][-1]
                rows.append({'dataset':task,'surrogate':a,'seed':r['config']['seed'],
                             'baseline_accuracy':r['evaluation'][0]['accuracy'],'final_accuracy':end['accuracy'],
                             'final_mean_tokens':end['mean_response_tokens'],'final_truncation_fraction':end['truncation_fraction'],
                             'optimizer_steps':end['updates'],'wall_seconds':r['wall_seconds']})
                for diagnostic,values in r['gate_audit'].items():
                    gate_rows.append({'dataset':task,'trained_surrogate':a,'seed':r['config']['seed'],
                                      'diagnostic_surrogate':diagnostic,**values})
        for a in ('dppo','glm5'):
            assert final_by_alg[a][0]==final_by_alg['ppo'][0]
            differences=final_by_alg[a][1]-final_by_alg['ppo'][1]
            task_difference=differences.mean(axis=0)
            rng=np.random.default_rng(20260915)
            bootstrap=task_difference[rng.integers(0,len(task_difference),(10000,len(task_difference)))].mean(axis=1)
            lo,hi=np.quantile(bootstrap,[.025,.975])
            entry['paired_vs_ppo'][a]={'mean_difference':float(task_difference.mean()),
                                      'seed_differences':[float(x) for x in differences.mean(axis=1)],
                                      'task_bootstrap_95':[float(lo),float(hi)]}
        summary['datasets'][task]=entry
        observed=[e['accuracy']*100 for r in task_runs for e in r['evaluation']]
        y_low=max(0,10*np.floor((min(observed)-5)/10))
        y_high=min(100,10*np.ceil((max(observed)+5)/10))
        for mobile in (False,True):
            fig,ax=plt.subplots(figsize=(4.0,3.8) if mobile else (7.2,4.5),layout='constrained')
            for a,runs in by_alg.items():
                x=np.array([e['updates'] for e in runs[0]['evaluation']])
                assert all([e['updates'] for e in r['evaluation']]==list(x) for r in runs)
                y=np.array([[e['accuracy']*100 for e in r['evaluation']] for r in runs])
                mean=y.mean(axis=0)
                ax.errorbar(x,mean,yerr=np.vstack((mean-y.min(axis=0),y.max(axis=0)-mean)),
                            color=COLORS[a],label=LABELS[a],fmt='o-',markersize=3.5,
                            linewidth=1.3,elinewidth=.8,capsize=2)
            ax.set(xlabel='Optimizer updates',ylabel='Held-out grader success (%)',ylim=(y_low,y_high),xlim=(-8,endpoint+8))
            ticks=list(range(0,endpoint+1,128))
            if ticks[-1]!=endpoint:ticks.append(endpoint)
            ax.set_xticks([0,256,endpoint] if mobile else ticks)
            ax.set_yticks(np.arange(y_low,y_high+1,10 if y_high-y_low<=50 else 20));ax.grid(axis='y',color='#e2e3e5',linewidth=.65)
            ax.set_axisbelow(True);ax.tick_params(labelsize=10)
            ax.set_title(title,fontsize=13,loc='left',pad=14,weight='bold')
            fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',frameon=False,ncol=3,fontsize=10)
            suffix='-mobile' if mobile else ''
            stem=f'rl-surrogates-2026-{task}{suffix}'
            fig.savefig(args.assets/(stem+'.svg'))
            if not mobile:fig.savefig(args.assets/(stem+'.png'),dpi=220)
            plt.close(fig)
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(summary,indent=2))
    with args.summary.with_name('endpoints.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with args.summary.with_name('gate-audit.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(gate_rows[0]));writer.writeheader();writer.writerows(gate_rows)
    print('Saved two desktop/mobile curve figures, endpoint CSV, and paired summaries')


if __name__=='__main__':main()
