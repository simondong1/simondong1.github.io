"""Collect measured learning curves, endpoints and audit records for publication."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def compact(collection):
    return {**{k:v for k,v in collection.items() if k not in ['samples','rewards','truncated']},
            'samples': [{k:v for k,v in row.items() if k != 'response'} for row in collection['samples']]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/tmp/rl-surrogate-2026-v2'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();root=args.root;out=args.out;out.mkdir(parents=True,exist_ok=True)
    state=json.loads((root/'study-status.json').read_text())
    if not args.allow_incomplete:assert state['state']=='complete' and len(state['runs'])==12
    source=Path(__file__).parent
    for name,digest in state['code_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,('Training source changed',name)
    manifest=json.loads((root/'data/manifest.json').read_text())
    for split,meta in manifest['splits'].items():
        assert hashlib.sha256((root/'data'/f'{split}.jsonl').read_bytes()).hexdigest()==meta['sha256'],('Data changed',split)
    results={'study':state,'data_manifest':manifest,'runs':[]};summary={'state':state['state'],'datasets':{}}
    endpoint_rows=[];gate_rows=[]
    for job in state['runs']:
        if job['state']!='complete':continue
        run=root/'runs'/job['name'];cfg=json.loads((run/'config.json').read_text())
        scalars=json.loads((run/'scalars.json').read_text())
        train=read_lines(run/'measurements/rollouts.jsonl');evaluation=read_lines(run/'measurements/eval.jsonl')
        updates=cfg['rollouts']*cfg['prompts_per_rollout']*cfg['group_size']//cfg['global_batch_size']
        assert [x['step'] for x in scalars['train/loss']]==list(range(updates)),(job['name'],'Missing optimizer logs')
        assert len(train)==cfg['rollouts'] and evaluation[-1]['optimizer_updates']==updates
        assert [x['optimizer_updates'] for x in evaluation]==list(range(0,updates+1,64))
        assert all(math.isfinite(row['value']) for series in scalars.values() for row in series)
        checkpoint=json.loads((root/'audits'/f"{job['name']}-checkpoint.json").read_text())
        measurement=json.loads((root/'audits'/f"{job['name']}-measurements.json").read_text())
        baseline={s['uid']:s for s in evaluation[0]['samples']};final={s['uid']:s for s in evaluation[-1]['samples']}
        assert set(baseline)==set(final) and len(final)==512
        samples=list(final.values());means=lambda key:statistics.mean(r['value'] for r in scalars[key])
        active=sum(x['value'] for x in scalars['train/nonzero_adv_fraction'])
        for algorithm in ['ppo','dapo','cispo','glm5','dppo','sapo','gspo']:
            gate_rows.append({'run':job['name'],'counterfactual':algorithm,
                              'conditional_gated_fraction':sum(x['value'] for x in scalars[f'train/audit_{algorithm}_gated'])/active,
                              'conditional_weight':sum(x['value'] for x in scalars[f'train/audit_{algorithm}_weight'])/active})
        gpu=read_lines(run/'gpu.jsonl');memory=[]
        for record in gpu:
            memory.extend(float(row.split(',')[2])/1024 for row in record.get('rows',[]))
        endpoint={'run':job['name'],'dataset':job['dataset'],'algorithm':job['algorithm'],'seed':job['seed'],
                  'updates':updates,'baseline_accuracy':statistics.mean(s['reward'] for s in baseline.values()),
                  'final_accuracy':statistics.mean(s['reward'] for s in samples),
                  'best_accuracy':max(statistics.mean(e['rewards']) for e in evaluation),
                  'final_truncation_fraction':statistics.mean(s['status']=='TRUNCATED' for s in samples),
                  'final_response_tokens':statistics.mean(s['response_length'] for s in samples),
                  'zero_gradient_step_fraction':statistics.mean(x['value']==0 for x in scalars['train/grad_norm']),
                  'mean_zero_variance_group_fraction':statistics.mean(x['zero_variance_group_fraction'] for x in train),
                  'conditional_gated_fraction':sum(x['value'] for x in scalars['train/pg_clipfrac'])/active,
                  'mean_behavior_approx_kl':means('train/approx_kl_behavior'),
                  'mean_initial_engine_logprob_abs_diff':means('train/initial_engine_logprob_abs_diff'),
                  'max_log_ratio_guard_fraction':max(x['value'] for x in scalars['train/ratio_log_clamp_fraction']),
                  'training_generated_tokens':sum(x['generated_tokens'] for x in train),
                  'wall_seconds':job['end']-job['start'],'peak_gpu_memory_gib':max(memory)}
        endpoint_rows.append(endpoint)
        results['runs'].append({'name':job['name'],'config':cfg,'endpoint':endpoint,
                                'training':[compact(x) for x in train],'evaluation':[compact(x) for x in evaluation],
                                'scalars':scalars,'checkpoint_audit':checkpoint,'measurement_audit':measurement})
    rng=np.random.default_rng(20260915)
    for dataset in ['gsm8k','dapo']:
        runs=[r for r in results['runs'] if r['config']['dataset']==dataset]
        if not runs:continue
        initial=[{s['uid']:s['reward'] for s in r['evaluation'][0]['samples']} for r in runs]
        uid_order=sorted(initial[0]);assert all(set(x)==set(uid_order) for x in initial)
        pair_disagreements=[np.mean([a[u]!=b[u] for u in uid_order]) for a,b in itertools.combinations(initial,2)]
        accuracies=[statistics.mean(x.values()) for x in initial]
        baseline=next((r for r in runs if r['config']['algorithm']=='ppo'),None);comparisons={}
        if baseline:
            original={s['uid']:s['reward'] for s in baseline['evaluation'][-1]['samples']}
            for run in runs:
                algorithm=run['config']['algorithm']
                if algorithm=='ppo':continue
                other={s['uid']:s['reward'] for s in run['evaluation'][-1]['samples']}
                difference=np.array([other[u]-original[u] for u in uid_order])
                bootstrap=rng.choice(difference,size=(3000,len(difference)),replace=True).mean(axis=1)
                comparisons[algorithm]={'difference_vs_ppo':float(difference.mean()),
                                        'conditional_task_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist()}
        # Verify the actual prompt groups, not only the data-file hashes.
        prompt_sequences=[]
        for run in runs:
            prompt_sequences.append([sorted((s['uid'],s['group_index']) for s in x['samples']) for x in run['training']])
        assert all(x==prompt_sequences[0] for x in prompt_sequences)
        summary['datasets'][dataset]={'endpoints':[r['endpoint'] for r in runs],
                                      'initial_accuracy_range':[min(accuracies),max(accuracies)],
                                      'mean_initial_pairwise_reward_disagreement':float(np.mean(pair_disagreements)) if pair_disagreements else None,
                                      'paired_vs_ppo':comparisons,'same_prompt_groups_verified':True}
    summary['completed_jobs']=len(results['runs'])
    summary['runtime_hours']=(max((j.get('end',j['start']) for j in state['runs']),default=state['start'])-state['start'])/3600
    summary['uncertainty_note']='One training seed. Task intervals condition on the realized models and decoded outputs; they do not estimate training-seed or inference-runtime variance.'
    summary['gate_note']='Gated fractions use sequence-normalized weight among nonzero advantages. They are not fractions of parameter-gradient norm. CISPO caps weights without setting its gate to zero.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with gzip.open(out/'results.json.gz','wt') as stream:json.dump(results,stream,separators=(',',':'))
    for name,rows in [('endpoints.csv',endpoint_rows),('gate-audit.csv',gate_rows)]:
        if rows:
            with (out/name).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'completed_jobs':len(results['runs']),'output':str(out),'state':state['state']}))


if __name__=='__main__':main()
