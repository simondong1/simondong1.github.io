"""Convert local Miles measurements into compact, auditable public results."""
import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import re
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def parse_run(directory, expected_steps=None):
    config = json.loads((directory/'config.json').read_text())
    text = re.sub(r'\x1b\[[0-9;]*m', '', (directory/'console.log').read_text())
    steps = []
    perf = []
    for line in text.splitlines():
        if re.search(r'log_utils\.py:\d+ - step \d+: ', line):
            steps.append(ast.literal_eval(line[line.index('{'):]))
        if 'train_metric_utils.py:' in line and ' - perf ' in line:
            perf.append(ast.literal_eval(line[line.index('{'):]))
    if expected_steps is not None:
        assert len(steps)==expected_steps, (directory.name,len(steps),expected_steps)
        assert [x['train/step'] for x in steps]==list(range(expected_steps))
        selected=config['algorithm']
        assert all(abs(s['train/pg_clipfrac']-s[f'train/audit_{selected}_gated'])<1e-6 and
                   abs(s['train/gradient_weight_mean']-s[f'train/audit_{selected}_weight'])<1e-6
                   for s in steps), 'Recorded gate/weight does not match the declared surrogate'
    evals = read_jsonl(directory/'measurements/eval.jsonl')
    rollout = read_jsonl(directory/'measurements/rollouts.jsonl')
    per_collection = config.get('prompts_per_rollout',8)*4//8
    evaluation = []
    for index,event in enumerate(evals):
        assert all(x is not None for x in event['rewards']), 'Evaluation errors must be investigated'
        samples = event['samples']
        assert all(s['status'] in ('COMPLETED','TRUNCATED') for s in samples), 'Unfinished evaluation response'
        assert event['rewards'] == [s['reward'] for s in samples], 'Evaluation reward records disagree'
        finished = [s for s in samples if s['status'] != 'TRUNCATED']
        row = {
            'updates': 0 if index==0 else (event['rollout_id']+1)*per_collection,
            'accuracy': statistics.mean(event['rewards']),
            'n': len(samples),
            'mean_response_tokens': statistics.mean(s['response_length'] for s in samples),
            'truncation_fraction': statistics.mean(s['status']=='TRUNCATED' for s in samples),
            'accuracy_given_not_truncated': statistics.mean(s['reward'] for s in finished) if finished else None,
            'examples':[{k:s[k] for k in ('uid','reward','response_length','status')} for s in samples],
        }
        assert len({s['uid'] for s in samples})==len(samples), 'Duplicate eval task identifiers'
        evaluation.append(row)
    if expected_steps is not None:
        expected_evals = [0] + list(range(config['eval_interval'] * per_collection,
                                          expected_steps + 1, config['eval_interval'] * per_collection))
        assert [e['updates'] for e in evaluation] == expected_evals, 'Missing or misaligned evaluation'
        assert all(e['n'] == 256 for e in evaluation), 'Incomplete held-out evaluation'
        ids = {s['uid'] for s in evaluation[0]['examples']}
        assert all({s['uid'] for s in e['examples']} == ids for e in evaluation), 'Evaluation tasks changed'
        assert [r['rollout_id'] for r in rollout] == list(range(config['rollouts']))
        prompt_counts = Counter()
        for collection in rollout:
            counts = Counter(s['uid'] for s in collection['samples'])
            assert len(counts) == config['prompts_per_rollout']
            assert set(counts.values()) == {4}, 'A prompt group is incomplete'
            assert all(s['reward'] in (0, 1) for s in collection['samples'])
            assert all(s['status'] in ('COMPLETED','TRUNCATED') for s in collection['samples']), 'Unfinished training response'
            prompt_counts.update(counts.keys())
        complete_passes,extra=divmod(config['rollouts']*config['prompts_per_rollout'],512)
        expected_counts=Counter({complete_passes:512-extra})
        if extra:expected_counts[complete_passes+1]=extra
        assert len(prompt_counts) == 512 and Counter(prompt_counts.values()) == expected_counts, 'Training tasks/passes differ'
        assert not ids.intersection(prompt_counts), 'Training/test task overlap'
        guard_max=max(s['train/ratio_log_clamp_fraction'] for s in steps)
        assert guard_max == 0 or config['algorithm']=='glm5', 'Ratio guard activated for a surrogate whose selected gradient needs inspection'
    advantage_total = sum(s['train/nonzero_adv_fraction'] for s in steps)
    gate_audit = {
        name: {
            'conditional_gated_fraction': sum(s[f'train/audit_{name}_gated'] for s in steps)/advantage_total if advantage_total else None,
            'conditional_mean_gradient_weight': sum(s[f'train/audit_{name}_weight'] for s in steps)/advantage_total if advantage_total else None,
        } for name in ('ppo','dapo','cispo','glm5','sapo','dppo')
    }
    out = {
        'name':directory.name,'config':config,'evaluation':evaluation,
        'steps':steps,'gate_audit':gate_audit,
        'rollout':[{'id':d['rollout_id'], 'seconds':d['rollout_seconds'],
                    'reward':statistics.mean(s['reward'] for s in d['samples']),
                    'mean_response_tokens':statistics.mean(s['response_length'] for s in d['samples']),
                    'truncation_fraction':statistics.mean(s['status']=='TRUNCATED' for s in d['samples']),
                    'zero_variance_group_fraction':d['zero_variance_group_fraction'],
                    'prompt_uids':list(dict.fromkeys(s['uid'] for s in d['samples']))}
                   for d in rollout],
        'train_seconds':sum(x['perf/train_time'] for x in perf),
        'generation_seconds':sum(x['rollout_seconds'] for x in rollout),
        'training_responses':sum(len(x['samples']) for x in rollout),
        'training_response_tokens':sum(s['response_length'] for x in rollout for s in x['samples']),
        'zero_gradient_step_fraction':statistics.mean(s['train/grad_norm']==0 for s in steps) if steps else None,
        'ratio_guard':{
            'log_ratio_bounds':[-20,20],
            'mean_sequence_weighted_fraction':statistics.mean(s['train/ratio_log_clamp_fraction'] for s in steps) if steps else None,
            'max_step_sequence_weighted_fraction':max((s['train/ratio_log_clamp_fraction'] for s in steps),default=None),
            'affected_steps':sum(s['train/ratio_log_clamp_fraction']>0 for s in steps),
            'glm_gradient_note':'Both exp(-20) and exp(20) are outside the GLM rejection interval; guarded tokens already have zero selected GLM coefficient.',
            'diagnostic_note':'Counterfactual weight diagnostics use guarded ratios and omit the derivative saturation of this numerical clamp. The algorithm-gate fraction is separate from the guard fraction.',
        },
        'max_step_mean_engine_logprob_abs_diff':max((s['train/initial_engine_logprob_abs_diff'] for s in steps),default=None),
        'mean_engine_logprob_abs_diff':statistics.mean(s['train/initial_engine_logprob_abs_diff'] for s in steps) if steps else None,
    }
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/tmp/rl-surrogate-2026'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    state=json.loads((args.root/'study-status.json').read_text())
    runs=[]
    monitor = []
    if (args.root/'gpu-monitor.csv').exists():
        with (args.root/'gpu-monitor.csv').open() as f:
            monitor = list(csv.DictReader(f))
    for r in state['runs']:
        if r['state']!='complete':continue
        expected=r.get('rollouts',state['plan']['rollouts'])*state['plan']['prompts_per_rollout']*4//8
        result=parse_run(args.root/'runs'/r['name'],expected)
        result['wall_seconds']=r['end']-r['start']
        # Timestamp filtering excludes the aborted attempt that reused the first run name.
        memory = [x for x in monitor if r['start'] <= float(x['time']) <= r['end']]
        result['sampled_peak_gpu_memory_mib'] = {
            uuid:max(int(x['used_mib']) for x in memory if x['uuid']==uuid)
            for uuid in sorted({x['uuid'] for x in memory})
        }
        audit_path=args.root/'runs'/r['name']/'checkpoint-audit.json'
        if audit_path.exists():result['checkpoint_audit']=json.loads(audit_path.read_text())
        tail_path=args.root/'runs'/r['name']/'logprob-tail-audit.json'
        if tail_path.exists():
            result['logprob_tail_audit']=json.loads(tail_path.read_text())
            assert result['logprob_tail_audit']['tokens']==result['training_response_tokens']
            assert abs(result['logprob_tail_audit']['sequence_mean_absdiff']-result['mean_engine_logprob_abs_diff'])<1e-6
        runs.append(result)
    # Matched prompt order and identical non-algorithm controls within each task/seed.
    for run in runs:
        matches=[r for r in runs if (r['config']['dataset'],r['config']['seed'])==(run['config']['dataset'],run['config']['seed'])]
        for other in matches:
            one={k:v for k,v in run['config'].items() if k not in ('name','algorithm')}
            two={k:v for k,v in other['config'].items() if k not in ('name','algorithm')}
            assert one==two, 'Non-algorithm config differs'
            assert [x['prompt_uids'] for x in run['rollout']]==[x['prompt_uids'] for x in other['rollout']], 'Prompt order differs'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'study':state,'runs':runs},indent=2))
    print(f'Validated and exported {len(runs)} completed runs to {args.output}')


if __name__=='__main__':main()
