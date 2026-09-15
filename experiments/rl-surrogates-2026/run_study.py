"""Run a matched study sequentially in the experiment's isolated two-GPU container.

Invoke on the host after the pilots. A JSON plan is required; its UTC deadline
bounds the whole queue. Restart only this named container between runs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/tmp/rl-surrogate-2026'))
    parser.add_argument('--container', default='rl-surrogates-2026')
    parser.add_argument('--resume', action='store_true', help='Resume a state containing only the completed plan prefix')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    code = Path(__file__).resolve().parent
    locked = {n: sha(code/n) for n in ['launch.py', 'surrogates.py', 'rewards.py', 'metrics.py']}
    state_path = args.root/'study-status.json'
    if args.resume:
        state=json.loads(state_path.read_text())
        assert all(r['state']=='complete' for r in state['runs']), 'Resolve and archive interrupted attempts before resuming'
        assert state['code_sha256']==locked, 'Training code changed before resume'
        assert [r['name'] for r in state['runs']]==[r['name'] for r in plan['runs'][:len(state['runs'])]]
        if state['plan']!=plan:
            state.setdefault('plan_history',[]).append(state['plan'])
        state['plan']=plan
        state.pop('outcome',None)
        state.pop('end',None)
    else:
        assert not state_path.exists(), 'Existing study state; use a new root or an explicit resume'
        state = {'start': time.time(), 'plan': plan, 'code_sha256': locked, 'runs': []}

    def persist():
        state['updated'] = time.time()
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(state_path)

    persist()
    for run in plan['runs'][len(state['runs']):]:
        assert all(sha(code/n) == h for n,h in locked.items()), 'Training code changed during study'
        if time.time() >= deadline:
            state['outcome'] = 'deadline_before_next_run'
            persist()
            break
        directory = args.root/'runs'/run['name']
        directory.mkdir(parents=True, exist_ok=True)
        assert not (directory/'console.log').exists(), f'Refusing to overwrite {directory}'
        subprocess.run(['docker','restart','-t','1',args.container], check=True, timeout=90)
        command = ['docker','exec',args.container,'python','/experiment/launch.py',
                   '--name',run['name'],'--algorithm',run['algorithm'],
                   '--dataset',run['dataset'],'--seed',str(run['seed']),
                   '--rollouts',str(run.get('rollouts',plan['rollouts'])),
                   '--prompts-per-rollout',str(plan['prompts_per_rollout']),
                   '--max-tokens',str(plan['max_tokens'][run['dataset']]),
                   '--eval-interval',str(plan['eval_interval'])]
        entry = {**run, 'start':time.time(),'command':command,'state':'running'}
        state['runs'].append(entry)
        persist()
        print(json.dumps({'event':'start',**run,'time':entry['start']}),flush=True)
        with (directory/'console.log').open('w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                code_return = proc.wait(timeout=max(1,deadline-time.time()))
            except subprocess.TimeoutExpired:
                subprocess.run(['docker','stop','-t','10',args.container],check=True,timeout=60)
                proc.wait(timeout=30)
                entry.update(state='deadline',end=time.time())
                state['outcome']='deadline_during_run'
                persist()
                return
        entry.update(state='complete' if code_return==0 else 'failed',returncode=code_return,end=time.time())
        persist()
        print(json.dumps({'event':entry['state'],**run,'seconds':entry['end']-entry['start']}),flush=True)
        if code_return != 0:
            state['outcome']='failed_run_requires_inspection'
            persist()
            return
    else:
        state['outcome']='complete'
        state['end']=time.time()
        persist()


if __name__=='__main__':
    main()
