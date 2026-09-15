"""Audit completed checkpoints on CPU, optionally while the GPU queue runs."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/tmp/rl-surrogate-2026'))
    parser.add_argument('--container',default='rl-surrogates-2026')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        state=json.loads((args.root/'study-status.json').read_text())
        pending=[r for r in state['runs'] if r['state']=='complete' and
                 not all((args.root/'runs'/r['name']/n).exists() for n in ('checkpoint-audit.json','logprob-tail-audit.json'))]
        for run in pending:
            name=run['name']
            iteration=run.get('rollouts',state['plan']['rollouts'])
            jobs=[('checkpoint-audit.json',['/experiment/audit_checkpoint.py',
                   '--checkpoint',f'/artifacts/runs/{name}/checkpoints/iter_{iteration:07d}',
                   '--output',f'/artifacts/runs/{name}/checkpoint-audit.json']),
                  ('logprob-tail-audit.json',['/experiment/audit_logprobs.py',
                   '--run-dir',f'/artifacts/runs/{name}'])]
            for artifact,arguments in jobs:
                if (args.root/'runs'/name/artifact).exists():continue
                command=['docker','exec','-e','CUDA_VISIBLE_DEVICES=',
                         '-e','OMP_NUM_THREADS=2',args.container,'python',*arguments]
                result=subprocess.run(command,capture_output=True,text=True,timeout=180)
                print(json.dumps({'run':name,'audit':artifact,'returncode':result.returncode,
                                  'stdout':result.stdout,'stderr':result.stderr}),flush=True)
                if result.returncode and (not args.watch or state.get('outcome')):
                    raise RuntimeError(f'Audit failed: {name}/{artifact}')
        if not args.watch or state.get('outcome'):
            return
        time.sleep(30)


if __name__=='__main__':main()
