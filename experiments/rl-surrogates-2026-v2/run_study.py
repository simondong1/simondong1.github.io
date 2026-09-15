"""Bounded sequential queue, telemetry and fail-fast handling for two allocated GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

FILES = ['launch.py','surrogates.py','rewards.py','metrics.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/tmp/rl-surrogate-2026-v2'))
    p.add_argument('--container',default='rl-surrogates-2026-v2');a=p.parse_args()
    plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parent
    hashes={n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in FILES}
    path=a.root/'study-status.json';assert not path.exists(), 'Use a new state file for a new queue'
    state={'start':time.time(),'plan':plan,'code_sha256':hashes,'runs':[],'state':'running'}
    deadline=time.monotonic()+plan.get('wall_budget_seconds',28800)
    def write():
        state['updated']=time.time();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    write()
    try:
        for run in plan['runs']:
            assert all(hashlib.sha256((code/n).read_bytes()).hexdigest()==h for n,h in hashes.items()), 'Source changed during queue'
            if time.monotonic()>=deadline:
                state['state']='deadline';break
            directory=a.root/'runs'/run['name'];directory.mkdir(exist_ok=True)
            assert not (directory/'console.log').exists(), 'Never overwrite an existing run'
            subprocess.run(['docker','restart','-t','1',a.container],check=True,timeout=60)
            settings={**plan['common'],**plan['datasets'][run['dataset']],**run}
            cmd=['docker','exec',a.container,'python','/experiment/launch.py']
            for k,v in settings.items():
                if isinstance(v,bool):
                    if v:cmd.append('--'+k.replace('_','-'))
                else:cmd.extend(['--'+k.replace('_','-'),str(v)])
            entry={**run,'start':time.time(),'state':'running','command':cmd};state['runs'].append(entry);write()
            print(json.dumps({'event':'start',**run}),flush=True)
            with (directory/'console.log').open('w') as log, (directory/'gpu.jsonl').open('w') as gpu:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if time.monotonic()>=deadline:
                        subprocess.run(['docker','stop','-t','10',a.container],check=True,timeout=60)
                        proc.wait(timeout=30);entry['state']='deadline';break
                    try:
                        output=subprocess.check_output(['nvidia-smi','--id=3,7',
                            '--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw',
                            '--format=csv,noheader,nounits'],text=True,timeout=10)
                        gpu.write(json.dumps({'time':time.time(),'rows':output.strip().splitlines()})+'\n');gpu.flush()
                    except subprocess.SubprocessError as e:gpu.write(json.dumps({'error':str(e)})+'\n')
                    time.sleep(10)
            entry.update(end=time.time(),returncode=proc.returncode)
            if entry['state']!='deadline':entry['state']='complete' if proc.returncode==0 else 'failed'
            write();print(json.dumps({'event':entry['state'],**run,'seconds':entry['end']-entry['start']}),flush=True)
            if entry['state']!='complete':state['state']=entry['state'];break
        else:state['state']='complete'
    except Exception as e:
        state.update(state='controller_error',error=repr(e));raise
    finally:
        # The container belongs solely to this experiment; release its GPUs.
        subprocess.run(['docker','stop','-t','10',a.container],timeout=60,check=False)
        state['end']=time.time();write()


if __name__=='__main__':main()
