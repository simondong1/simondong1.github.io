"""Prepare public inputs and a container exposing exactly two allocated GPU UUIDs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from huggingface_hub import snapshot_download
from prepare_data import prepare

MILES_REV='2ef603a8ca44410beaad8c1dda5bf85f2a9d2533'
MODEL_REV='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
IMAGE='radixark/miles@sha256:16dc2819ed890455d54102fb88f3cdffdd210104b1707583051582af46bb6c5b'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/tmp/rl-surrogate-2026'))
    p.add_argument('--gpus',default=os.environ.get('CUDA_VISIBLE_DEVICES',''))
    p.add_argument('--container',default='rl-surrogates-2026')
    args=p.parse_args()
    uuids=args.gpus.split(',')
    if len(set(uuids))!=2 or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]+',u) for u in uuids):
        p.error('--gpus must contain exactly two allocated GPU UUIDs separated by a comma')
    present=subprocess.run(['docker','container','inspect',args.container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if present.returncode==0:
        p.error(f'Container {args.container} already exists; refusing to replace it')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    miles=root/'miles'
    if not miles.exists():
        subprocess.run(['git','clone','https://github.com/radixark/miles.git',str(miles)],check=True)
        subprocess.run(['git','-C',str(miles),'checkout',MILES_REV],check=True)
    head=subprocess.check_output(['git','-C',str(miles),'rev-parse','HEAD'],text=True).strip()
    assert head==MILES_REV, 'Existing Miles checkout has a different revision'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    snapshot_download('Qwen/Qwen3.5-9B',revision=MODEL_REV,local_dir=root/'model',token=False)
    prepare(root/'data')
    code=Path(__file__).resolve().parent
    expected=json.loads((code/'data-manifest.json').read_text())
    actual=json.loads((root/'data/manifest.json').read_text())
    assert expected==actual, 'Data manifest differs from the published study'
    subprocess.run(['docker','pull',IMAGE],check=True)
    subprocess.run(['docker','run','-d','--name',args.container,'--runtime=nvidia',
                    '--gpus','"device='+','.join(uuids)+'"','--shm-size=64g','--ipc=private',
                    '-e','HF_HUB_DISABLE_IMPLICIT_TOKEN=1','-e','WANDB_MODE=disabled',
                    '-v',f'{miles}:/work/miles','-v',f'{root}:/artifacts',
                    '-v',f'{root}/model:/model:ro','-v',f'{code}:/experiment',
                    '-w','/work/miles',IMAGE,'sleep','infinity'],check=True)
    visible=subprocess.check_output(['docker','exec',args.container,'nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True)
    assert set(visible.split())==set(uuids), 'GPU visibility differs from the allocation'
    for script in ['test_surrogates.py','test_native.py']:
        subprocess.run(['docker','exec','-e','PYTHONPATH=/experiment:/work/miles',args.container,'python','/experiment/'+script],check=True)
    print('Pinned inputs, two-GPU container, and analytic checks are ready.')


if __name__=='__main__':main()
