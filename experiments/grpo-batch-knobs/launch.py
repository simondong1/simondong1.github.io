"""Portable launcher for the published experiment; no cluster provisioning.

Supply an existing compatible Miles/Megatron environment and two allocated
GPUs. --dry-run checks the exact batch invariant and prints the Miles argv.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from training_arguments import ROOT, arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--miles', type=Path, default=Path(os.environ.get('MILES_PATH', '../miles')))
    parser.add_argument('--megatron', type=Path, default=Path(os.environ.get('MEGATRON_PATH', '../Megatron-LM')))
    parser.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS', 'auto'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('load') == 'initial_checkpoint':
        from training_arguments import INITIAL
        config['load'] = str(INITIAL)
    elif config.get('load', '').startswith('checkpoints/'):
        config['load'] = str(ROOT / config['load'])
    flags = arguments(config)
    if args.dry_run:
        print(json.dumps(flags, indent=2))
        return

    code = Path(__file__).resolve().parent
    miles, megatron = args.miles.resolve(), args.megatron.resolve()
    environment = {
        'PYTHONPATH': os.pathsep.join(map(str, [megatron, miles, code])),
        'WANDB_BASE_URL': os.environ.get('WANDB_BASE_URL', 'https://api.wandb.ai'),
        'WANDB_MODE': os.environ.get('WANDB_MODE', 'offline'),
        'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_CUMEM_ENABLE': '0',
        'NCCL_NVLS_ENABLE': '0', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'PYTHONUNBUFFERED': '1',
    }
    directory = ROOT / 'results' / config['name']
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'status.json').exists():
        raise RuntimeError('Use a new run name; existing results are never overwritten')
    environment['STUDY_RUN_DIR'] = str(directory)
    os.environ.update(environment)
    sys.path[:0] = list(map(str, [megatron, miles, code]))
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    (directory / 'arguments.json').write_text(json.dumps(flags, indent=2))
    (directory / 'source_hashes.json').write_text(json.dumps({
        name: hashlib.sha256((code / name).read_bytes()).hexdigest()
        for name in ['launch.py', 'training_arguments.py', 'study_config.py', 'study_rewards.py', 'study_hooks.py']
    }, indent=2))
    for name in ['data_manifest.json', 'model_manifest.json']:
        shutil.copyfile(code / name, directory / name)
    observed = {}
    for package in ['torch', 'ray', 'sglang', 'transformers', 'flash-attn', 'wandb', 'megatron-core']:
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed[package] = 'unavailable'
    for name, path in [('miles_git', miles), ('Megatron-LM_git', megatron)]:
        observed[name] = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    (directory / 'environment.json').write_text(json.dumps(observed, indent=2))
    status = {'state': 'starting', 'name': config['name'], 'started': time.time()}
    (directory / 'status.json').write_text(json.dumps(status, indent=2))
    import ray
    from study_hooks import setup_worker
    try:
        ray.init(address=args.ray_address, namespace='miles-four-knob-reproduction',
                 runtime_env={'env_vars': environment, 'working_dir': str(code),
                              'excludes': ['data', 'results', 'checkpoints', 'model', 'initial_checkpoint', '.cache', 'wandb'],
                              'worker_process_setup_hook': 'study_hooks.setup_worker'})
        setup_worker()
        os.chdir(miles)
        spec = importlib.util.spec_from_file_location('study_train', miles / 'train.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = ['train.py', *flags]
        parsed = module.parse_args()
        status['state'] = 'running'
        (directory / 'status.json').write_text(json.dumps(status, indent=2))
        asyncio.run(module.train(parsed))
        status['state'] = 'finished'
    except BaseException as error:
        status.update(state='failed', error=type(error).__name__)
        raise
    finally:
        from miles.utils.tracking_utils import finish_tracking
        finish_tracking()
        ray.shutdown()
        status.update(finished=time.time(), job_seconds=time.time() - status['started'])
        (directory / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
