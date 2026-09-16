"""Read-only queue health snapshots. This process never changes training."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import time


def last_record(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        position, data = stream.tell(), b''
        while position and data.count(b'\n') < 2:
            size = min(position, 1024 * 1024)
            position -= size
            stream.seek(position)
            data = stream.read(size) + data
    lines = data.split(b'\n')
    return json.loads(lines[-2]) if len(lines) >= 2 else None


def snapshot(root):
    now = time.time()
    state = json.loads((root / 'study-status.json').read_text())
    upload = json.loads((root / 'wandb-status.json').read_text())
    active = next((r for r in state['runs'] if r['state'] == 'running'), None)
    result = {'utc': dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime('%H:%M:%S'),
              'state': state['state'], 'completed': sum(r['state'] == 'complete' for r in state['runs']),
              'total': len(state['plan']['runs']), 'wandb': upload['state'],
              'wandb_entity': upload['entity'], 'wandb_poll_age_s': round(now - upload['updated']),
              'elapsed_minutes': round((now - state['start']) / 60, 1)}
    if active:
        directory = root / 'runs' / active['name']
        config_path = directory / 'config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else None
        console = directory / 'console.log'
        with console.open('rb') as stream:
            stream.seek(max(0, console.stat().st_size - 1024 * 1024))
            tail = stream.read().decode(errors='replace')
        updates = [int(x) + 1 for x in re.findall(r'log_utils.py:\d+ - step (\d+):', tail)]
        result.update(active=active['name'], completed_updates=max(updates, default=0),
                      job_minutes=round((now - active['start']) / 60, 1),
                      log_age_s=round(now - console.stat().st_mtime))
        if config:
            result['target_updates'] = config['rollouts'] * config['prompts_per_rollout'] * config['group_size'] // config['global_batch_size']
        evaluation = last_record(directory / 'measurements/eval.jsonl')
        if evaluation:
            result.update(eval_updates=evaluation['optimizer_updates'],
                          eval_accuracy_pct=round(100 * sum(evaluation['rewards']) / len(evaluation['rewards']), 2))
        gpu = last_record(directory / 'gpu.jsonl')
        if gpu and 'rows' in gpu:
            result['gpu_util_pct'] = [float(row.split(',')[3]) for row in gpu['rows']]
        result['attention'] = result['log_age_s'] > 600 or result['wandb_poll_age_s'] > 120
    elif state['state'] not in ['running', 'complete']:
        result['attention'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/rl-surrogate-2026-v2'))
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=45)
    args = parser.parse_args()
    while True:
        result = snapshot(args.root)
        text = json.dumps(result)
        print(text, flush=True)
        (args.root / 'monitor-status.json').write_text(text + '\n')
        with (args.root / 'monitor-events.jsonl').open('a') as stream:
            stream.write(text + '\n')
        if not args.watch or (result['state'] != 'running' and result['wandb'] == 'complete'):
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
