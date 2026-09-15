"""Backfill and tail Miles logs without changing or restarting training.

Dependencies: wandb==0.30.0, tensorboard==2.20.0. Credentials come from the
environment or the selected server's netrc entry; they are never serialized.
The uploader runs on the host and does not initialize CUDA. All eight GPUs'
automatic W&B telemetry is disabled; only the experiment's saved GPU data
is uploaded. Raw records accompany the derived scalar history.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import netrc
import os
from pathlib import Path
import statistics
import tarfile
import time
from urllib.parse import urlparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def credential_available(base_url):
    if os.environ.get('WANDB_API_KEY'):
        return True
    try:
        auth = netrc.netrc().authenticators(urlparse(base_url).netloc)
        return bool(auth and auth[2])
    except (OSError, netrc.NetrcParseError):
        return False


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def jsonl_records(path, offset):
    """Do not consume a partly written final line."""
    if not path.exists():
        return
    with path.open('rb') as stream:
        stream.seek(offset)
        while line := stream.readline():
            if not line.endswith(b'\n'):
                break
            yield stream.tell(), json.loads(line)


class Collector:
    def __init__(self, directory, config, started, cursor):
        self.directory, self.config, self.started = directory, config, started
        self.cursor = cursor
        self.accumulators = {}
        self.updates_per_collection = (config['prompts_per_rollout'] *
                                       config['group_size'] // config['global_batch_size'])

    def pending(self):
        """Return chronological records and the cursors each record commits.

        Each TensorBoard file/tag has its own cursor. Delayed writes and
        different writers at the same step must not cause skipped metrics.
        W&B's transport step is independent of the original metric axes.
        """
        records = []
        groups = {}
        for path in sorted((self.directory / 'tensorboard').glob('events.out.tfevents.*')):
            relative = str(path.relative_to(self.directory))
            if relative not in self.accumulators:
                self.accumulators[relative] = EventAccumulator(
                    str(path), size_guidance={'scalars': 0}, purge_orphaned_data=False)
            acc = self.accumulators[relative]
            acc.Reload()
            for tag in acc.Tags()['scalars']:
                key = 'tb:' + relative + ':' + tag
                events = acc.Scalars(tag)
                start = self.cursor.get(key, 0)
                if start > len(events):
                    raise RuntimeError('A TensorBoard source was truncated: ' + relative)
                for index, event in enumerate(events[start:], start):
                    namespace = tag.split('/', 1)[0]
                    group = groups.setdefault((relative, namespace, event.step), {
                        'time': event.wall_time, 'data': {}, 'marks': {}})
                    group['time'] = max(group['time'], event.wall_time)
                    # Keep native names and native collection/evaluation steps.
                    # Only train's zero-based update index becomes completed updates.
                    axis = ('axis/optimizer_updates' if namespace == 'train' else
                            'axis/native_eval_step' if namespace == 'eval' else
                            'axis/native_rollout_step')
                    group['data'].update({axis: event.step + (namespace == 'train'),
                                          tag: event.value})
                    group['marks'][key] = index + 1
        records.extend(groups.values())
        for filename in ['measurements/rollouts.jsonl', 'measurements/eval.jsonl', 'gpu.jsonl']:
            key = 'jsonl:' + filename
            for offset, item in jsonl_records(self.directory / filename, self.cursor.get(key, 0)):
                data = {}
                if filename.endswith('rollouts.jsonl'):
                    # The collection is recorded BEFORE it is trained. Its
                    # optimizer_updates field is a target, not completed work.
                    data['axis/sampling_policy_updates'] = item['rollout_id'] * self.updates_per_collection
                    for name, value in item.items():
                        if numeric(value) and name not in ['time', 'optimizer_updates']:
                            data['sampling/' + name] = value
                    for name, value in item.get('length_quantiles', {}).items():
                        data['sampling/response_length_' + name] = value
                    samples = item.get('samples', [])
                    if samples:
                        data['sampling/response_length_mean'] = statistics.mean(s['response_length'] for s in samples)
                    data['sampling/generated_tokens_per_second'] = item['generated_tokens'] / item['rollout_seconds']
                elif filename.endswith('eval.jsonl'):
                    rewards = item['rewards']
                    data.update({'axis/optimizer_updates': item['optimizer_updates'],
                                 'evaluation/accuracy': statistics.mean(rewards),
                                 'evaluation/num_questions': len(rewards),
                                 'evaluation/completed_collections': item['completed_collections']})
                    samples = item.get('samples', [])
                    if samples:
                        data['evaluation/response_length_mean'] = statistics.mean(s['response_length'] for s in samples)
                        data['evaluation/truncation_fraction'] = statistics.mean(s['status'] == 'TRUNCATED' for s in samples)
                else:
                    for row in item.get('rows', []):
                        index, uuid, memory, utilization, power = [part.strip() for part in row.split(',')]
                        if index not in ['3', '7']:
                            raise ValueError('Unexpected GPU in study telemetry: ' + index)
                        prefix = 'hardware/gpu_' + index + '/'
                        data.update({prefix + 'memory_gib': float(memory) / 1024,
                                     prefix + 'utilization_pct': float(utilization),
                                     prefix + 'power_watts': float(power)})
                    if 'error' in item:
                        data['hardware/telemetry_error'] = 1
                if 'time' not in item:
                    data['source/timestamp_missing'] = 1
                records.append({'time': item.get('time', self.started), 'data': data, 'marks': {key: offset}})
        for record in records:
            record['data']['axis/wall_seconds'] = max(0, record['time'] - self.started)
            record['data']['source/timestamp'] = record['time']
        return sorted(records, key=lambda record: record['time'])

    def commit(self, record):
        for key, value in record['marks'].items():
            self.cursor[key] = max(self.cursor.get(key, 0), value)


def define_axes(run):
    axes = {'train/*': 'axis/optimizer_updates', 'evaluation/*': 'axis/optimizer_updates',
            'sampling/*': 'axis/sampling_policy_updates', 'rollout/*': 'axis/native_rollout_step',
            'perf/*': 'axis/native_rollout_step', 'eval/*': 'axis/native_eval_step',
            'hardware/*': 'axis/wall_seconds'}
    for axis in set(axes.values()):
        run.define_metric(axis, hidden=True)
    for pattern, axis in axes.items():
        run.define_metric(pattern, step_metric=axis, step_sync=False)
    run.define_metric('source/*', hidden=True)
    run.define_metric('evaluation/accuracy', step_metric='axis/optimizer_updates',
                      step_sync=False, summary='last,max')


def recover_journal(path):
    """Recover source cursors and replayable history after an uploader restart."""
    cursor, records, offset = {}, [], 0
    for offset, record in jsonl_records(path, 0):
        records.append(record)
        for key, value in record['marks'].items():
            cursor[key] = max(cursor.get(key, 0), value)
    if path.exists() and path.stat().st_size != offset:
        # Only our own incomplete final journal write is removed, never source logs.
        with path.open('r+b') as stream:
            stream.truncate(offset)
    return cursor, records


def all_jobs(state, root, include_pilots):
    jobs = list(state['runs'])
    if include_pilots:
        for path in sorted((root / 'runs').glob('pilot_*')):
            cfg = read_json(path / 'config.json')
            if cfg is None:
                continue
            events = list(path.glob('tensorboard/events.out.tfevents.*'))
            started = min(float(p.name.split('.')[3]) for p in events)
            jobs.append({'name': path.name, 'dataset': cfg['dataset'], 'algorithm': cfg['algorithm'],
                         'seed': cfg['seed'], 'start': started, 'end': (path / 'console.log').stat().st_mtime,
                         'state': 'complete', 'pilot': True,
                         'valid_learning_result': path.name != 'pilot_gsm_packed8192'})
    return jobs


def upload_records(run, directory, root, sync_root, terminal_state):
    """Upload compact raw logs, not multi-gigabyte model/optimizer tensors."""
    import wandb
    archive = sync_root / (directory.name + '-records.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for relative in ['config.json', 'command.json', 'console.log', 'gpu.jsonl',
                         'measurements', 'tensorboard']:
            path = directory / relative
            if path.exists():
                tar.add(path, arcname=relative)
        for relative in ['study-plan.json', 'study-status.json', 'data/data-manifest.json',
                         'data/manifest.json', 'data/dapo_conflicts.json']:
            path = root / relative
            if path.exists():
                tar.add(path, arcname='study/' + relative)
        for path in Path(__file__).parent.iterdir():
            if path.suffix in ['.py', '.json', '.md']:
                tar.add(path, arcname='source/' + path.name)
    artifact = wandb.Artifact(run.id + '-records', type='rl-metrics',
                              metadata={'training_state': terminal_state,
                                        'checkpoints_included': False,
                                        'raw_per_response_records_included': True})
    artifact.add_file(str(archive))
    run.log_artifact(artifact)


def synchronize(args):
    import wandb
    root = args.root
    sync_root = root / 'wandb-sync'
    sync_root.mkdir(exist_ok=True)
    lock = (sync_root / 'uploader.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / 'wandb-status.json'
    status = read_json(status_path, {'runs': {}})
    announced = None
    active, active_name, collector, local, journal = None, None, None, None, None
    while True:
        settings = read_json(args.settings, {}) if args.settings else {}
        base_url = settings.get('base_url', args.base_url)
        project = settings.get('project', args.project)
        entity = settings.get('entity', args.entity)
        if not credential_available(base_url):
            status.update(state='awaiting_credentials', updated=time.time(), base_url=base_url,
                          project=project, entity=entity)
            atomic_json(status_path, status)
            if announced != 'credentials':
                print('Awaiting a WANDB_API_KEY or netrc entry for ' + base_url, flush=True)
                announced = 'credentials'
            time.sleep(args.poll_seconds)
            continue
        os.environ.update(WANDB_BASE_URL=base_url, WANDB_MODE='online', WANDB_SILENT='true')
        state = read_json(root / 'study-status.json')
        group = 'math-surrogates-' + dt.datetime.fromtimestamp(state['start'], dt.timezone.utc).strftime('%Y%m%d-%H%M')
        jobs = all_jobs(state, root, args.include_pilots)
        remaining = [job for job in jobs if not status['runs'].get(job['name'], {}).get('uploaded_complete')]
        if not remaining:
            status.update(state='complete' if state['state'] != 'running' else 'waiting_for_next_job', updated=time.time())
            atomic_json(status_path, status)
            if state['state'] != 'running':
                return
            time.sleep(args.poll_seconds)
            continue
        # Keep a live run open; terminal jobs are drained first at startup.
        job = next((job for job in remaining if job['name'] == active_name),
                   sorted(remaining, key=lambda job: (job['state'] == 'running', job['start']))[0])
        directory = root / 'runs' / job['name']
        config = read_json(directory / 'config.json')
        if config is None:
            time.sleep(args.poll_seconds)
            continue
        local_path = sync_root / (job['name'] + '.json')
        if active is None:
            journal_path = sync_root / (job['name'] + '-journal.jsonl')
            cursor, previous = recover_journal(journal_path)
            local = {'cursor': cursor, 'history_rows': len(previous)}
            run_id = hashlib.sha256((group + ':' + job['name']).encode()).hexdigest()[:16]
            meta = {**config, 'model': state['plan']['model'], 'full_parameter_training': True,
                    'trainable_parameters': 8953803264, 'advantage_estimator': 'grpo',
                    'surrogate': job['algorithm'], 'actor_critic_ppo': False,
                    'study_group': group, 'pilot': job.get('pilot', False),
                    'valid_learning_result': job.get('valid_learning_result', True),
                    'training_started_utc': dt.datetime.fromtimestamp(job['start'], dt.timezone.utc).isoformat(),
                    'code_sha256': state['code_sha256'] if not job.get('pilot') else {},
                    'data_manifest': read_json(Path(__file__).parent / 'data-manifest.json'),
                    'physical_gpu_ids': [3, 7], 'training_log_source': 'Miles TensorBoard + JSONL',
                    'sampling_axis': 'policy updates BEFORE collecting the trajectories',
                    'native_eval_axis': 'Miles rollout_id (0 is baseline; 3 means after collection 4)',
                    'checkpoint_upload': False}
            tags = ['surrogate-study-v2', job['dataset'], job['algorithm'], 'full-parameter',
                    'pilot' if job.get('pilot') else 'main-comparison']
            if not job.get('valid_learning_result', True):
                tags.append('excluded-grader-bug')
            active = wandb.init(entity=entity, project=project, id=run_id, resume='allow',
                                name=job['name'], group=group, job_type='pilot' if job.get('pilot') else 'train',
                                tags=tags, config=meta, dir=str(sync_root),
                                settings=wandb.Settings(x_disable_stats=True, console='off', quiet=True,
                                                        disable_code=True, disable_git=True, init_timeout=30))
            active_name = job['name']
            define_axes(active)
            # W&B reports its next transport step when resuming. Replay any
            # journaled records that had not reached the server before a failure.
            next_step = active.step
            if next_step > len(previous):
                raise RuntimeError('Remote history exceeds the local upload journal')
            for index in range(next_step, len(previous)):
                active.log(previous[index]['data'], step=index, commit=True)
            journal = journal_path.open('a')
            collector = Collector(directory, config, job['start'], local['cursor'])
            status['runs'][job['name']] = {'url': active.url, 'run_id': run_id, 'uploaded_complete': False}
            print(json.dumps({'event': 'connected', 'job': job['name'], 'url': active.url}), flush=True)
        for record in collector.pending():
            journal.write(json.dumps(record) + '\n')
            journal.flush()
            active.log(record['data'], step=local['history_rows'], commit=True)
            collector.commit(record)
            local['history_rows'] += 1
        active.summary.update({'training_state': job['state'], 'history_rows_imported': local['history_rows'],
                               'training_elapsed_seconds': job.get('end', time.time()) - job['start']})
        local['last_poll'] = time.time()
        atomic_json(local_path, local)
        status.update(state='syncing', updated=time.time(), base_url=base_url, entity=active.entity,
                      project=project, group=group)
        status['runs'][job['name']].update(training_state=job['state'], history_rows=local['history_rows'],
                                          last_poll=time.time())
        atomic_json(status_path, status)
        if job['state'] != 'running':
            upload_records(active, directory, root, sync_root, job['state'])
            active.finish(exit_code=0 if job['state'] == 'complete' else 1)
            status['runs'][job['name']]['uploaded_complete'] = True
            atomic_json(status_path, status)
            print(json.dumps({'event': 'uploaded_complete', 'job': job['name']}), flush=True)
            journal.close()
            active, active_name, collector, local, journal = None, None, None, None, None
        else:
            time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/rl-surrogate-2026-v2'))
    parser.add_argument('--base-url', default=os.environ.get('WANDB_BASE_URL', 'https://api.wandb.ai'))
    parser.add_argument('--project', default='rl-surrogates-2026')
    parser.add_argument('--entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--settings', type=Path, help='Optional JSON with base_url, project, entity; no secrets.')
    parser.add_argument('--poll-seconds', type=float, default=20)
    parser.add_argument('--include-pilots', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            synchronize(args)
            return
        except BlockingIOError:
            raise SystemExit('An uploader is already running for this study')
        except Exception as error:
            # Authentication/network failures affect only the uploader. Preserve
            # its journal and retry; the independent training queue keeps running.
            import wandb
            status_path = args.root / 'wandb-status.json'
            status = read_json(status_path, {'runs': {}})
            status.update(state='retrying', updated=time.time(), error_type=type(error).__name__)
            atomic_json(status_path, status)
            print(json.dumps({'event': 'retrying', 'error_type': type(error).__name__}), flush=True)
            wandb.finish(exit_code=1)
            wandb.teardown()
            time.sleep(max(30, args.poll_seconds))


if __name__ == '__main__':
    main()
