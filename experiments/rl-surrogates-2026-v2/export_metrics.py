"""Export all TensorBoard scalars to inspectable JSON without losing step axes."""
import argparse
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export(root):
    for run in sorted((root/'runs').iterdir()):
        tb=run/'tensorboard'
        if not tb.exists():continue
        events=list(tb.rglob('events.out.tfevents.*'))
        metrics={}
        for event in events:
            acc=EventAccumulator(str(event),size_guidance={'scalars':0});acc.Reload()
            for tag in acc.Tags()['scalars']:
                metrics.setdefault(tag,[]).extend({'step':r.step,'time':r.wall_time,'value':r.value} for r in acc.Scalars(tag))
        for rows in metrics.values():rows.sort(key=lambda x:(x['step'],x['time']))
        (run/'scalars.json').write_text(json.dumps(metrics))
        print(run.name,len(metrics),'scalar series')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);export(p.parse_args().root)
