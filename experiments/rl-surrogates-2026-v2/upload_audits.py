"""Attach completed-run audit summaries and artifacts to their existing W&B runs."""
import argparse
import json
import os
from pathlib import Path

import wandb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/tmp/rl-surrogate-2026-v2'))
    args=parser.parse_args();root=args.root
    config=json.loads((root/'wandb-sync-config.json').read_text())
    status=json.loads((root/'wandb-status.json').read_text())
    path=root/'wandb-audits-status.json'
    uploaded=json.loads(path.read_text()) if path.exists() else {}
    directory=root/'wandb-audit-uploads';directory.mkdir(exist_ok=True)
    os.environ.update(WANDB_BASE_URL=config['base_url'],WANDB_MODE='online',WANDB_SILENT='true',CUDA_VISIBLE_DEVICES='')
    for name,entry in status['runs'].items():
        if name.startswith('pilot_') or not entry.get('uploaded_complete') or name in uploaded:continue
        checkpoint=root/'audits'/f'{name}-checkpoint.json';measurements=root/'audits'/f'{name}-measurements.json'
        if not checkpoint.exists() or not measurements.exists():continue
        c=json.loads(checkpoint.read_text());m=json.loads(measurements.read_text())
        run=wandb.init(entity=config['entity'],project=config['project'],id=entry['run_id'],resume='must',dir=str(directory),
                       settings=wandb.Settings(x_disable_stats=True,console='off',quiet=True,disable_code=True,disable_git=True,init_timeout=30))
        run.summary.update({'audit/rewards_replayed':m['rewards_replayed'],
                            'audit/saved_text_parameters':sum(c['saved_text_parameter_elements_by_dtype'].values()),
                            'audit/changed_decoder_layers':sum(x['changed_elements']>0 for x in c['decoder_layer_norm_changes']),
                            'audit/scored_tokens':m['scored_tokens'],
                            'audit/initial_logprob_abs_p999':m['token_quantiles']['p999'],
                            'audit/initial_logprob_abs_max':m['token_quantiles']['max'],
                            'audit/initial_logprob_tokens_above_1':m['tokens_above']['1'],
                            'audit/initial_logprob_tokens_above_5':m['tokens_above']['5']})
        artifact=wandb.Artifact(entry['run_id']+'-audit',type='rl-audit')
        artifact.add_file(str(checkpoint));artifact.add_file(str(measurements))
        logged=run.log_artifact(artifact);logged.wait();uploaded[name]={'artifact':logged.name,'url':entry['url']}
        run.finish();path.write_text(json.dumps(uploaded,indent=2))
        print('Audit uploaded: '+name,flush=True)


if __name__=='__main__':main()
