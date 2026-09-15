"""Audit same-weight trainer/inference log-probability tails from saved rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--run-dir',type=Path,required=True)
 p.add_argument('--output',type=Path)
 args=p.parse_args()
 name=args.run_dir.name
 chunks=[]; seqmeans=[]; tails=[]
 for path in sorted((args.run_dir/'train_data').glob('*.pt')):
  d=torch.load(path,map_location='cpu',weights_only=False);r=d['rollout_data']
  for i,(p,q,mask) in enumerate(zip(r['log_probs'],r['rollout_log_probs'],r['loss_masks'])):
   p=p.float();q=q.float();mask=mask.bool()
   assert p.shape==q.shape==mask.shape
   delta=(p-q)[mask].numpy();absolute=np.abs(delta)
   assert np.isfinite(absolute).all()
   chunks.append(absolute);seqmeans.append(float(absolute.mean()))
   if absolute.max()>1:
    pos=int(torch.where(mask)[0][int(absolute.argmax())])
    prompt_len=r['total_lengths'][i]-r['response_lengths'][i]
    tails.append({'collection':d['rollout_id'],'rank':d['rank'],'sample_index':r['sample_indices'][i],
                  'response_position':pos,'trainer_logp':float(p[pos]),'rollout_logp':float(q[pos]),
                  'token_id':int(r['tokens'][i][prompt_len+pos]),'absdiff':float(absolute.max())})
 allvals=np.concatenate(chunks)
 result={'run':name,'tokens':len(allvals),'sequence_mean_absdiff':float(np.mean(seqmeans)),
         'token_mean_absdiff':float(allvals.mean()),'quantiles':dict(zip(['median','p99','p999','p9999','max'],[float(x) for x in np.quantile(allvals,[.5,.99,.999,.9999,1])])),
         'fraction_above':{str(t):float(np.mean(allvals>t)) for t in [1,5,20]},
         'largest_sequence_outliers':sorted(tails,key=lambda x:x['absdiff'],reverse=True)[:10]}
 output=args.output or args.run_dir/'logprob-tail-audit.json'
 tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(output)
 print(json.dumps({k:v for k,v in result.items() if k!='largest_sequence_outliers'}),flush=True)


if __name__=='__main__':main()
