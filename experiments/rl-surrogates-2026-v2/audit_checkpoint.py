"""CPU-only audit of saved text-policy weights and changes across all layers.

The study omits optimizer checkpoints for I/O efficiency. Audit their coverage
only when an optimizer checkpoint is present; never imply it was saved.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/model'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model_reader=dcp.FileSystemReader(str(args.checkpoint/'model'))
    metadata=model_reader.read_metadata().state_dict_metadata
    text_weights={k:v for k,v in metadata.items() if hasattr(v,'size') and '.visual.' not in k and '.mtp.' not in k}
    saved_counts={}
    for v in text_weights.values():
        dtype=str(v.properties.dtype)
        saved_counts[dtype]=saved_counts.get(dtype,0)+math.prod(v.size)
    assert sum(saved_counts.values())==8_953_803_264, 'Checkpoint text-policy size differs from the declared model'
    moments={}
    if (args.checkpoint/'optimizer'/'.metadata').is_file():
        optimizer=dcp.FileSystemReader(str(args.checkpoint/'optimizer')).read_metadata().state_dict_metadata
        moments={k:v for k,v in optimizer.items() if k.endswith('.exp_avg')}
    counts={}
    for k,v in moments.items():
        dtype=str(v.properties.dtype)
        counts[dtype]=counts.get(dtype,0)+math.prod(v.size)
    keys=[k for k in metadata if '.language_model.layers.' in k and k.endswith('.input_layernorm.weight')]
    assert len(keys)==32, (len(keys),keys)
    state={k:torch.empty(metadata[k].size,dtype=metadata[k].properties.dtype,device='cpu') for k in keys}
    dcp.load(state_dict=state,storage_reader=model_reader)
    index=json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    differences=[]
    for k in sorted(keys,key=lambda x:int(x.split('.layers.')[1].split('.')[0])):
        original_key=k.removeprefix('model_state.model.')
        with safe_open(args.model/index[original_key],framework='pt',device='cpu') as f:
            before=f.get_tensor(original_key).float()
        difference=(state[k].float()-before).abs()
        differences.append({'parameter':original_key,'max_abs_change':difference.max().item(),
                            'mean_abs_change':difference.mean().item(),
                            'changed_elements':int((difference>0).sum()),'elements':difference.numel()})
    assert all(x['changed_elements']>0 for x in differences), 'A decoder layer did not change'
    if moments:
        assert sum(counts.values())==8_953_803_264, 'Optimizer state does not cover the complete language policy'
        assert set(counts)=={'torch.float32'}, 'Unexpected Adam state precision'
    assert not any('lora' in k.lower() for k in metadata), 'Unexpected adapter parameters'
    result={'checkpoint':str(args.checkpoint),'saved_text_parameter_elements_by_dtype':saved_counts,
            'optimizer_checkpoint_present':bool(moments),'optimizer_first_moment_elements_by_dtype':counts,
            'optimizer_parameter_tensors':len(moments),'decoder_layer_norm_changes':differences}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'saved_text_parameters':sum(saved_counts.values()),'optimizer_checkpoint_present':bool(moments),
                      'changed_decoder_layers':len(differences)}))


if __name__=='__main__':main()
