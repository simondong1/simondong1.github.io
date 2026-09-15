"""Small, local per-example records for paired evaluation and length analysis."""
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np


def record(sample, args):
    return {
        "uid": sample.metadata.get("uid"),
        "group_index": sample.group_index,
        "index": sample.index,
        "reward": sample.get_reward_value(args),
        "response_length": sample.response_length,
        "status": sample.status.name,
        "response": sample.response,
        "prompt_tokens": len(sample.tokens) - sample.response_length,
    }


def write(args, name, payload):
    directory = Path(args.save) / "../measurements"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as stream:
        stream.write(json.dumps({"time": time.time(), **payload}) + "\n")


def rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    rows = [record(s, args) for s in samples]
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_index"]].append(row["reward"])
    zero_variance = sum(len(set(g)) == 1 for g in groups.values()) / len(groups)
    write(args, "rollouts.jsonl", {
        "rollout_id": rollout_id, "rollout_seconds": rollout_time,
        "optimizer_updates": (rollout_id+1)*args.rollout_batch_size*args.n_samples_per_prompt//args.global_batch_size,
        "prompt_presentations": (rollout_id+1)*args.rollout_batch_size,
        "reward_mean": float(np.mean([r['reward'] for r in rows])),
        "all_correct_group_fraction": sum(all(x==1 for x in g) for g in groups.values())/len(groups),
        "all_incorrect_group_fraction": sum(all(x==0 for x in g) for g in groups.values())/len(groups),
        "generated_tokens": sum(r['response_length'] for r in rows),
        "prompt_tokens": sum(r['prompt_tokens'] for r in rows),
        "truncation_fraction": sum(r['status']=='TRUNCATED' for r in rows)/len(rows),
        "length_quantiles": dict(zip(['p25','p50','p90','p99'],np.quantile([r['response_length'] for r in rows],[.25,.5,.9,.99]).tolist())),
        "zero_variance_group_fraction": zero_variance, "samples": rows,
    })
    return False


def evaluation(rollout_id, args, data, extra_metrics):
    history = Path(args.save) / '../measurements/rollouts.jsonl'
    if history.exists():
        with history.open() as stream: completed = sum(1 for _ in stream)
    else: completed = 0
    for dataset, result in data.items():
        write(args, "eval.jsonl", {
            "rollout_id": rollout_id, "dataset": dataset,
            "completed_collections": completed,
            "optimizer_updates": completed*args.rollout_batch_size*args.n_samples_per_prompt//args.global_batch_size,
            "rewards": result["rewards"],
            "truncated": result.get("truncated"),
            "samples": [record(s, args) for s in result.get("samples", [])],
        })
    return False
