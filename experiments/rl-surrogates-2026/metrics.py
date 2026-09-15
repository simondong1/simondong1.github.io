"""Small, local per-example records for paired evaluation and length analysis."""
from collections import defaultdict
import json
from pathlib import Path
import time


def record(sample, args):
    return {
        "uid": sample.metadata.get("uid"),
        "group_index": sample.group_index,
        "index": sample.index,
        "reward": sample.get_reward_value(args),
        "response_length": sample.response_length,
        "status": sample.status.name,
        "response": sample.response,
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
        "zero_variance_group_fraction": zero_variance, "samples": rows,
    })
    return False


def evaluation(rollout_id, args, data, extra_metrics):
    for dataset, result in data.items():
        write(args, "eval.jsonl", {
            "rollout_id": rollout_id, "dataset": dataset,
            "rewards": result["rewards"],
            "truncated": result.get("truncated"),
            "samples": [record(s, args) for s in result.get("samples", [])],
        })
    return False
