"""Record native Miles metrics without modifying rewards or gradients."""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

_generated = 0
_generated_tokens = 0
_mixed_group_fraction = 0.0


def append(name, value):
    root = Path(os.environ["STUDY_RUN_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / name).open("a") as handle:
        handle.write(json.dumps(value, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def setup_worker():
    from study_topology import install
    install()
    from miles.utils import tracking_utils as tracking
    if getattr(tracking.log, "_study_observer", False):
        return
    original = tracking.log
    axes_defined = False

    def observed(args, metrics, step_key):
        nonlocal axes_defined
        import torch
        metrics = dict(metrics)
        if step_key == "train/step":
            metrics["train/samples"] = (metrics["train/step"] + 1) * args.global_batch_size
        elif step_key == "eval/step":
            metrics.update({"eval/samples": _generated, "eval/output_tokens": _generated_tokens})
        elif step_key == "rollout/step":
            metrics["rollout/samples"] = (metrics["rollout/step"] + 1) * args.rollout_batch_size * args.n_samples_per_prompt
            if _generated:
                metrics.update({"rollout/output_tokens": _generated_tokens,
                                "rollout/mixed_group_fraction": _mixed_group_fraction})
        if not axes_defined:
            import wandb
            if wandb.run:
                for prefix in ("train", "eval", "rollout"):
                    wandb.define_metric(f"{prefix}/samples")
                    wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/samples", overwrite=True)
                wandb.define_metric("perf/*", step_metric="rollout/samples", overwrite=True)
                axes_defined = True
        record = {"time": time.time(), "step_key": step_key, "metrics": metrics}
        if torch.cuda.is_initialized():
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        append(f"metrics-{os.getpid()}.jsonl", record)
        result = original(args, metrics, step_key)
        if "perf/train_time" in metrics and metrics.get("rollout/step") == args.num_rollout - 1:
            tracking.finish_tracking()
        elif step_key == "eval/step" and _generated == args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt:
            tracking.finish_tracking()
        return result

    observed._study_observer = True
    tracking.log = observed


def rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    global _generated, _generated_tokens, _mixed_group_fraction
    _generated += len(samples)
    _generated_tokens += sum(sample.response_length for sample in samples)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(float(sample.get_reward_value(args)))
    _mixed_group_fraction = sum(min(g) != max(g) for g in groups.values()) / len(groups)
    append("rollouts.jsonl", {"time": time.time(), "rollout_id": rollout_id, "samples": len(samples),
                              "cumulative_samples": _generated, "cumulative_output_tokens": _generated_tokens,
                              "prompt_ids": [sample.metadata.get("id") for sample in samples],
                              "group_rewards": list(groups.values()),
                              "output_tokens": sum(sample.response_length for sample in samples),
                              "generation_seconds": rollout_time,
                              "reward": sum(sum(g) for g in groups.values()) / len(samples),
                              "mixed_group_fraction": _mixed_group_fraction,
                              "lengths": [s.response_length for s in samples],
                              "truncated": [s.status.value == "truncated" for s in samples]})
    if rollout_id == 0:
        for sample in samples[:16]:
            append("samples.jsonl", {"response": sample.response, "label": sample.label,
                                     "reward": sample.get_reward_value(args), "status": sample.status.value})
    return False


def evaluation(rollout_id, args, data, extra_metrics):
    for task, item in data.items():
        rewards = item["rewards"]
        append("evaluations.jsonl", {"time": time.time(), "rollout_id": rollout_id, "task": task,
                                    "cumulative_samples": _generated, "cumulative_output_tokens": _generated_tokens,
                                    "accuracy": sum(rewards) / len(rewards), "n": len(rewards),
                                    "extra_metrics": extra_metrics})
        for sample in item.get("samples") or []:
            append(f"eval-{_generated}-{task}.jsonl", {"id": sample.metadata.get("id"),
                   "response": sample.response, "label": sample.label, "reward": sample.get_reward_value(args),
                   "response_length": sample.response_length, "status": sample.status.value})
    return False
