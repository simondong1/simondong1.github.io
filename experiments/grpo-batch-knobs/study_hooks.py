"""Side-effect-free metric observation plus local artifacts for reproducible analysis."""
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

_generated = 0
_generated_tokens = 0
_last_mixed_fraction = 0.0


def append(name, record):
    directory = Path(os.environ["STUDY_RUN_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as handle:
        handle.write(json.dumps(record, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n")


def setup_worker():
    """Ray's standard worker setup hook: observe existing log calls without changing metrics."""
    from miles.utils import tracking_utils
    if getattr(tracking_utils.log, "_study_observer", False):
        return
    original = tracking_utils.log
    axes_defined = False

    def observed(args, metrics, step_key):
        nonlocal axes_defined
        metrics = dict(metrics)
        if step_key == "train/step":
            metrics["train/samples"] = (metrics["train/step"] + 1) * args.global_batch_size
        elif step_key == "eval/step":
            metrics.update({"eval/samples": _generated, "eval/output_tokens": _generated_tokens})
        elif "perf/rollout_time" in metrics:
            metrics.update({"rollout/samples": _generated, "rollout/output_tokens": _generated_tokens,
                            "rollout/mixed_group_fraction": _last_mixed_fraction})
        if not axes_defined:
            import wandb
            if wandb.run:
                wandb.define_metric("train/samples")
                wandb.define_metric("eval/samples")
                wandb.define_metric("train/*", step_metric="train/samples", overwrite=True)
                wandb.define_metric("eval/*", step_metric="eval/samples", overwrite=True)
                axes_defined = True
        record = {"time": time.time(), "step_key": step_key,
                  "wandb_run_id": getattr(args, "wandb_run_id", None), "metrics": metrics}
        if step_key == "train/step":
            import torch
            if torch.cuda.is_initialized():
                record["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        append(f"metrics-{os.getpid()}.jsonl", record)
        result = original(args, metrics, step_key)
        # Shared W&B runs require secondary writers to flush before the primary
        # exits. Ray actor termination otherwise discards the last buffered batch.
        last_rollout = getattr(args, "num_rollout", 0) - 1
        final_training_metrics = "perf/train_time" in metrics and metrics.get("rollout/step") == last_rollout
        final_evaluation = step_key == "eval/step" and _generated == args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt
        if final_training_metrics or final_evaluation:
            tracking_utils.finish_tracking()
        return result

    observed._study_observer = True
    tracking_utils.log = observed


def rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    global _generated, _generated_tokens, _last_mixed_fraction
    _generated += len(samples)
    _generated_tokens += sum(sample.response_length for sample in samples)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(float(sample.get_reward_value(args)))
    mixed = sum(min(values) != max(values) for values in groups.values())
    _last_mixed_fraction = mixed / len(groups)
    append("rollouts.jsonl", {"time": time.time(), "rollout_id": rollout_id,
                              "cumulative_samples": _generated, "cumulative_output_tokens": _generated_tokens,
                              "rollout_samples": len(samples), "rollout_groups": len(groups),
                              "generation_seconds": rollout_time,
                              "mixed_group_fraction": mixed / len(groups),
                              "reward": sum(sum(values) for values in groups.values()) / len(samples),
                              "group_rewards": list(groups.values()),
                              "prompt_ids": [sample.metadata.get("id") for sample in samples],
                              "response_lengths": [sample.response_length for sample in samples],
                              "truncated": [sample.status.value == "truncated" for sample in samples]})
    if rollout_id == 0:
        for sample in samples[:16]:
            append("sample_audit.jsonl", {"id": sample.metadata.get("id"), "response": sample.response,
                                          "label": sample.label, "reward": sample.get_reward_value(args),
                                          "status": sample.status.value})
    return False


def evaluation(rollout_id, args, data, extra_metrics):
    for name, item in data.items():
        rewards = item["rewards"]
        record = {"time": time.time(), "rollout_id": rollout_id, "dataset": name,
                  "cumulative_samples": _generated, "cumulative_output_tokens": _generated_tokens,
                  "accuracy": sum(rewards) / len(rewards), "n": len(rewards),
                  "rewards": rewards, "extra_metrics": extra_metrics}
        samples = item.get("samples") or []
        if samples:
            record["answer_marker_fraction"] = sum(bool(re.search(r"####|\\boxed|<answer>", sample.response)) for sample in samples) / len(samples)
        append("evaluations.jsonl", record)
        for index, sample in enumerate(item.get("samples") or []):
            append(f"eval-{_generated}-{name}.jsonl", {"id": sample.metadata.get("id"), "index": index,
                                                      "response": sample.response, "label": sample.label,
                                                      "reward": sample.get_reward_value(args),
                                                      "response_length": sample.response_length,
                                                      "status": sample.status.value})
    return False
