"""Audit and summarize the two frozen learning runs; also works on live snapshots.

Run from the study directory. JSONL is read by file iteration because some
dataset strings contain Unicode line separators. No infrastructure or response
text is copied into the numerical output.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from main_runs import main_runs


def rows(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_rows(directory):
    paths = list(directory.glob("metrics-*.jsonl"))
    if paths:
        return [r for f in paths for r in rows(f)]
    path = directory / "metrics.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else []


def stats(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    assert np.isfinite(x).all()
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), min=float(x.min()), max=float(x.max()))


def paired_change(before, after, seed=20260915, replicates=20000):
    """Percentile interval from paired prompt bootstrap, not training seeds."""
    assert before.keys() == after.keys() and len(before)
    delta = np.array([after[k] - before[k] for k in sorted(before)])
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(delta, size=(min(500, replicates - i), len(delta))).mean(1)
                            for i in range(0, replicates, 500)])
    return {"change": float(delta.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "prompts": len(delta), "improved": int((delta > 0).sum()),
            "regressed": int((delta < 0).sum()), "bootstrap_replicates": replicates}


def scored(path):
    data = rows(path)
    result = {r["id"]: float(r["reward"]) for r in data}
    assert len(result) == len(data) and None not in result, path
    assert all(x in (0, 1) for x in result.values()), path
    return result


def logical_records(directory, config, status):
    """Join retained checkpoint history; preserve every attempt for cost accounting.

    Each source directory remains an immutable W&B-audited execution segment.
    Rewound updates never enter the accepted learning curve a second time.
    """
    records = {k: [] for k in ("metrics", "rollouts", "phases", "checkpoints", "evaluations")}
    cost_metrics, cost_phases, cost_checkpoints, segments = [], [], [], []
    evaluation_paths, evaluation_elapsed = {}, {}
    offset = 0.0
    history = config.get("resume_history", [])
    for entry in [*history, {"run": directory.name, "through_samples": None}]:
        source = directory.parent / entry["run"]
        source_status = json.loads((source/"status.json").read_text())
        source_config = json.loads((source/"config.json").read_text())
        limit = entry["through_samples"]
        if limit is not None:
            assert source_status["state"] in {"failed", "finished", "aborted"}
            assert limit % (config["prompts"]*config["group"]) == 0
        batch = config["prompts"]*config["group"]
        raw_metrics = metric_rows(source)
        raw_rollouts = rows(source/"rollouts.jsonl")
        raw_phases = rows(source/"phases.jsonl")
        raw_checkpoints = rows(source/"checkpoints.jsonl")
        cost_metrics.extend(raw_metrics); cost_phases.extend(raw_phases); cost_checkpoints.extend(raw_checkpoints)
        def keep_metric(row):
            if limit is None:
                return True
            m = row["metrics"]
            if row["step_key"] == "train/step":
                return m["train/samples"] <= limit
            if row["step_key"] == "eval/step":
                return m["eval/samples"] <= limit
            return m.get("rollout/step", -1) < limit//batch
        records["metrics"].extend(r for r in raw_metrics if keep_metric(r))
        records["rollouts"].extend(r for r in raw_rollouts if limit is None or r["cumulative_samples"] <= limit)
        records["phases"].extend(r for r in raw_phases if limit is None or r["rollout_id"] < limit//batch)
        records["checkpoints"].extend(r for r in raw_checkpoints if limit is None or r["samples"] <= limit)
        for e in rows(source/"evaluations.jsonl"):
            n = e["cumulative_samples"]
            if limit is not None and n > limit:
                continue
            assert n not in evaluation_paths, "Repeated validation exposure across execution segments"
            records["evaluations"].append(e)
            evaluation_paths[n] = source/f"eval-{n}-{config['task']}.jsonl"
            evaluation_elapsed[n] = offset+e["time"]-source_status["started"]
        discarded = [r for r in raw_rollouts if limit is not None and r["cumulative_samples"] > limit]
        segments.append({"run": source.name, "state": source_status["state"],
                         "colocate":source_config.get("colocate",False),
                         "train_gpus":source_config["train_gpus"], "rollout_gpus":source_config["rollout_gpus"],
                         "gpu_node_count":source_config.get("gpu_node_count",len(source_config.get("cluster_nodes",[])) or 2),
                         "cluster_node_gpus":source_config.get("cluster_node_gpus",[source_config["worker_gpus"]]*2),
                         "reserved_gpus":source_config.get("reserved_gpus",source_config["train_gpus"]+source_config["rollout_gpus"]),
                         "resume_from_samples":source_config.get("resume_from_samples",0),
                         "save_every_samples":source_config.get("save_every_samples"),
                         "retained_through_responses": limit,
                         "observed_completed_responses": sum(r["samples"] for r in raw_rollouts),
                         "observed_output_tokens": sum(r["output_tokens"] for r in raw_rollouts),
                         "discarded_responses": sum(r["samples"] for r in discarded),
                         "discarded_output_tokens": sum(r["output_tokens"] for r in discarded),
                         "observed_optimizer_updates": sum(r["step_key"]=="train/step" for r in raw_metrics),
                         "discarded_optimizer_updates": sum(r["step_key"]=="train/step" and limit is not None and r["metrics"]["train/samples"]>limit for r in raw_metrics),
                         "job_seconds": source_status.get("job_seconds"), "started": source_status["started"],
                         "finished": source_status.get("finished"),
                         "job_seconds_estimated":source_status.get("job_seconds_estimated",False),
                         "job_seconds_bounds":source_status.get("job_seconds_bounds")})
        if limit is not None:
            offset += source_status["job_seconds"]
    return records, cost_metrics, cost_phases, cost_checkpoints, evaluation_paths, evaluation_elapsed, segments


def summarize(directory, data_root):
    config = json.loads((directory / "config.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    task = config["task"]
    p, g, b, s = [config[k] for k in ("prompts", "group", "batch", "steps")]
    assert p * g == b * s
    records, cost_metrics, cost_phases, cost_checkpoints, evaluation_paths, evaluation_elapsed, segments = logical_records(directory, config, status)
    metrics = sorted(records["metrics"], key=lambda r: r["time"])
    train = [r for r in metrics if r["step_key"] == "train/step"]
    assert [r["metrics"]["train/step"] for r in train] == list(range(len(train)))
    assert all(r["metrics"]["train/samples"] == (i + 1) * b for i, r in enumerate(train))
    assert all(math.isfinite(v) for r in train for v in r["metrics"].values() if isinstance(v, (int, float)))
    assert all(r["metrics"]["train/lr-pg_0"] == config["lr"] for r in train)
    rollouts = records["rollouts"]
    assert [r["rollout_id"] for r in rollouts] == list(range(len(rollouts)))
    cumulative_tokens = 0
    for i, r in enumerate(rollouts):
        assert r["samples"] == p * g and r["cumulative_samples"] == (i + 1) * p * g
        assert len(r["group_rewards"]) == p and all(len(x) == g for x in r["group_rewards"])
        assert len(r["lengths"]) == len(r["truncated"]) == len(r["prompt_ids"]) == p * g
        cumulative_tokens += sum(r["lengths"])
        assert cumulative_tokens == r["cumulative_output_tokens"]
        assert sum(r["lengths"]) == r["output_tokens"]
    train_ids = {r["metadata"]["id"] for r in rows(data_root / config["data_file"])}
    assert all(pid in train_ids for r in rollouts for pid in r["prompt_ids"]), "Training used an unexpected prompt"
    expected_ids = {r["metadata"]["id"] for r in rows(data_root / config["eval_file"])}
    assert train_ids.isdisjoint(expected_ids)
    assert len(expected_ids) == 512
    phases = records["phases"]
    finished_phases = [r for r in phases if r["event"] == "end"]
    cost_finished_phases = [r for r in cost_phases if r["event"] == "end"]
    training_phases = [r for r in metrics if "perf/train_time" in r["metrics"]]
    weight_phases = [r for r in metrics if "perf/update_weights_time" in r["metrics"]]
    assert len({r["metrics"]["rollout/step"] for r in training_phases}) == len(training_phases)
    if not config.get("resume_history"):
        assert len({r["metrics"]["rollout/step"] for r in weight_phases}) == len(weight_phases)
    cost_training_phases = sorted([r for r in cost_metrics if "perf/train_time" in r["metrics"]], key=lambda r:r["time"])
    cost_weight_phases = [r for r in cost_metrics if "perf/update_weights_time" in r["metrics"]]
    evaluations = records["evaluations"]
    by_sample, eval_curve = {}, []
    for e in evaluations:
        n = e["cumulative_samples"]
        result_path = evaluation_paths[n]
        predictions = scored(result_path)
        assert predictions.keys() == expected_ids
        assert len(predictions) == e["n"]
        assert abs(np.mean(list(predictions.values())) - e["accuracy"]) < 1e-12
        assert n not in by_sample
        by_sample[n] = predictions
        end = e["time"]
        active = sum(r["seconds"] for r in cost_finished_phases if r["time"] <= end and r["kind"] == "generation")
        active += sum(r["metrics"]["perf/train_time"] for r in cost_training_phases if r["time"] <= end)
        active += sum(r["metrics"]["perf/update_weights_time"] for r in cost_weight_phases if r["time"] <= end)
        eval_curve.append({"responses": n, "output_tokens": e["cumulative_output_tokens"],
                           "updates": n // b, "accuracy": e["accuracy"], "n": e["n"],
                           "elapsed_seconds": evaluation_elapsed[n], "measured_active_seconds": active,
                           "prompt_results_sha256": digest(result_path)})
    checkpoints = records["checkpoints"]
    saved_samples = [r["samples"] for r in checkpoints if r.get("archived")]
    completed = status["state"] == "finished"
    if completed:
        assert len(rollouts) == config["rollouts"]
        assert len(train) == config["rollouts"] * s
        assert len(training_phases) == config["rollouts"]
        assert sorted(by_sample) == list(range(0, p*g*config["rollouts"]+1, config["eval_every_samples"]))
        expected_saved = []
        for segment in segments:
            end = segment["retained_through_responses"]
            end = p*g*config["rollouts"] if end is None else end
            interval = segment["save_every_samples"]
            expected_saved.extend(range(segment["resume_from_samples"]+interval,end+1,interval))
        assert saved_samples == expected_saved
    selection = None
    if completed:
        selection_interval=config.get("selection_every_samples",config["save_every_samples"])
        eligible = [e for e in eval_curve if e["responses"] in [0, *saved_samples] and e["responses"] % selection_interval == 0]
        best = max(eligible, key=lambda e: (e["accuracy"], -e["responses"]))
        selection = {"responses": best["responses"], "validation_accuracy": best["accuracy"],
                     "checkpoint_rollout_index": best["responses"] // (p*g) - 1 if best["responses"] else None,
                     "rule": "Best saved checkpoint or initial model on validation; ties favor earlier"}
    total_phase = lambda key: sum(r["metrics"].get(key, 0) for r in cost_metrics)
    job_seconds = sum(s["job_seconds"] for s in segments) if completed else None
    timing = {"generation_seconds": sum(r["seconds"] for r in cost_finished_phases if r["kind"] == "generation"),
              "training_seconds": total_phase("perf/train_time"),
              "weight_transfer_seconds": total_phase("perf/update_weights_time"),
              "evaluation_seconds": sum(r["seconds"] for r in cost_finished_phases if r["kind"] == "evaluation"),
              "checkpoint_archive_seconds": sum(r["archive_seconds"] for r in cost_checkpoints),
              "complete_job_seconds": job_seconds,
              "elapsed_including_recovery_seconds": status["finished"]-segments[0]["started"] if completed else None,
              "train_phase_seconds": stats([r["metrics"]["perf/train_time"] for r in cost_training_phases]),
              "train_phase_after_first_seconds": stats([r["metrics"]["perf/train_time"] for r in cost_training_phases[1:]]),
              "train_phase_last16_seconds": stats([r["metrics"]["perf/train_time"] for r in training_phases[-16:]]),
              "generation_last16_seconds": stats([r["generation_seconds"] for r in rollouts[-16:]])}
    timing["measured_active_seconds"] = sum(timing[k] for k in ("generation_seconds", "training_seconds", "weight_transfer_seconds"))
    timing["allocated_gpu_hours"] = sum(s["job_seconds"]*s["reserved_gpus"] for s in segments)/3600 if completed else None
    diagnostic_keys = ["pg_clipfrac", "ppo_kl", "ess_ratio", "grad_norm", "train_rollout_logprob_abs_diff"]
    diagnostic = {str(i+1): {key: stats([r["metrics"]["train/"+key] for r in train if r["metrics"]["train/step"] % s == i])
                             for key in diagnostic_keys} for i in range(s)}
    numerical_rollouts = []
    for r in rollouts:
        item = {k: r[k] for k in ("rollout_id", "samples", "cumulative_samples", "cumulative_output_tokens", "output_tokens", "generation_seconds", "reward", "mixed_group_fraction")}
        item.update(mean_response_tokens=float(np.mean(r["lengths"])), truncation_fraction=float(np.mean(r["truncated"])),
                    all_correct_group_fraction=float(np.mean([min(x) == 1 for x in r["group_rewards"]])),
                    all_wrong_group_fraction=float(np.mean([max(x) == 0 for x in r["group_rewards"]])))
        numerical_rollouts.append(item)
    last_change = paired_change(by_sample[0], by_sample[max(by_sample)]) if len(by_sample)>1 else None
    late_change = paired_change(by_sample[65536], by_sample[131072]) if 131072 in by_sample and 65536 in by_sample else None
    return {"task": task, "run": directory.name, "state": status["state"], "completed_audit": completed,
            "configuration": {**{k: config[k] for k in ("prompts", "group", "batch", "steps", "rollouts", "lr", "response_length", "packing", "thinking", "train_gpus", "rollout_gpus", "tp")},
                              "colocate":config.get("colocate",False),
                              "gpu_node_count":config.get("gpu_node_count",len(config.get("cluster_nodes",[])) or 2),
                              "reserved_gpus":config.get("reserved_gpus",config["train_gpus"]+config["rollout_gpus"]),
                              "cluster_node_gpus":config.get("cluster_node_gpus",[config["worker_gpus"]]*config.get("gpu_node_count",2))},
            "counts": {"generated_responses": len(rollouts)*p*g, "consumed_responses": len(train)*b,
                       "updates": len(train), "output_tokens": cumulative_tokens,
                       "observed_completed_responses_all_segments": sum(s["observed_completed_responses"] for s in segments),
                       "observed_output_tokens_all_segments": sum(s["observed_output_tokens"] for s in segments),
                       "discarded_responses": sum(s["discarded_responses"] for s in segments),
                       "discarded_output_tokens": sum(s["discarded_output_tokens"] for s in segments),
                       "observed_optimizer_updates_all_segments": sum(s["observed_optimizer_updates"] for s in segments),
                       "discarded_optimizer_updates": sum(s["discarded_optimizer_updates"] for s in segments),
                       "unique_training_prompts": len({pid for r in rollouts for pid in r["prompt_ids"]})},
            "execution_segments": segments,
            "timing": timing, "validation": eval_curve, "validation_latest_minus_initial": last_change,
            "validation_endpoint_minus_halfway": late_change, "selection": selection,
            "checkpoints": checkpoints, "inner_step_diagnostics": diagnostic, "rollouts": numerical_rollouts,
            "training": [{"step": r["metrics"]["train/step"], "responses": r["metrics"]["train/samples"],
                          **{key: r["metrics"]["train/"+key] for key in diagnostic_keys}} for r in train],
            "observed_main_rank_peak_GiB": max([r.get("peak_allocated_bytes", 0) for r in cost_metrics], default=0)/2**30,
            "notes": ["One learning seed per task; prompt bootstrap does not estimate training-seed uncertainty.",
                      "Measured active time excludes startup, evaluation, checkpointing and uninstrumented orchestration.",
                      "Fresh sequence lengths can trigger compilation after the first rollout.",
                      "After recovery, sample/token learning axes retain only the checkpoint ancestry; time includes discarded execution work. Recovery downtime is reported separately.",
                      "Memory is the peak observed by the logging rank, not the maximum over every GPU."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/main-analysis.json"))
    args = parser.parse_args()
    report = {task: summarize(args.root / run, args.data) for task, run in main_runs().items()}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: {"state": v["state"], **v["counts"], "latest_validation": v["validation"][-1],
                          "selection": v["selection"]} for k,v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
