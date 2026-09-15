"""Analyze actual Miles records; preserve distinct sample, token, and time axes."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from validate_run import rows, validate_run


def analyze_run(directory):
    config = json.loads((directory / "config.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    accounting = validate_run(directory)
    if config.get("phase") in {"test", "evaluation"}:
        evaluation = rows(directory / "evaluations.jsonl")[0]
        examples = rows(directory / f"eval-0-{evaluation['dataset']}.jsonl")
        return {"name": directory.name, "config": config, "accounting": accounting,
                "summary": {"accuracy": evaluation["accuracy"], "n": evaluation["n"], "job_seconds": status["job_seconds"],
                            "mean_response_length": float(np.mean([r['response_length'] for r in examples])),
                            "truncation_fraction": float(np.mean([r['status'] == 'truncated' for r in examples]))},
                "examples": [{k: r[k] for k in ["id", "reward", "response_length", "status"]} for r in examples],
                "source_hashes": json.loads((directory / "source_hashes.json").read_text()),
                "wandb_run_id": status.get("wandb_run_id")}
    rollouts = rows(directory / "rollouts.jsonl")
    evaluations = rows(directory / "evaluations.jsonl")
    logs = [r for f in directory.glob("metrics-*.jsonl") for r in rows(f)]
    train = sorted([r for r in logs if r["step_key"] == "train/step"], key=lambda r: r["metrics"]["train/step"])
    perf = {r["metrics"]["rollout/step"]: r["metrics"] for r in logs if "perf/train_time" in r["metrics"]}
    gen = {r["metrics"]["rollout/step"]: r["metrics"] for r in logs if "perf/rollout_time" in r["metrics"]}
    curves = []
    for e in evaluations:
        count = e["cumulative_samples"]
        seen = [r for r in rollouts if r["cumulative_samples"] <= count]
        phase_seconds = sum(r["generation_seconds"] + perf[r["rollout_id"]]["perf/train_time"]
                            + perf[r["rollout_id"]].get("perf/update_weights_time", 0) for r in seen)
        examples = rows(directory / f"eval-{count}-{e['dataset']}.jsonl")
        curves.append({"samples": count, "output_tokens": e["cumulative_output_tokens"],
                       "optimizer_steps": count // config["shape"]["batch"],
                       "active_phase_seconds": phase_seconds,
                       "job_elapsed_seconds": e["time"] - status["started"],
                       "accuracy": e["accuracy"], "n": e["n"],
                       "answer_marker_fraction": e.get("answer_marker_fraction"),
                       "examples": [{k: r[k] for k in ["id", "reward", "response_length", "status"]} for r in examples]})
    metric_names = ["train/pg_clipfrac", "train/grad_norm", "train/ppo_kl", "train/ess_ratio",
                    "train/train_rollout_logprob_abs_diff", "train/train_rollout_kl"]
    inner = []
    for i in range(config["shape"]["steps"]):
        selected = [r for r in train if r["metrics"]["train/step"] % config["shape"]["steps"] == i]
        inner.append({"inner_step": i+1, **{key: float(np.mean([r["metrics"][key] for r in selected])) for key in metric_names}})
    lengths = [n for r in rollouts for n in r["response_lengths"]]
    summary = {"initial_accuracy": curves[0]["accuracy"], "final_accuracy": curves[-1]["accuracy"],
               "sample_auc": float(np.trapezoid([c["accuracy"] for c in curves], [c["samples"] for c in curves]) / config["samples"]),
               "active_phase_seconds": curves[-1]["active_phase_seconds"], "job_seconds": status["job_seconds"],
               "output_tokens": sum(lengths), "mean_response_length": float(np.mean(lengths)),
               "truncation_fraction": float(np.mean([x for r in rollouts for x in r["truncated"]])),
               "mixed_group_fraction": float(np.mean([r["mixed_group_fraction"] for r in rollouts])),
               "mean_training_reward": float(np.mean([r["reward"] for r in rollouts])),
               "distinct_training_prompts": len(set(x for r in rollouts for x in r["prompt_ids"])),
               "prompt_exposures": config["samples"] // config["shape"]["group"],
               "peak_allocated_gib": max(r.get("cuda_max_memory_allocated_bytes", 0) for r in train) / 2**30,
               "generation_seconds": sum(r["generation_seconds"] for r in rollouts),
               "training_seconds": sum(p["perf/train_time"] for p in perf.values()),
               "weight_sync_seconds": sum(p.get("perf/update_weights_time", 0) for p in perf.values())}
    return {"name": directory.name, "config": config, "accounting": accounting, "summary": summary,
            "curves": curves, "inner_steps": inner, "train_metrics": [r["metrics"] for r in train],
            "rollouts": [{**r, "performance": perf[r["rollout_id"]], "generation_metrics": gen[r["rollout_id"]]} for r in rollouts],
            "source_hashes": json.loads((directory / "source_hashes.json").read_text()),
            "wandb_run_id": status.get("wandb_run_id")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "artifacts/results")
    parser.add_argument("--phase", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for directory in sorted(args.root.iterdir()):
        if not (directory / "config.json").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        if args.phase and config.get("phase") not in args.phase:
            continue
        if status["state"] != "finished":
            print(directory.name, status["state"])
            continue
        result = analyze_run(directory)
        results.append(result)
        print(directory.name, json.dumps(result["summary"]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(results, separators=(",", ":")).encode()
        args.output.write_bytes(gzip.compress(serialized, mtime=0) if args.output.suffix == ".gz" else serialized)


if __name__ == "__main__":
    main()
