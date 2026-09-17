"""Audit zero-training test jobs and compute paired prompt uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_main import digest, metric_rows, paired_change, rows, scored


def analyze(task, root, data_root=None):
    data_root = data_root or root / "data"
    selection = json.loads((root / "artifacts/terminal-evaluations" / task / "selection.json").read_text())
    expected_ids = {r["metadata"]["id"] for r in rows(data_root / f"{task}-test.jsonl")}
    assert digest(data_root/f"{task}-test.jsonl") == selection["test_sha256"]
    budgets = list(dict.fromkeys([0, selection["endpoint_responses"], selection["selected_responses"]]))
    outcomes, scores, completed_scores = {}, {}, {}
    for n in budgets:
        directory = root / "artifacts/results" / f"{task}-qwen35-4b-test-{n}-v1"
        config = json.loads((directory/"config.json").read_text())
        status = json.loads((directory/"status.json").read_text())
        assert status["state"] == "finished", directory
        assert config["rollouts"] == 0 and config["checkpoint_training_responses"] == n
        assert config["persistent_data_source"] is False
        metrics = metric_rows(directory)
        assert not any(r["step_key"] == "train/step" for r in metrics)
        assert not rows(directory/"rollouts.jsonl")
        evaluations = rows(directory/"evaluations.jsonl")
        assert len(evaluations) == 1 and evaluations[0]["cumulative_samples"] == 0
        data_path = directory/f"eval-0-{task}.jsonl"
        scores[n] = scored(data_path)
        assert scores[n].keys() == expected_ids
        records = rows(data_path)
        completed_scores[n] = {r["id"]: float(r["reward"]) if r["status"] == "completed" else 0.0 for r in records}
        accuracy = float(np.mean(list(scores[n].values())))
        assert accuracy == evaluations[0]["accuracy"]
        outcomes[str(n)] = {"accuracy": accuracy, "n":len(expected_ids),
                            "correct": sum(scores[n].values()), "job_seconds":status["job_seconds"],
                            "reserved_gpus":config["reserved_gpus"],
                            "allocated_gpu_hours":status["job_seconds"]*config["reserved_gpus"]/3600,
                            "truncation_fraction":float(np.mean([r["status"]=="truncated" for r in records])),
                            "correct_truncated_responses":sum(r["status"]=="truncated" and r["reward"]==1 for r in records),
                            "mean_response_tokens":float(np.mean([r["response_length"] for r in records])),
                            "predictions_sha256":digest(data_path)}
    return {"task":task, "selection":selection, "tests":outcomes,
            "endpoint_minus_initial":paired_change(scores[0],scores[selection["endpoint_responses"]]),
            "selected_minus_initial":paired_change(scores[0],scores[selection["selected_responses"]]),
            "endpoint_minus_selected":paired_change(scores[selection["selected_responses"]],scores[selection["endpoint_responses"]]),
            "completed_response_sensitivity": {
                "rule":"Post hoc: count every response without completed status as incorrect; do not change the frozen grader or checkpoint selection.",
                "accuracy":{str(n):float(np.mean(list(result.values()))) for n,result in completed_scores.items()},
                "endpoint_minus_initial":paired_change(completed_scores[0],completed_scores[selection["endpoint_responses"]]),
                "selected_minus_initial":paired_change(completed_scores[0],completed_scores[selection["selected_responses"]]),
                "endpoint_minus_selected":paired_change(completed_scores[selection["selected_responses"]],completed_scores[selection["endpoint_responses"]]),
            },
            "note":"Paired prompt bootstrap; one training seed per task. Test never selected checkpoints."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task",choices=["gsm8k","dapo"])
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    report=analyze(args.task,root)
    (root/"artifacts"/f"test-analysis-{args.task}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
