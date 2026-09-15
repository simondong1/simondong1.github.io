"""Verify complete sample/optimizer/evaluation accounting from local artifacts."""
import json
import math
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_run(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    assert status["state"] == "finished", status["state"]
    shape = config["shape"]
    assert shape['prompts'] * shape['group'] == shape['batch'] * shape['steps']
    evaluations = rows(directory / "evaluations.jsonl")
    train = [r for f in directory.glob("metrics-*.jsonl") for r in rows(f) if r["step_key"] == "train/step"]
    for evaluation in evaluations:
        examples = rows(directory / f"eval-{evaluation['cumulative_samples']}-{evaluation['dataset']}.jsonl")
        assert len(examples) == len({r['id'] for r in examples}) == evaluation['n']
        assert all(r['reward'] in [0.0, 1.0] for r in examples)
        assert math.isclose(sum(r['reward'] for r in examples) / len(examples), evaluation['accuracy'], abs_tol=1e-9)
    if config.get("phase") in {"test", "evaluation"}:
        assert config['samples'] == 0 and not train
        assert len(evaluations) == 1
        assert evaluations[0]['cumulative_samples'] == 0
        expected_n = config.get("eval_expected_n", {"gsm8k": 1319, "countdown": 1024}[config["task"]])
        assert evaluations[0]["n"] == expected_n
        assert not (directory / "rollouts.jsonl").exists()
        return {"valid": True, "evaluations": 1, "test_examples": evaluations[0]["n"]}
    rollout_rows = rows(directory / "rollouts.jsonl")
    expected_rollouts = config["samples"] // (shape["prompts"] * shape["group"])
    assert [r["rollout_id"] for r in rollout_rows] == list(range(expected_rollouts))
    assert rollout_rows[-1]["cumulative_samples"] == config["samples"]
    assert all(r["rollout_samples"] == shape["prompts"] * shape["group"] for r in rollout_rows)
    steps = sorted(r["metrics"]["train/step"] for r in train)
    assert steps == list(range(config["samples"] // shape["batch"])), steps
    assert all(math.isfinite(float(v)) for r in train for v in r["metrics"].values())
    expected_evals = [0] + list(range(config["eval_every_samples"], config["samples"] + 1, config["eval_every_samples"]))
    assert [e["cumulative_samples"] for e in evaluations] == expected_evals
    return {"valid": True, "rollouts": len(rollout_rows), "optimizer_steps": len(train),
            "samples": config["samples"], "evaluations": len(evaluations)}


if __name__ == "__main__":
    import sys
    print(json.dumps(validate_run(sys.argv[1]), indent=2))
