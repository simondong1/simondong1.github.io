"""Pin public datasets, deduplicate task identities, and freeze disjoint splits."""
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from datasets import load_dataset
from huggingface_hub import HfApi


def write_split(task, split, rows):
    path = ROOT / "data" / f"{task}-{split}.jsonl"
    path.parent.mkdir(exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text)
    return {"path": path.name, "rows": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def gsm_row(row):
    question = row["question"].strip()
    return {
        "prompt": [{"role": "user", "content": question + "\n\nSolve step by step. End with your final numerical answer after ####."}],
        "label": json.dumps({"task": "gsm8k", "answer": row["answer"].rsplit("####", 1)[1].strip()}),
        "metadata": {"task": "gsm8k", "id": hashlib.sha256(question.encode()).hexdigest()[:20]},
    }


def countdown_row(row):
    numbers = [int(n) for n in row["nums"]]
    target = int(row["target"])
    identity = json.dumps([sorted(numbers), target])
    prompt = (f"Using the numbers {numbers}, write an arithmetic expression that equals {target}. "
              "Use every number exactly once. You may use +, -, *, / and parentheses. "
              "Do not combine digits or use other numbers. Think step by step, then put only the "
              "final expression inside <answer>...</answer>.")
    return {"prompt": [{"role": "user", "content": prompt}],
            "label": json.dumps({"task": "countdown", "numbers": numbers, "target": target}),
            "metadata": {"task": "countdown", "id": hashlib.sha256(identity.encode()).hexdigest()[:20]}}


def deduplicate(rows, forbidden=None):
    seen = set() if forbidden is None else set(forbidden)
    result = []
    for row in rows:
        identity = row["metadata"]["id"]
        if identity not in seen:
            seen.add(identity)
            result.append(row)
    return result


def main():
    api = HfApi()
    # Existing manifests pin reproduction to the measured source revisions.
    existing_path = ROOT / "data_manifest.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    manifest = {"split_seed": 1729, "validation_rows": 512, "sources": {}, "splits": {}}
    for task, repo, config in [("gsm8k", "openai/gsm8k", "main"),
                               ("countdown", "Jiayi-Pan/Countdown-Tasks-3to4", None)]:
        revision = existing.get("sources", {}).get(task, {}).get("revision") or api.dataset_info(repo).sha
        print(f"Loading {repo}@{revision}", flush=True)
        dataset = load_dataset(repo, config, revision=revision, cache_dir=str(ROOT / ".cache" / "datasets"))
        manifest["sources"][task] = {"repo": repo, "revision": revision, "config": config}
        if task == "gsm8k":
            test = deduplicate([gsm_row(row) for row in dataset["test"]])
            rows = deduplicate([gsm_row(row) for row in dataset["train"]],
                               {row["metadata"]["id"] for row in test})
            random.Random(1729).shuffle(rows)
            validation, train = rows[:512], rows[512:]
        else:
            rows = deduplicate([countdown_row(row) for row in dataset["train"]])
            random.Random(1729).shuffle(rows)
            validation, test, train = rows[:512], rows[512:1536], rows[1536:51536]
        identities = [{r["metadata"]["id"] for r in part} for part in [train, validation, test]]
        assert not (identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2])
        manifest["splits"][task] = {name: write_split(task, name, part)
                                      for name, part in [("train", train), ("validation", validation), ("test", test)]}
        print(json.dumps(manifest["splits"][task]), flush=True)
    (ROOT / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
