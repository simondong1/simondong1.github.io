"""Rebuild the measured math splits from pinned HF sources and ordered IDs.

Ordered IDs avoid dependence on shuffle-library implementation details. The
original files are never modified; every rebuilt split must match its hash.
"""
import argparse
import hashlib
import json
from pathlib import Path


def identity(text):
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def gsm_rows(source):
    from datasets import load_dataset

    dataset = load_dataset(source["repo"], "main", revision=source["revision"])
    rows = {}
    for part in ("train", "test"):
        for raw in dataset[part]:
            question = raw["question"].strip()
            key = identity(question)
            row = {"prompt": [{"role": "user", "content": question +
                    "\n\nSolve step by step. End with your final numerical answer after ####."}],
                   "label": json.dumps({"task": "gsm8k", "answer": raw["answer"].rsplit("####", 1)[1].strip()}),
                   "metadata": {"task": "gsm8k", "id": key}}
            if key in rows:
                assert rows[key] == row, f"Conflicting GSM8K source: {key}"
            rows[key] = row
    return rows


def dapo_rows(source, local_parquet=None):
    import pyarrow.parquet as pq

    if local_parquet:
        paths = [local_parquet]
    else:
        from huggingface_hub import HfApi, hf_hub_download
        files = HfApi().list_repo_files(source["repo"], repo_type="dataset", revision=source["revision"])
        paths = [hf_hub_download(source["repo"], file, repo_type="dataset", revision=source["revision"])
                 for file in files if file.endswith(".parquet")]
    rows, row_groups, answers = {}, {}, {}
    source_rows = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["prompt", "reward_model"]):
            for raw in batch.to_pylist():
                source_rows += 1
                prompt = raw["prompt"]
                key = identity(json.dumps(prompt, sort_keys=True))
                normalized = [{**message, "content": " ".join(message["content"].split())} for message in prompt]
                group = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
                answer = str(raw["reward_model"]["ground_truth"]).strip()
                answers.setdefault(group, set()).add(answer)
                row_groups[key] = group
                if key not in rows:
                    rows[key] = {"prompt": prompt, "label": json.dumps({"task": "dapo", "answer": answer}),
                                 "metadata": {"task": "dapo", "id": key}}
    conflicting = {key for key, values in answers.items() if len(values) != 1}
    report = {"source_rows": source_rows, "unique_prompt_groups": len(answers),
              "conflicting_groups_excluded": len(conflicting), "retained_unique_prompts": len(answers) - len(conflicting)}
    # Keep original text variants here; the published ordered IDs choose the
    # exact representative used by the study for each whitespace-normalized group.
    return {key: row for key, row in rows.items() if row_groups[key] not in conflicting}, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/reproduction-data-manifest.json"))
    parser.add_argument("--ordered-ids", type=Path, default=Path("artifacts/ordered-split-ids.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reproduced-data"))
    parser.add_argument("--dapo-parquet", type=Path)
    parser.add_argument("--task", choices=["gsm8k", "dapo", "both"], default="both")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    order = json.loads(args.ordered_ids.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for task in ("gsm8k", "dapo") if args.task == "both" else (args.task,):
        if task == "gsm8k":
            rows = gsm_rows(manifest["sources"][task])
        else:
            rows, report = dapo_rows(manifest["sources"][task], args.dapo_parquet)
            assert report == manifest["dapo_deduplication"], report
            print(json.dumps(report), flush=True)
        used = set()
        for split, ids in order[task].items():
            assert len(ids) == len(set(ids)) and not used.intersection(ids)
            used.update(ids)
            content = "".join(json.dumps(rows[key], ensure_ascii=False) + "\n" for key in ids)
            expected = manifest["splits"][task][split]
            digest = hashlib.sha256(content.encode()).hexdigest()
            assert len(ids) == expected["rows"] and digest == expected["sha256"], (task, split, digest)
            (args.output / expected["file"]).write_text(content)
            print(json.dumps({"task": task, "split": split, "rows": len(ids), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
