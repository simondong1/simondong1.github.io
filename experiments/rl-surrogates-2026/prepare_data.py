"""Pin public HF sources and make disjoint, deduplicated experiment splits."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
import requests

SOURCES = {
    "countdown": ("Jiayi-Pan/Countdown-Tasks-3to4", "408f70d177020686d34a56bba5952feb45aaaee4", "data/train-00000-of-00001.parquet"),
    "deepmath": ("zwhe99/DeepMath-103K", "5cf055d1fe3d7a2eb19719ac020211469736ae44", "data/train-00000-of-00010.parquet"),
}


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": 20260914, "splits": {}, "sources": {}}
    for task, (repo, revision, filename) in SOURCES.items():
        target = out / f"{task}.parquet"
        if not target.exists():
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with target.with_suffix(".part").open("wb") as f:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        f.write(chunk)
            target.with_suffix(".part").rename(target)
        columns = ["nums", "target"] if task == "countdown" else ["question", "final_answer", "difficulty", "topic"]
        rows = pq.read_table(target, columns=columns).to_pylist()
        data = []
        seen = set()
        for index, row in enumerate(rows):
            if task == "countdown":
                if len(row["nums"]) != 4:
                    continue
                identity = json.dumps([sorted(row["nums"]), row["target"]])
                prompt = (f"Use the numbers {row['nums']} exactly once each to make {row['target']}. "
                          "Use only addition, subtraction, multiplication, division and parentheses. "
                          "Do not concatenate numbers or introduce new numbers. "
                          "You may reason briefly, then put your arithmetic expression inside \\boxed{...}.")
                label = {"task": task, "nums": row["nums"], "target": row["target"]}
            else:
                answer = row["final_answer"].strip()
                if not re.fullmatch(r"[+-]?\d{1,8}", answer):
                    continue
                identity = " ".join(row["question"].split())
                prompt = (row["question"] + "\nSolve the problem. Give concise reasoning and put your final integer answer inside \\boxed{...}.")
                label = {"task": task, "answer": answer}
            if identity in seen:
                continue
            seen.add(identity)
            uid = hashlib.sha256(identity.encode()).hexdigest()
            data.append({"prompt": [{"role": "user", "content": prompt}],
                         "label": json.dumps(label),
                         "metadata": {"uid": uid, "source_row": index, "dataset": task}})
        random.Random(20260914).shuffle(data)
        assert len(data) >= 832, (task, len(data))
        splits = {"pilot": data[:64], "test": data[64:320], "train": data[320:832]}
        for split, examples in splits.items():
            path = out / f"{task}_{split}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in examples))
            manifest["splits"][f"{task}_{split}"] = {"count": len(examples), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest["sources"][task] = {"repo": repo, "revision": revision, "file": filename,
                                      "eligible_deduplicated_rows": len(data),
                                      "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        print(task, len(data), manifest["splits"], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
