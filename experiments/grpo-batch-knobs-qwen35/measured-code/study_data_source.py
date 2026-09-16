"""Preserve data position, controller RNG and sample axes for continuation.

Megatron saves optimizer and trainer RNG separately. This does not promise
bitwise identical SGLang sampling after a process restart.
"""
import random
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from miles.rollout.data_source import RolloutDataSourceWithBuffer

import study_hooks


class PersistentDataSource(RolloutDataSourceWithBuffer):
    def save(self, rollout_id):
        super().save(rollout_id)
        state = dict(samples=study_hooks._generated, tokens=study_hooks._generated_tokens,
                     buffer=self.buffer, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state())
        path = Path(self.args.save) / "rollout" / f"study_state_{rollout_id}.pt"
        torch.save(state, path)
        if prefix := os.environ.get("STUDY_CHECKPOINT_PREFIX"):
            # Publish the pointer last, after weights, Adam, trainer RNG and
            # data-source state have all reached durable storage.
            start = time.time()
            base = Path(self.args.save)
            iteration = f"iter_{rollout_id:07d}"
            for local, remote in [(base / iteration, f"{prefix}/{iteration}"),
                                  (base / "rollout", f"{prefix}/rollout")]:
                subprocess.run(["aws", "s3", "sync", str(local), remote, "--only-show-errors"], check=True)
            subprocess.run(["aws", "s3", "cp", str(base / "latest_checkpointed_iteration.txt"),
                            f"{prefix}/latest_checkpointed_iteration.txt", "--only-show-errors"], check=True)
            study_hooks.append("checkpoints.jsonl", {"rollout_id": rollout_id, "samples": state["samples"],
                               "archive_seconds": time.time() - start, "archived": True})

    def load(self, rollout_id=None):
        super().load(rollout_id)
        if not self.args.load or rollout_id is None or rollout_id < 0:
            return
        path = Path(self.args.load) / "rollout" / f"study_state_{rollout_id}.pt"
        if not path.exists():
            raise RuntimeError(f"Missing study continuation state: {path}")
        state = torch.load(path, weights_only=False, map_location="cpu")
        self.buffer = state["buffer"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        study_hooks._generated = state["samples"]
        study_hooks._generated_tokens = state["tokens"]
        study_hooks.append("resume.jsonl", {"rollout_id": rollout_id, "samples": state["samples"],
                                          "tokens": state["tokens"], "buffer_groups": len(self.buffer)})
