"""Time native Miles generation and evaluation without changing their output."""
import time

from miles.rollout.sglang_rollout import generate_rollout

from study_hooks import append


def generate(args, rollout_id, data_source, evaluation=False):
    started = time.time()
    kind = "evaluation" if evaluation else "generation"
    append("phases.jsonl", {"kind": kind, "event": "start", "rollout_id": rollout_id, "time": started})
    result = generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    ended = time.time()
    append("phases.jsonl", {"kind": kind, "event": "end", "rollout_id": rollout_id,
                            "time": ended, "seconds": ended - started})
    return result
