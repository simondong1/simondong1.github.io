"""Exact batching validation and the prospectively declared experiment grid."""
from dataclasses import asdict, dataclass
from math import sqrt


@dataclass(frozen=True)
class BatchShape:
    prompts: int
    group: int
    batch: int
    steps: int

    def validate(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("All four batch knobs must be positive integers")
        if self.prompts * self.group != self.batch * self.steps:
            raise ValueError(f"Inconsistent batch shape: {self}")
        return self

    @property
    def trajectories(self):
        return self.prompts * self.group


GRID = {
    "onpolicy-small": (BatchShape(16, 8, 128, 1), 1.0),
    "split-2": (BatchShape(32, 8, 128, 2), 1.0),
    "split-4": (BatchShape(64, 8, 128, 4), 1.0),
    "split-8": (BatchShape(128, 8, 128, 8), 1.0),
    "batch-256": (BatchShape(64, 8, 256, 2), 1.0),
    "batch-256-sqrt": (BatchShape(64, 8, 256, 2), sqrt(2)),
    "onpolicy-large": (BatchShape(64, 8, 512, 1), 1.0),
    "onpolicy-large-sqrt": (BatchShape(64, 8, 512, 1), 2.0),
    "group-4": (BatchShape(128, 4, 128, 4), 1.0),
    "group-16": (BatchShape(32, 16, 128, 4), 1.0),
    "group-32": (BatchShape(16, 32, 128, 4), 1.0),
}


def validate_config(config):
    shape = BatchShape(**config["shape"]).validate()
    eval_only = config.get("phase") in {"test", "evaluation"} and bool(config.get("load")) and config["samples"] == 0
    if not eval_only and (config["samples"] <= 0 or config["samples"] % shape.trajectories):
        raise ValueError("Sample budget must contain whole rollouts")
    if config["eval_every_samples"] <= 0 or config["eval_every_samples"] % shape.trajectories:
        raise ValueError("Evaluation spacing must contain whole rollouts")
    if shape.group < 2:
        raise ValueError("This study requires non-singleton GRPO groups")
    return shape
