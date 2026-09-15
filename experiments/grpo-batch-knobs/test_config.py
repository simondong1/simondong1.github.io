import pytest

from study_config import GRID, BatchShape, validate_config


def test_grid_consumes_exactly_once():
    for shape, _ in GRID.values():
        shape.validate()
        slices = [list(range(i * shape.batch, (i + 1) * shape.batch)) for i in range(shape.steps)]
        flattened = sum(slices, [])
        assert flattened == list(range(shape.trajectories))


def test_rejects_integer_floor_mismatch_and_partial_budgets():
    with pytest.raises(ValueError):
        BatchShape(3, 8, 4, 5).validate()
    with pytest.raises(ValueError):
        validate_config({"shape": {"prompts": 16, "group": 8, "batch": 128, "steps": 1},
                         "samples": 129, "eval_every_samples": 128})


def test_zero_budget_is_only_for_explicit_checkpoint_evaluation():
    config = {"shape": {"prompts": 16, "group": 8, "batch": 128, "steps": 1},
              "samples": 0, "eval_every_samples": 128}
    with pytest.raises(ValueError):
        validate_config(config)
    validate_config(dict(config, phase="test", load="checkpoint"))
