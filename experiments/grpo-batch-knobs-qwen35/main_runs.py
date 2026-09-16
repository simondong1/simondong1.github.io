"""Names of the two learning trajectories eligible for final reporting."""
import json
from pathlib import Path


def main_runs(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    return json.loads((root / "main_runs.json").read_text())
