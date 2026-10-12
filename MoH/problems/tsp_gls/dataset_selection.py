"""Select fixed, reproducible validation and test instances from separate files."""

import pickle
import random
from pathlib import Path

INSTANCE_COUNTS = {"val": 8, "test": 100}
DATA_FIELDS = ("cost", "distance_matrix", "coordinate", "optimal_tour")


def select_instance_indices(data, mode, seed=0, count=None):
    if mode not in INSTANCE_COUNTS:
        raise ValueError(f"Unsupported evaluation mode: {mode!r}; expected val or test")
    lengths = {field: len(data[field]) for field in DATA_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Dataset fields have inconsistent lengths: {lengths}")
    available = lengths["cost"]
    required = INSTANCE_COUNTS[mode] if count is None else count
    if not isinstance(required, int) or required <= 0:
        raise ValueError("Evaluation instance count must be a positive integer")
    if available < required:
        raise ValueError(f"Evaluation mode {mode} requires {required} instances; dataset has {available}")
    if mode == "val":
        return random.Random(seed).sample(range(available), required)
    return list(range(required))


def load_eval_dataset(root_dir, problem_size, mode, seed=0, count=None):
    if mode not in INSTANCE_COUNTS:
        raise ValueError(f"Unsupported evaluation mode: {mode!r}; expected val or test")
    path = (
        Path(root_dir) / "problems" / "tsp_gls" / "data" / mode
        / f"{mode}_instance_data_sol_{problem_size}.pkl"
    )
    with path.open("rb") as source:
        data = pickle.load(source)
    return data, select_instance_indices(data, mode, seed, count), path
