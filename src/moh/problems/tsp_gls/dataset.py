"""Load pickle-free benchmarks or generate seeded exact smoke instances."""

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np

from moh.core.seeds import derive_seed
from moh.problems.tsp_gls.exact import held_karp
from moh.problems.tsp_gls.task import GLSInstance, GLSTask

REQUIRED_ARRAYS = ("coordinates", "distance_matrix", "cost", "optimal_tour")


def validate_dataset_arrays(arrays: dict, id_prefix: str) -> tuple[GLSInstance, ...]:
    """Validate every reference before exposing any selected instances."""
    if any(name not in arrays for name in REQUIRED_ARRAYS):
        raise ValueError("dataset is missing required arrays")
    coordinates, matrices, costs, tours = (
        np.asarray(arrays[name]) for name in REQUIRED_ARRAYS
    )
    if coordinates.ndim != 3 or coordinates.shape[2] != 2:
        raise ValueError("coordinates must have shape (k, n, 2)")
    count, size, _ = coordinates.shape
    if count < 1 or size < 4 or matrices.shape != (count, size, size):
        raise ValueError("distance_matrix must have shape (k, n, n), n >= 4")
    if costs.shape != (count,) or costs.dtype.kind not in "iuf":
        raise ValueError("cost must be a real numeric array with shape (k,)")
    if tours.shape != (count, size) or tours.dtype.kind not in "iu":
        raise ValueError("optimal_tour must be an integer array with shape (k, n)")
    instances = []
    for index in range(count):
        tour = tuple(int(city) for city in tours[index])
        if set(tour) != set(range(size)):
            raise ValueError("reference tour must visit each city exactly once")
        start = tour.index(0)
        normalized = (*tour[start:], *tour[:start], 0)
        instances.append(
            GLSInstance(
                f"{id_prefix}:{index}",
                coordinates[index],
                matrices[index],
                costs[index],
                normalized,
            )
        )
    return tuple(instances)


def _validate_splits(validation_indices, test_indices):
    splits = (tuple(validation_indices), tuple(test_indices))
    for indices in splits:
        if not indices or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("split indices must be nonempty nonnegative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("split indices must not repeat")
    if set(splits[0]) & set(splits[1]):
        raise ValueError("validation and test splits must be disjoint")
    return splits


def load_npz_task(
    path: Path, validation_indices: tuple[int, ...], test_indices: tuple[int, ...]
) -> GLSTask:
    validation_indices, test_indices = _validate_splits(
        validation_indices, test_indices
    )
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    instances = validate_dataset_arrays(arrays, digest)
    if max((*validation_indices, *test_indices)) >= len(instances):
        raise ValueError("split index outside dataset range")
    size = len(instances[0].distances)
    return GLSTask(
        f"tsp_gls{size}",
        size,
        tuple(instances[i] for i in validation_indices),
        tuple(instances[i] for i in test_indices),
        {
            "kind": "npz",
            "path": str(path),
            "sha256": digest,
            "validation_indices": list(validation_indices),
            "test_indices": list(test_indices),
        },
    )


def synthetic_task(
    size: int, validation_count: int, test_count: int, root_seed: int
) -> GLSTask:
    if type(size) is not int or not 4 <= size <= 12:
        raise ValueError("synthetic task size must be between 4 and 12")
    if any(
        type(count) is not int or count < 1 for count in (validation_count, test_count)
    ):
        raise ValueError("split counts must be positive integers")
    splits = []
    seeds = {}
    for split, count in (("validation", validation_count), ("test", test_count)):
        instances = []
        seeds[split] = []
        for index in range(count):
            seed = derive_seed(root_seed, "gls", size, split, index)
            seeds[split].append(seed)
            coordinates = np.random.default_rng(seed).uniform(0.0, 1.0, (size, 2))
            distances = np.linalg.norm(
                coordinates[:, None] - coordinates[None, :], axis=-1
            )
            cost, tour = held_karp(distances)
            instances.append(
                GLSInstance(
                    f"tsp_gls{size}:{split}:{index}", coordinates, distances, cost, tour
                )
            )
        splits.append(tuple(instances))
    return GLSTask(
        f"tsp_gls{size}",
        size,
        *splits,
        {
            "kind": "synthetic",
            "root_seed": root_seed,
            "seed_labels": ["gls", size, "<split>", "<index>"],
            "seeds": seeds,
        },
    )
