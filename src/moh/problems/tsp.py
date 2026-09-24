import itertools
import math
from dataclasses import dataclass

import numpy as np

from moh.core.seeds import derive_seed


def tour_length(coordinates, tour):
    coordinates = np.asarray(coordinates, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < 2
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("coordinates must be finite Nx2 data")
    n = len(coordinates)
    if len(tour) != n + 1 or any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
        for x in tour
    ):
        raise ValueError("tour needs n+1 integer cities")
    if tour[0] != 0 or tour[-1] != 0 or sorted(tour[:-1]) != list(range(n)):
        raise ValueError("tour must visit each city once, starting and ending at 0")
    length = math.fsum(
        math.hypot(*(coordinates[a] - coordinates[b]))
        for a, b in itertools.pairwise(tour)
    )
    if not math.isfinite(length):
        raise ValueError("nonfinite tour length")
    return length


@dataclass(frozen=True)
class TSPTask:
    id: str
    instances: tuple[np.ndarray, ...]
    instance_seeds: tuple[int, ...]

    @classmethod
    def create(cls, size, count, root_seed):
        if type(size) is not int or size not in (10, 20, 50):
            raise ValueError("supported sizes are 10, 20, 50")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        seeds = tuple(
            derive_seed(root_seed, "task", size, "instance", i) for i in range(count)
        )
        instances = tuple(
            np.random.default_rng(s).uniform(0.0, 1.0, size=(size, 2)) for s in seeds
        )
        for instance in instances:
            instance.setflags(write=False)
        return cls(f"tsp{size}", instances, seeds)

    def score_tour(self, instance_index, tour):
        return tour_length(self.instances[instance_index], tour)
