"""Immutable validated TSP instances and explicit validation/test splits."""

import math
from dataclasses import dataclass

import numpy as np

from moh.problems.tsp_gls.tour import tour_cost, validate_distances, validate_tour


def _immutable_array(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class GLSInstance:
    id: str
    coordinates: np.ndarray
    distances: np.ndarray
    optimal_cost: float
    optimal_tour: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("instance ID must be a nonempty string")
        matrix = validate_distances(self.distances)
        coordinates = np.asarray(self.coordinates)
        if coordinates.shape != (len(matrix), 2) or coordinates.dtype.kind not in "iuf":
            raise ValueError("coordinates must have shape (n, 2) and numeric values")
        if not np.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")
        if isinstance(self.optimal_cost, (bool, np.bool_)):
            raise ValueError("optimal cost must be positive and finite")  # noqa: TRY004
        cost = float(self.optimal_cost)
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("optimal cost must be positive and finite")
        tour = validate_tour(self.optimal_tour, len(matrix))
        if not math.isclose(tour_cost(matrix, tour), cost, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("reference tour and optimal cost disagree")
        object.__setattr__(self, "coordinates", _immutable_array(coordinates))
        object.__setattr__(self, "distances", _immutable_array(matrix))
        object.__setattr__(self, "optimal_cost", cost)
        object.__setattr__(self, "optimal_tour", tour)


@dataclass(frozen=True)
class GLSTask:
    id: str
    size: int
    validation: tuple[GLSInstance, ...]
    test: tuple[GLSInstance, ...]
    provenance: dict

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("task ID must be a nonempty string")
        if type(self.size) is not int or self.size < 4:
            raise ValueError("task size must be an integer >= 4")
        validation, test = tuple(self.validation), tuple(self.test)
        if not validation or not test:
            raise ValueError("task splits must be nonempty")
        instances = (*validation, *test)
        if any(
            not isinstance(x, GLSInstance) or len(x.distances) != self.size
            for x in instances
        ):
            raise ValueError("all instances must have the task size")
        if len({x.id for x in instances}) != len(instances):
            raise ValueError("task splits must have unique instance IDs")
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "test", test)
        object.__setattr__(self, "provenance", dict(self.provenance))
