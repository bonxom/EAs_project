"""Deterministic MoH guided local search; generated callbacks belong in workers."""

from dataclasses import dataclass, fields

import numpy as np

from moh.problems.tsp_gls.local_search import local_search, relocate, two_opt
from moh.problems.tsp_gls.tour import (
    nearest_neighbor,
    route_to_tour,
    tour_cost,
    tour_to_route,
    validate_distances,
)


@dataclass(frozen=True)
class GLSOptions:
    iterations: int = 10
    perturbation_moves: int = 1
    edges_per_move: int = 5
    neighborhood_size: int = 100
    reset_interval: int = 50

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            minimum = 0 if field.name == "iterations" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field.name} must be an integer >= {minimum}")


def _validate_guidance(value, size):
    matrix = np.asarray(value)
    if (
        matrix.shape != (size, size)
        or matrix.dtype.kind not in "iuf"
        or not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10)
    ):
        raise ValueError(
            "guidance must be a finite nonnegative symmetric numeric matrix"
        )
    # Guided diagonal may be nonzero (as in the upstream baseline).
    return matrix.astype(np.float64, copy=True)


def solve_gls(distances, update_edge_distance, options: GLSOptions) -> tuple[int, ...]:
    """Solve with fixed iteration budgets and score every route on original costs.

    This function does not isolate callbacks. Generated code must be loaded and
    passed here only inside the GLS worker, never in the experiment process.
    """
    original = validate_distances(distances).astype(np.float64, copy=True)
    n = len(original)
    width = min(options.neighborhood_size, n - 1)
    neighbors = np.array(
        [
            [j for j in np.argsort(row, kind="stable") if j != i][:width]
            for i, row in enumerate(original)
        ],
        dtype=np.int64,
    )
    current = local_search(
        tour_to_route(nearest_neighbor(original)), original, neighbors
    )
    best = current.copy()
    best_cost = tour_cost(original, route_to_tour(best))
    penalties = np.zeros_like(original)
    for iteration in range(options.iterations):
        for _ in range(options.perturbation_moves):
            # Upstream tour starts at successor(0); keep its ordering and no closure.
            tour = route_to_tour(current)
            callback_tour = np.array(tour[1:], dtype=np.int64)
            guided = _validate_guidance(
                update_edge_distance(original.copy(), callback_tour, penalties.copy()),
                n,
            )
            gap = guided - original
            for _ in range(options.edges_per_move):
                row, column = np.unravel_index(np.argmax(gap), gap.shape)
                penalties[row, column] += 1
                penalties[column, row] += 1
                gap[row, column] = gap[column, row] = 0
                for city in (row, column):
                    _, current = two_opt(current, guided, neighbors, city)
                    _, current = relocate(current, guided, neighbors, city)
        current = local_search(current, original, neighbors)
        cost = tour_cost(original, route_to_tour(current))
        if cost < best_cost:
            best, best_cost = current.copy(), cost
        if (iteration + 1) % options.reset_interval == 0:
            current = best.copy()
    return route_to_tour(best)
