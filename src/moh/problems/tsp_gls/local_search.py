"""Predecessor/successor moves adapted from MoH; see third_party/moh."""

import numpy as np


def _two_opt_move(route, i, j):
    a, b = route[i, 0], route[j, 0]
    route[i, 0] = route[i, 1]
    route[i, 1] = j
    route[j, 0] = i
    route[a, 1] = b
    route[b, 1] = route[b, 0]
    route[b, 0] = a
    c = route[b, 1]
    while route[c, 1] != j:
        d = route[c, 0]
        route[c, 0] = route[c, 1]
        route[c, 1] = d
        c = d


def _relocate_move(route, i, j):
    a, b = route[i]
    route[a, 1] = b
    route[b, 0] = a
    d = route[j, 1]
    route[d, 0] = i
    route[i] = j, d
    route[j, 1] = i


def _search(route, distances, neighbors, city, *, relocation):
    # Float promotion prevents arithmetic overflow on narrow integer matrices.
    weights = np.asarray(distances, dtype=np.float64)
    result = route.copy()
    best_delta = 0.0
    total_delta = 0.0
    best_move = None
    cities = range(len(result) - 1) if city is None else (city,)
    for i in cities:
        for j in neighbors[i]:
            if i == j:
                continue
            if relocation:
                if result[j, 1] == i:
                    continue
                a, c = result[i]
                e = result[j, 1]
                delta = (
                    -weights[a, i]
                    - weights[i, c]
                    + weights[a, c]
                    - weights[j, e]
                    + weights[j, i]
                    + weights[i, e]
                )
            else:
                if i in result[j] or j in result[i]:
                    continue
                a, b = result[i, 0], result[j, 0]
                delta = weights[a, b] + weights[i, j] - weights[a, i] - weights[b, j]
            if delta < best_delta and not np.isclose(0.0, delta):
                best_delta = float(delta)
                best_move = i, j
                # Upstream per-city traversal applies each increasingly good
                # move immediately, so later deltas use the changed route.
                if city is not None:
                    (_relocate_move if relocation else _two_opt_move)(result, i, j)
                    total_delta += float(delta)
    if city is None and best_move is not None:
        (_relocate_move if relocation else _two_opt_move)(result, *best_move)
        total_delta = best_delta
    return total_delta, result


def two_opt(route, distances, neighbors, city=None) -> tuple[float, np.ndarray]:
    """Copy route and apply upstream 2-opt traversal, returning full cost delta."""
    return _search(route, distances, neighbors, city, relocation=False)


def relocate(route, distances, neighbors, city=None) -> tuple[float, np.ndarray]:
    """Copy route and relocate cities, returning the total cost change."""
    return _search(route, distances, neighbors, city, relocation=True)


def local_search(route, distances, neighbors) -> np.ndarray:
    """Alternate best 2-opt and relocation until neither improves the route."""
    current = route.copy()
    while True:
        delta_two, current = two_opt(current, distances, neighbors)
        delta_relocate, current = relocate(current, distances, neighbors)
        if delta_two == 0 and delta_relocate == 0:
            return current
