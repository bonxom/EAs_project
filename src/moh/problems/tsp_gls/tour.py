"""Validate closed tours and convert predecessor/successor routes."""

from itertools import pairwise

import numpy as np


def validate_distances(distances) -> np.ndarray:
    """Return a numeric symmetric matrix, rejecting invalid TSP costs."""
    matrix = np.asarray(distances)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 4:
        raise ValueError("distances must be square with at least four cities")
    if matrix.dtype.kind not in "iuf" or not np.isfinite(matrix).all():
        raise ValueError("distances must be finite real numbers")
    if (matrix < 0).any() or not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError("distances must be nonnegative and symmetric")
    if not np.allclose(np.diag(matrix), 0, rtol=0, atol=1e-10):
        raise ValueError("distances must have a zero diagonal")
    return matrix


def validate_tour(tour, size: int | None = None) -> tuple[int, ...]:
    """Require each city once, with city zero repeated only at the end."""
    try:
        values = tuple(tour)
    except TypeError as exc:
        raise ValueError("tour must be an integer sequence") from exc
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in values
    ):
        raise ValueError("tour cities must be integers")
    size = len(values) - 1 if size is None else size
    if size < 4 or len(values) != size + 1 or values[0] != 0 or values[-1] != 0:
        raise ValueError("tour must start/end at zero and contain n+1 cities")
    if set(values[:-1]) != set(range(size)):
        raise ValueError("tour must visit each city exactly once")
    return tuple(int(v) for v in values)


def tour_cost(distances, tour) -> float:
    matrix = validate_distances(distances)
    cities = validate_tour(tour, len(matrix))
    return float(sum(matrix[a, b] for a, b in pairwise(cities)))


def tour_to_route(tour) -> np.ndarray:
    cities = validate_tour(tour)
    route = np.empty((len(cities) - 1, 2), dtype=np.int64)
    for index, city in enumerate(cities[:-1]):
        route[city] = cities[index - 1] if index else cities[-2], cities[index + 1]
    return route


def route_to_tour(route) -> tuple[int, ...]:
    route = np.asarray(route)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 4:
        raise ValueError("route must have shape (n, 2), n >= 4")
    if route.dtype.kind not in "iu" or (route < 0).any() or (route >= len(route)).any():
        raise ValueError("route must contain valid integer cities")
    cities = [0]
    for _ in range(len(route)):
        current = cities[-1]
        successor = int(route[current, 1])
        if route[successor, 0] != current:
            raise ValueError("route predecessor/successor links disagree")
        cities.append(successor)
    return validate_tour(cities, len(route))


def nearest_neighbor(distances) -> tuple[int, ...]:
    matrix = validate_distances(distances)
    tour = [0]
    remaining = set(range(1, len(matrix)))
    while remaining:
        city = min(remaining, key=lambda city: (matrix[tour[-1], city], city))
        tour.append(city)
        remaining.remove(city)
    return (*tour, 0)
