import itertools

import numpy as np
import pytest

from moh.problems.tsp_gls.exact import held_karp
from moh.problems.tsp_gls.tour import (
    nearest_neighbor,
    route_to_tour,
    tour_cost,
    tour_to_route,
)


def square_distances():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)


def test_square_optimum_and_route_roundtrip():
    distance = square_distances()
    cost, tour = held_karp(distance)
    assert cost == pytest.approx(4.0)
    assert tour_cost(distance, tour) == pytest.approx(cost)
    assert route_to_tour(tour_to_route(tour)) == tour
    assert nearest_neighbor(distance) == (0, 1, 2, 3, 0)


def test_exact_matches_brute_force_non_euclidean():
    rng = np.random.default_rng(82)
    matrix = rng.uniform(0.1, 3.0, (7, 7))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    expected = min(
        tour_cost(matrix, (0, *p, 0)) for p in itertools.permutations(range(1, 7))
    )
    cost, tour = held_karp(matrix)
    assert cost == pytest.approx(expected)
    assert tour_cost(matrix, tour) == pytest.approx(expected)


def test_exact_size_guard():
    with pytest.raises(ValueError):
        held_karp(np.ones((13, 13)))


@pytest.mark.parametrize(
    "tour",
    [
        (0, 1, 2, 2, 0),
        (0, 1, 2, 4, 0),
        (0.0, 1.0, 2.0, 3.0, 0.0),
        (False, 1, 2, 3, False),
        (1, 2, 3, 0, 1),
        (0, 1, 2, 3),
    ],
)
def test_reject_invalid_tours(tour):
    with pytest.raises(ValueError):
        tour_to_route(tour)


def test_reject_disconnected_route():
    route = np.array([[1, 1], [0, 0], [3, 3], [2, 2]])
    with pytest.raises(ValueError):
        route_to_tour(route)
    with pytest.raises(ValueError):
        route_to_tour(route.astype(float))


@pytest.mark.parametrize("dtype", [np.uint8, np.int8])
def test_tour_cost_does_not_overflow_narrow_integer_matrix(dtype):
    distances = np.full((4, 4), 100, dtype=dtype)
    np.fill_diagonal(distances, 0)
    assert tour_cost(distances, (0, 1, 2, 3, 0)) == 400.0
