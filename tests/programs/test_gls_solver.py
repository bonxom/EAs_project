import numpy as np
import pytest

from moh.problems.tsp_gls.local_search import local_search, relocate, two_opt
from moh.problems.tsp_gls.solver import GLSOptions, solve_gls
from moh.problems.tsp_gls.tour import (
    nearest_neighbor,
    route_to_tour,
    tour_cost,
    tour_to_route,
)


def distances(seed=7, n=9):
    xy = np.random.default_rng(seed).random((n, 2))
    return np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)


def neighbors(d):
    return np.array(
        [
            [j for j in np.argsort(row, kind="stable") if j != i]
            for i, row in enumerate(d)
        ]
    )


def test_local_search_uncrosses_square():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    solved = route_to_tour(
        local_search(tour_to_route((0, 2, 1, 3, 0)), d, neighbors(d))
    )
    assert tour_cost(d, solved) == pytest.approx(4.0)


@pytest.mark.parametrize("move", [two_opt, relocate])
@pytest.mark.parametrize("city", [None, 0, 3, 8])
@pytest.mark.parametrize("seed", range(5))
def test_move_delta_matches_full_cost(move, city, seed):
    d = distances(seed)
    route = tour_to_route((*range(9), 0))
    before = route.copy()
    delta, result = move(route, d, neighbors(d), city)
    np.testing.assert_array_equal(route, before)
    assert tour_cost(d, route_to_tour(result)) - tour_cost(
        d, route_to_tour(route)
    ) == pytest.approx(delta)
    assert delta <= 0


def test_guidance_cannot_replace_true_distances_or_penalties():
    d = distances()
    original = d.copy()
    seen = []

    def guide(matrix, tour, penalties):
        assert len(tour) == len(d) and set(tour) == set(range(len(d)))
        seen.append(penalties.copy())
        penalties[:] = -999
        matrix[:] = 0.0
        return matrix

    result = solve_gls(d, guide, GLSOptions(iterations=3))
    np.testing.assert_array_equal(d, original)
    assert len(seen) == 3
    assert all((p >= 0).all() and np.array_equal(p, p.T) for p in seen)
    baseline = route_to_tour(
        local_search(tour_to_route(nearest_neighbor(d)), d, neighbors(d))
    )
    assert tour_cost(d, result) <= tour_cost(d, baseline) + 1e-10
    # Zero gaps select (0, 0) five times, preserving upstream's doubled diagonal penalty.
    assert seen[1][0, 0] == 10


def test_zero_iterations_does_not_call_guidance():
    def guide(*args):
        pytest.fail("zero budget called heuristic")

    d = distances()
    result = solve_gls(d, guide, GLSOptions(iterations=0))
    assert result == route_to_tour(
        local_search(tour_to_route(nearest_neighbor(d)), d, neighbors(d))
    )


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros((4, 3)),
        np.full((9, 9), np.nan),
        np.full((9, 9), -1.0),
        np.eye(9, k=1),
        np.full((9, 9), "x"),
    ],
)
def test_invalid_guidance_rejected(bad):
    with pytest.raises(ValueError, match="guidance"):
        solve_gls(distances(), lambda *args: bad, GLSOptions(iterations=1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("iterations", -1),
        ("iterations", True),
        ("reset_interval", 0),
        ("edges_per_move", 0),
        ("neighborhood_size", 0),
        ("perturbation_moves", 0),
    ],
)
def test_invalid_options(field, value):
    with pytest.raises(ValueError):
        GLSOptions(**{field: value})


def test_duplicate_zero_distances_filter_self(monkeypatch):
    from moh.problems.tsp_gls import solver

    captured = []
    original = solver.local_search

    def capture(route, matrix, nearest):
        captured.append(nearest.copy())
        return original(route, matrix, nearest)

    monkeypatch.setattr(solver, "local_search", capture)
    d = np.zeros((5, 5))
    assert solve_gls(d, lambda d, *_: d, GLSOptions(iterations=2)) == (0, 1, 2, 3, 4, 0)
    for nearest in captured:
        for city, row in enumerate(nearest):
            assert city not in row
            assert set(row) == set(range(5)) - {city}


def test_reset_restores_best_without_resetting_penalties(monkeypatch):
    from moh.problems.tsp_gls import solver

    d = distances()
    calls = []
    # This controlled perturbation produces a worse valid route and lets us
    # observe the iteration-boundary reset independently of local optimality.
    worse = tour_to_route((*range(9), 0))
    original_search = solver.local_search
    monkeypatch.setattr(
        solver,
        "local_search",
        lambda route, matrix, near: (
            original_search(route, matrix, near) if not calls else route
        ),
    )
    monkeypatch.setattr(solver, "two_opt", lambda *args: (0.0, worse.copy()))
    monkeypatch.setattr(solver, "relocate", lambda route, *args: (0.0, route))

    def guide(matrix, tour, penalties):
        calls.append((tour.copy(), penalties.copy()))
        return matrix

    solve_gls(d, guide, GLSOptions(iterations=3, reset_interval=2))
    assert not np.array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][0], calls[2][0])
    assert calls[2][1].sum() == 20


def test_per_city_applies_several_improvements():
    # Seed 3 exercises upstream's immediate traversal: accumulated delta is
    # larger than any single improvement and the returned route reflects both.
    d = distances(3)
    initial = tour_to_route((*range(9), 0))
    delta, result = two_opt(initial, d, neighbors(d), city=0)
    assert delta == pytest.approx(-0.9681751775789693)
    assert route_to_tour(result) == (0, 8, 1, 7, 6, 5, 4, 3, 2, 0)


def test_recorded_upstream_local_search_fixture():
    # Recorded from MoH e8c6154 gls.py (Numba decorators removed), seed 7.
    d = distances(7)
    assert solve_gls(d, lambda d, *_: d, GLSOptions(iterations=0)) == (
        0,
        2,
        3,
        6,
        5,
        7,
        1,
        4,
        8,
        0,
    )
    assert tour_cost(
        d, solve_gls(d, lambda d, *_: d, GLSOptions(iterations=0))
    ) == pytest.approx(3.026072316423462)


def test_guided_edge_selection_zeros_both_directions_and_counts_budget():
    d = distances()
    seen = []

    def guide(matrix, tour, penalties):
        seen.append(penalties.copy())
        matrix[0, 1] += 5
        matrix[1, 0] += 5
        matrix[2, 3] += 3
        matrix[3, 2] += 3
        return matrix

    options = GLSOptions(iterations=2, perturbation_moves=2, edges_per_move=2)
    first = solve_gls(d, guide, options)
    assert len(seen) == 4
    assert seen[1][0, 1] == seen[1][1, 0] == 1
    assert seen[1][2, 3] == seen[1][3, 2] == 1
    assert np.count_nonzero(seen[1]) == 4
    baseline = route_to_tour(
        local_search(tour_to_route(nearest_neighbor(d)), d, neighbors(d))
    )
    assert tour_cost(d, first) <= tour_cost(d, baseline)
    assert solve_gls(d, guide, options) == first
