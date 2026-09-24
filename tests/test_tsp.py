import numpy as np
import pytest

from moh.problems.tsp import TSPTask, tour_length


def test_square():
    assert (
        tour_length(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), [0, 1, 2, 3, 0]
        )
        == 4.0
    )


@pytest.mark.parametrize("size", [10, 20, 50])
def test_instances(size):
    a, b, c = (
        TSPTask.create(size, 3, 42),
        TSPTask.create(size, 3, 42),
        TSPTask.create(size, 3, 43),
    )
    for x, y, z in zip(a.instances, b.instances, c.instances, strict=True):
        np.testing.assert_array_equal(x, y)
        assert not np.array_equal(x, z)
        assert not x.flags.writeable
        assert ((x >= 0) & (x <= 1)).all()


@pytest.mark.parametrize(
    "tour",
    [
        [0, 1, 0],
        [0, 1, 1, 3, 0],
        [1, 0, 2, 3, 1],
        [0, True, 2, 3, 0],
        [0, 1.0, 2, 3, 0],
        [0, 1, 2, 3, 4],
    ],
)
def test_bad_tour(tour):
    with pytest.raises(ValueError):
        tour_length(np.zeros((4, 2)), tour)


@pytest.mark.parametrize("size,count", [(30, 1), (True, 1), (10, 0), (10, True)])
def test_bad_task(size, count):
    with pytest.raises(ValueError):
        TSPTask.create(size, count, 42)


def test_nonfinite():
    with pytest.raises(ValueError):
        tour_length(np.full((4, 2), np.nan), [0, 1, 2, 3, 0])
