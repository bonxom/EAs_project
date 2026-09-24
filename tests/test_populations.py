import pytest

from moh.core.populations import Population, rank_key


def test_rank():
    assert rank_key("success", -1000.0, "z") < rank_key("failed", None, "a")
    assert rank_key("success", -2.0, "a") < rank_key("success", -2.0, "b")
    assert rank_key("success", -1.0, "z") < rank_key("success", -2.0, "a")


def test_capacity():
    assert Population(2, [1, 2]).members == (1, 2)
    for capacity, members in [(True, []), (0, []), (1, [1, 2])]:
        with pytest.raises(ValueError):
            Population(capacity, members)
