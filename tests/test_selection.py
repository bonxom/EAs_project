import random

import pytest

from moh.core.models import (
    EvaluationContext,
    EvaluationResult,
    Heuristic,
    ScoredHeuristic,
)
from moh.optimizers.selection import select_parents, select_survivors


def scored(identity, source, utility):
    return ScoredHeuristic(
        Heuristic(identity, source),
        EvaluationResult(
            identity,
            EvaluationContext("t", (1,), (2,)),
            "success" if utility is not None else "failed",
            utility,
            (-utility,) if utility is not None else (),
            None if utility is not None else "bad",
            1,
        ),
    )


def members():
    return [scored("a", "A", -1.0), scored("b", "A", -2.0), scored("c", "B", -3.0)]


def test_survivors():
    assert [x.heuristic.id for x in select_survivors(members(), 2, "diversity")] == [
        "a",
        "c",
    ]
    assert [x.heuristic.id for x in select_survivors(members(), 3, "diversity")] == [
        "a",
        "b",
        "c",
    ]
    assert [x.heuristic.id for x in select_survivors(members(), 2, "elitist")] == [
        "a",
        "b",
    ]


@pytest.mark.parametrize("policy", ["best", "random", "tournament"])
def test_distinct_parents(policy):
    result = select_parents(members(), policy, 2, random.Random(42))
    assert len({x.heuristic.id for x in result}) == 2
    assert result == select_parents(members(), policy, 2, random.Random(42))
    if policy == "best":
        assert [x.heuristic.id for x in result] == ["a", "b"]


def test_invalid():
    for values, policy, count in [
        (members(), "bad", 2),
        (members(), "best", 4),
        ([members()[0]] * 2, "random", 2),
    ]:
        with pytest.raises(ValueError):
            select_parents(values, policy, count, random.Random(42))
    with pytest.raises(ValueError):
        select_survivors(members(), 0, "elitist")
