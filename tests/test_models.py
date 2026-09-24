import pytest

from moh.core.models import (
    EvaluationContext,
    EvaluationResult,
    Heuristic,
    ScoredHeuristic,
)


@pytest.mark.parametrize(
    "status,utility,lengths,error",
    [
        ("success", float("nan"), (1.0,), None),
        ("failed", -1.0, (), "bad"),
        ("success", -1.0, (), None),
        ("failed", None, (), None),
    ],
)
def test_invalid_result(status, utility, lengths, error):
    with pytest.raises(ValueError):
        EvaluationResult(
            "h1", EvaluationContext("t", (1,), (2,)), status, utility, lengths, error, 1
        )


def test_scored_identity():
    result = EvaluationResult(
        "h1", EvaluationContext("t", (1,), (2,)), "success", -1.0, (1.0,), None, 1
    )
    with pytest.raises(ValueError):
        ScoredHeuristic(Heuristic("h2", "code"), result)
