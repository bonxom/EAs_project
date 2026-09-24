import json

import pytest
from test_selection import scored

from moh.core.models import OptimizerEvaluation, TaskResult, WorkCounts
from moh.experiment import run_outer_loop
from moh.llm.fake import FakeLLM


class Evaluator:
    def __init__(self, utilities):
        self.utilities = iter(utilities)
        self.ids = []

    def evaluate(self, candidate):
        self.ids.append(candidate.id)
        utility = next(self.utilities)
        best = scored("h1", "code", utility) if utility is not None else None
        return OptimizerEvaluation(
            candidate.id,
            "success" if best else "failed",
            utility,
            (TaskResult("t", best, WorkCounts()),),
            WorkCounts(),
            None if best else "bad",
        )


def run(utilities, iterations, fake=None):
    evaluator = Evaluator(utilities)
    events = []
    result = run_outer_loop(
        evaluator=evaluator,
        llm=fake or FakeLLM(42),
        iterations=iterations,
        outer_capacity=2,
        initial_inner_capacity=3,
        emit=lambda e, p: events.append((e, p)),
    )
    return result, evaluator, events


def test_last_child_evaluated():
    result, evaluator, events = run([-5.0, -4.0, -1.0], 1)
    assert result.winner.candidate.id == "o000003"
    assert evaluator.ids == ["o000001", "o000002", "o000003"]
    assert [p["ids"] for k, p in events if k == "population_updated"][-1] == [
        "o000003",
        "o000002",
    ]


@pytest.mark.parametrize(
    "utilities,iterations,winner,status",
    [
        ([-5.0, -4.0], 0, "o000002", "success"),
        ([-5.0, -4.0, None], 1, "o000002", "success"),
        ([None, None, None], 1, None, "failed"),
        ([-1.0, -1.0, -1.0], 1, "o000001", "success"),
    ],
)
def test_boundaries(utilities, iterations, winner, status):
    result, _, _ = run(utilities, iterations)
    assert result.status == status
    assert (result.winner.candidate.id if result.winner else None) == winner


def test_invalid_proposal_and_identical_specs(base_spec):
    _result, evaluator, events = run(
        [-5.0, -4.0], 1, FakeLLM(42, {"optimizer_spec": ("invalid",)})
    )
    assert len(evaluator.ids) == 2
    assert any(k == "generation_failed" for k, p in events)
    _, evaluator, _ = run(
        [-5.0, -4.0, -3.0, -2.0],
        2,
        FakeLLM(42, {"optimizer_spec": (json.dumps(base_spec),) * 2}),
    )
    assert len(evaluator.ids) == 4
