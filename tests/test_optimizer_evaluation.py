import pytest

from moh.core.models import EvaluationResult
from moh.evaluation import OptimizerEvaluator
from moh.llm.fake import FakeLLM
from moh.optimizers.seed_optimizers import initial_optimizer_candidates
from moh.problems.tsp import TSPTask


class TaskRunner:
    def __init__(self, failed=None):
        self.failed = failed
        self.records = []

    def evaluate(self, heuristic, task, context):
        self.records.append((heuristic, task.instances[0].tobytes(), context))
        if task.id == self.failed:
            return EvaluationResult(
                heuristic.id, context, "failed", None, (), "failed", 1
            )
        length = 2.0 if task.id == "tsp10" else 4.0
        return EvaluationResult(
            heuristic.id, context, "success", -length, (length,), None, 1
        )


def evaluator(weights=(1.0, 3.0), failed=None):
    scopes = []

    def factory(scope):
        scopes.append(scope)
        return FakeLLM(42)

    runner = TaskRunner(failed)
    value = OptimizerEvaluator(
        tuple(TSPTask.create(n, 1, 42) for n in (10, 20)),
        weights,
        runner,
        factory,
        1,
        42,
        lambda *_: None,
    )
    return value, runner, scopes


def test_weighted_and_order_independent():
    value, runner, scopes = evaluator()
    a, b = initial_optimizer_candidates(3)
    first = value.evaluate(a)
    value.evaluate(b)
    again = value.evaluate(a)
    assert first == again
    assert first.utility == -3.5
    assert len(scopes) == 6
    assert first.task_results[0].best is not again.task_results[0].best
    assert runner.records[0][0] is not runner.records[8][0]
    assert runner.records[0][1:] == runner.records[8][1:]
    assert first.counts.heuristic_evaluations == 8


def test_failure_and_large_weights():
    a = initial_optimizer_candidates(3)[0]
    assert evaluator(failed="tsp20")[0].evaluate(a).utility is None
    assert evaluator(weights=(1e308, 1e308))[0].evaluate(a).utility == -3.0


@pytest.mark.parametrize(
    "weights",
    [
        (),
        (1.0,),
        (1.0, 2.0, 3.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("inf"), 1.0),
        (float("nan"), 1.0),
        (True, 1.0),
    ],
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        evaluator(weights)
