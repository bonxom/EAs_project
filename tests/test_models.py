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


def test_context_copies_seed_sequences():
    instances, workers = [1], [2]
    context = EvaluationContext("t", instances, workers)
    instances.append(3)
    workers.append(4)
    assert context.instance_seeds == (1,)
    assert context.worker_seeds == (2,)


def test_result_copies_lengths():
    lengths = [1.0]
    result = EvaluationResult(
        "h1", EvaluationContext("t", (1,), (2,)), "success", -1.0, lengths, None, 1
    )
    lengths.append(2.0)
    assert result.lengths == (1.0,)


def test_search_records_copy_sequences(base_spec):
    from moh.core.models import (
        InnerResult,
        OptimizerCandidate,
        OptimizerEvaluation,
        RunResult,
        ScoredOptimizer,
        TaskResult,
        WorkCounts,
    )
    from moh.core.specs import OptimizerSpec

    heuristic = ScoredHeuristic(
        Heuristic("h1", "source"),
        EvaluationResult(
            "h1", EvaluationContext("t", (1,), (2,)), "success", -1.0, (1.0,), None, 1
        ),
    )
    population = [heuristic]
    inner = InnerResult(population, WorkCounts())
    population.clear()
    assert inner.population == (heuristic,)
    tasks = [TaskResult("t", heuristic, WorkCounts())]
    evaluation = OptimizerEvaluation("o1", "success", -1.0, tasks, WorkCounts(), None)
    tasks.clear()
    assert len(evaluation.task_results) == 1
    candidate = ScoredOptimizer(
        OptimizerCandidate("o1", OptimizerSpec(**base_spec)), evaluation
    )
    population = [candidate]
    result = RunResult("success", candidate, population)
    population.clear()
    assert result.population == (candidate,)
