import math
from dataclasses import FrozenInstanceError

import pytest

from moh.core.models import Heuristic, WorkCounts
from moh.core.programs import (
    GapEvaluation,
    OptimizerProgram,
    ProgramEvaluation,
    ScoredProgram,
    TaskOutcome,
)


def scored(identity="a", gap=1.0, source=None):
    candidate = Heuristic(identity, source or f"# {identity}")
    evaluation = GapEvaluation(
        identity,
        "tsp4",
        "validation",
        "success",
        gap,
        (4.0,),
        (gap,),
        ((0, 1, 2, 3, 0),),
        counts=WorkCounts(1, 1, 0),
    )
    return ScoredProgram(candidate, evaluation)


@pytest.mark.parametrize("gap", [math.nan, math.inf, -1.0, True])
def test_invalid_success_score(gap):
    with pytest.raises(ValueError):
        scored(gap=gap)


@pytest.mark.parametrize(
    "changes",
    [
        {"costs": ()},
        {"gaps": (1.0, 2.0)},
        {"tours": ()},
        {"utility": 2.0},
        {"costs": (math.nan,)},
        {"gaps": (True,)},
        {"split": "training"},
        {"error": "oops"},
    ],
)
def test_invalid_gap_record(changes):
    fields = {
        "candidate_id": "a",
        "task_id": "tsp4",
        "split": "validation",
        "status": "success",
        "utility": 1.0,
        "costs": (4.0,),
        "gaps": (1.0,),
        "tours": ((0, 1, 2, 3, 0),),
    }
    with pytest.raises(ValueError):
        GapEvaluation(**(fields | changes))


@pytest.mark.parametrize(
    "utility,error", [(1.0, "oops"), (None, None), (None, ""), (None, "x" * 1025)]
)
def test_invalid_failure(utility, error):
    with pytest.raises(ValueError):
        GapEvaluation("a", "tsp4", "validation", "failed", utility, (), (), (), error)


def test_failed_record_and_tuple_normalization():
    record = GapEvaluation("a", "tsp4", "test", "failed", None, [], [], [], "timeout")
    assert record.costs == record.gaps == record.tours == ()
    assert record.counts == WorkCounts()
    with pytest.raises(FrozenInstanceError):
        record.utility = 3


def test_identity_properties_and_safe_ids():
    item = scored()
    assert (item.id, item.source_code, item.idea, item.utility) == (
        "a",
        "# a",
        None,
        1.0,
    )
    with pytest.raises(ValueError):
        ScoredProgram(Heuristic("b", "# b"), item.evaluation)
    for identity in ("../a", ".", "", "/a", "a/b", "a\\b"):
        with pytest.raises(ValueError):
            scored(identity)
        with pytest.raises(ValueError):
            OptimizerProgram(identity, "# source")


def test_program_evaluation_requires_successful_task_heuristics():
    outcome = TaskOutcome("tsp4", scored())
    record = ProgramEvaluation("opt", "success", 1.0, [outcome])
    item = ScoredProgram(OptimizerProgram("opt", "# optimizer"), record)
    assert item.utility == 1.0
    assert record.task_results == (outcome,)
    with pytest.raises(ValueError):
        TaskOutcome("other", scored())
    for outcomes in ((), (TaskOutcome("tsp4", None),), (outcome, outcome)):
        with pytest.raises(ValueError):
            ProgramEvaluation("opt", "success", 1.0, outcomes)
    with pytest.raises(TypeError):
        ScoredProgram(Heuristic("opt", "# source"), record)


@pytest.mark.parametrize(
    "status,utility,error",
    [
        ("failed", 0, "bad"),
        ("success", math.inf, None),
        ("success", -1, None),
        ("other", None, "bad"),
    ],
)
def test_invalid_program_outcomes(status, utility, error):
    with pytest.raises(ValueError):
        ProgramEvaluation("opt", status, utility, (), error)


def test_failure_can_preserve_partial_task_results_and_counts():
    outcome = TaskOutcome("tsp4", scored())
    counts = WorkCounts(2, 3, 4)
    evaluation = ProgramEvaluation(
        "opt",
        "failed",
        None,
        (outcome, TaskOutcome("tsp5", None)),
        "second task failed",
        counts,
    )
    assert evaluation.task_results[0].selected.utility == 1.0
    assert evaluation.counts == counts


def test_nested_lists_cannot_mutate_evaluation():
    tours = [[0, 1, 2, 3, 0]]
    costs, gaps = [4.0], [1.0]
    record = GapEvaluation("a", "tsp4", "validation", "success", 1, costs, gaps, tours)
    tours[0][1] = 99
    costs[0] = gaps[0] = 99
    assert (record.costs, record.gaps, record.tours) == (
        (4.0,),
        (1.0,),
        ((0, 1, 2, 3, 0),),
    )
