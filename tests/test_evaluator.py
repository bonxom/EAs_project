"""Unit tests for M3 Evaluator Contract and Inner Search Tracker."""

import pytest

from moh.optimizers.evaluator import (
    CandidateEvaluationResult,
    FakeEvaluator,
    InnerSearchResult,
    InnerSearchTracker,
)


def test_candidate_evaluation_result_valid():
    res = CandidateEvaluationResult(
        candidate_id="req-001",
        source_code="def solve(): return 42",
        score=12.5,
        valid=True,
    )
    assert res.candidate_id == "req-001"
    assert res.source_code == "def solve(): return 42"
    assert res.score == 12.5
    assert isinstance(res.score, float)
    assert res.valid is True
    assert res.error is None


def test_candidate_evaluation_result_invalid():
    res = CandidateEvaluationResult(
        candidate_id="req-002",
        source_code="def solve(): raise Error()",
        score=None,
        valid=False,
        error="evaluator_failed: RuntimeError",
    )
    assert res.candidate_id == "req-002"
    assert res.score is None
    assert res.valid is False
    assert res.error == "evaluator_failed: RuntimeError"


@pytest.mark.parametrize(
    "bad_score",
    [float("nan"), float("inf"), float("-inf"), True, False, "1.0", [1.0]],
)
def test_candidate_evaluation_result_rejects_bad_scores(bad_score):
    with pytest.raises((ValueError, TypeError)):
        CandidateEvaluationResult(
            candidate_id="req-003",
            source_code="def f(): pass",
            score=bad_score,
            valid=True,
        )


def test_inner_search_result_valid_utility():
    res = InnerSearchResult(
        best_candidate_id="req-002",
        best_score=0.85,
        utility=0.85,
        evaluations=(),
        generated_count=2,
        evaluated_count=2,
        valid_evaluation_count=2,
        invalid_evaluation_count=0,
    )
    assert res.best_candidate_id == "req-002"
    assert res.best_score == 0.85
    assert res.utility == 0.85


def test_inner_search_result_empty():
    res = InnerSearchResult(
        best_candidate_id=None,
        best_score=None,
        utility=None,
        evaluations=(),
        generated_count=0,
        evaluated_count=0,
        valid_evaluation_count=0,
        invalid_evaluation_count=0,
    )
    assert res.best_candidate_id is None
    assert res.best_score is None
    assert res.utility is None


def test_inner_search_result_mismatched_utility_raises():
    with pytest.raises(ValueError):
        InnerSearchResult(
            best_candidate_id="req-001",
            best_score=0.85,
            utility=0.0,  # Must match best_score
            evaluations=(),
            generated_count=1,
            evaluated_count=1,
            valid_evaluation_count=1,
            invalid_evaluation_count=0,
        )


def test_fake_evaluator_dict():
    evaluator = FakeEvaluator({"code_a": 0.5, "code_b": 0.9}, default_score=0.1)
    assert evaluator("code_a") == 0.5
    assert evaluator("code_b") == 0.9
    assert evaluator("code_c") == 0.1


def test_fake_evaluator_list():
    evaluator = FakeEvaluator([10.0, 20.0], default_score=-1.0)
    assert evaluator("any1") == 10.0
    assert evaluator("any2") == 20.0
    assert evaluator("any3") == -1.0


def test_fake_evaluator_exception():
    evaluator = FakeEvaluator({"bad": ValueError("eval error")})
    with pytest.raises(ValueError, match="eval error"):
        evaluator("bad")


def test_tracker_best_so_far_sequence():
    fake = FakeEvaluator({"code_a": 0.2, "code_b": 0.7, "code_c": 0.4})
    tracker = InnerSearchTracker(fake)

    assert tracker.evaluate_candidate("req-001", "code_a") == 0.2
    assert tracker.evaluate_candidate("req-002", "code_b") == 0.7
    assert tracker.evaluate_candidate("req-003", "code_c") == 0.4

    res = tracker.get_result(generated_count=3, evaluated_count=3)
    assert res.best_candidate_id == "req-002"
    assert res.best_score == 0.7
    assert res.utility == 0.7
    assert res.valid_evaluation_count == 3
    assert res.invalid_evaluation_count == 0


def test_tracker_negative_scores():
    fake = FakeEvaluator({"code_a": -10.0, "code_b": -3.0, "code_c": -7.0})
    tracker = InnerSearchTracker(fake)

    tracker.evaluate_candidate("req-001", "code_a")
    tracker.evaluate_candidate("req-002", "code_b")
    tracker.evaluate_candidate("req-003", "code_c")

    res = tracker.get_result(generated_count=3, evaluated_count=3)
    assert res.best_candidate_id == "req-002"
    assert res.best_score == -3.0
    assert res.utility == -3.0


def test_tracker_first_wins_ties():
    fake = FakeEvaluator({"code_a": 0.7, "code_b": 0.7, "code_c": 0.5})
    tracker = InnerSearchTracker(fake)

    tracker.evaluate_candidate("req-001", "code_a")
    tracker.evaluate_candidate("req-002", "code_b")
    tracker.evaluate_candidate("req-003", "code_c")

    res = tracker.get_result(generated_count=3, evaluated_count=3)
    assert res.best_candidate_id == "req-001"  # First-wins
    assert res.best_score == 0.7


def test_tracker_invalid_result_does_not_affect_best():
    fake = FakeEvaluator({"code_bad": ValueError("crash"), "code_good": 0.4})
    tracker = InnerSearchTracker(fake)

    with pytest.raises(ValueError):
        tracker.evaluate_candidate("req-001", "code_bad")

    tracker.evaluate_candidate("req-002", "code_good")

    res = tracker.get_result(generated_count=2, evaluated_count=2)
    assert res.best_candidate_id == "req-002"
    assert res.best_score == 0.4
    assert res.valid_evaluation_count == 1
    assert res.invalid_evaluation_count == 1


def test_tracker_all_invalid():
    fake = FakeEvaluator({"code_a": float("nan"), "code_b": float("inf")})
    tracker = InnerSearchTracker(fake)

    with pytest.raises(ValueError):
        tracker.evaluate_candidate("req-001", "code_a")

    with pytest.raises(ValueError):
        tracker.evaluate_candidate("req-002", "code_b")

    res = tracker.get_result(generated_count=2, evaluated_count=2)
    assert res.best_candidate_id is None
    assert res.best_score is None
    assert res.utility is None
    assert res.valid_evaluation_count == 0
    assert res.invalid_evaluation_count == 2


def test_tracker_zero_evaluations():
    fake = FakeEvaluator()
    tracker = InnerSearchTracker(fake)
    res = tracker.get_result(generated_count=1, evaluated_count=0)

    assert res.best_candidate_id is None
    assert res.best_score is None
    assert res.utility is None
    assert res.generated_count == 1
    assert res.evaluated_count == 0
    assert res.valid_evaluation_count == 0
    assert res.invalid_evaluation_count == 0
