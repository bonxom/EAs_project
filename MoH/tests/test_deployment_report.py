import csv
import hashlib
import json

from deployment_fakes import HEURISTIC, OPTIMIZER, FakeEvaluator
from inference import SelectedHeuristic
from utils.checkpoint import TrainingCheckpoint


def _checkpoint():
    return TrainingCheckpoint(OPTIMIZER, (), {})


def _selected(evaluator, size=500, score=2.0, metadata=None):
    evaluator.scores[HEURISTIC] = score
    validation = evaluator.evaluate(HEURISTIC, size, "val")
    return SelectedHeuristic(
        size, "chosen", HEURISTIC, validation, metadata or {}
    )


def test_test_scores_never_change_validation_selection(tmp_path):
    from utils.deployment_report import evaluate_selected_heuristics

    evaluator = FakeEvaluator({HEURISTIC: 2.0})
    selected = _selected(evaluator)
    validation = selected.validation
    rows = evaluate_selected_heuristics(
        [selected], evaluator, tmp_path, _checkpoint()
    )
    assert evaluator.calls[-1] == (HEURISTIC, 500, "test")
    assert selected.code == HEURISTIC
    assert rows[0]["validation_gap_pct"] == validation.mean_gap_pct


def test_test_infrastructure_failure_is_reported(tmp_path):
    from utils.deployment_report import evaluate_selected_heuristics

    evaluator = FakeEvaluator({HEURISTIC: 2.0})
    selected = _selected(evaluator)
    evaluator.failed_codes.add(HEURISTIC)
    rows = evaluate_selected_heuristics(
        [selected], evaluator, tmp_path, _checkpoint()
    )
    assert rows[0]["status"] == "test_failed"
    assert rows[0]["test_gap_pct"] is None
    assert rows[0]["error"] == "fake failure"


def test_selected_artifact_hash_matches_evaluated_code(tmp_path):
    from utils.deployment_report import evaluate_selected_heuristics

    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    selected = _selected(evaluator)
    rows = evaluate_selected_heuristics(
        [selected], evaluator, tmp_path, _checkpoint()
    )
    code_path = tmp_path / "heuristics" / "tsp500.py"
    assert code_path.read_text() == HEURISTIC
    assert rows[0]["heuristic_sha256"] == hashlib.sha256(
        code_path.read_bytes()
    ).hexdigest()
    assert evaluator.calls[-1][0] == code_path.read_text()


def test_report_records_actual_instance_count(tmp_path):
    from utils.deployment_report import evaluate_selected_heuristics

    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    evaluator.val_count = 7
    evaluator.test_count = 13
    rows = evaluate_selected_heuristics(
        [_selected(evaluator)], evaluator, tmp_path, _checkpoint()
    )
    assert rows[0]["val_n_instances"] == 7
    assert rows[0]["test_n_instances"] == 13


def test_failed_target_does_not_hide_successful_target(tmp_path):
    from utils.deployment_report import evaluate_selected_heuristics

    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    rows = evaluate_selected_heuristics(
        [_selected(evaluator, 500)],
        evaluator,
        tmp_path,
        _checkpoint(),
        target_errors={1000: "initialization exhausted"},
    )
    assert [(row["size"], row["status"]) for row in rows] == [
        (500, "ok"),
        (1000, "inference_failed"),
    ]


def test_json_and_csv_have_matching_values(tmp_path):
    from utils.deployment_report import evaluate_selected_heuristics, write_report

    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    selected = _selected(
        evaluator,
        metadata={
            "iterations_completed": 3,
            "evaluation_count": 4,
            "early_stop_reason": None,
        },
    )
    rows = evaluate_selected_heuristics(
        [selected], evaluator, tmp_path, _checkpoint()
    )
    write_report(rows, tmp_path)
    json_row = json.loads((tmp_path / "report.json").read_text())[0]
    with (tmp_path / "report.csv").open(newline="") as source:
        csv_row = next(csv.DictReader(source))
    for field in ("size", "status", "validation_gap_pct", "test_gap_pct"):
        assert str(json_row[field]) == csv_row[field]
    assert json_row["selection_metadata"]["iterations_completed"] == 3


def test_validation_only_report_never_evaluates_test(tmp_path):
    from utils.deployment_report import evaluate_selected_heuristics

    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    selected = _selected(evaluator)
    calls_before = list(evaluator.calls)
    rows = evaluate_selected_heuristics(
        [selected],
        evaluator,
        tmp_path,
        _checkpoint(),
        evaluate_test=False,
    )
    assert evaluator.calls == calls_before
    assert rows[0]["status"] == "validation_only"
    assert rows[0]["test_gap_pct"] is None
    assert rows[0]["test_n_instances"] == 0


def test_partial_solver_failures_remain_visible(tmp_path):
    from dataclasses import replace

    from utils.deployment_report import evaluate_selected_heuristics

    evaluator = FakeEvaluator({HEURISTIC: 1e6})
    selected = _selected(evaluator, score=1e6)
    original_evaluate = evaluator.evaluate

    def evaluate(code, size, mode):
        result = original_evaluate(code, size, mode)
        return replace(result, failures=3)

    evaluator.evaluate = evaluate
    rows = evaluate_selected_heuristics(
        [selected], evaluator, tmp_path, _checkpoint()
    )
    assert rows[0]["test_failures"] == 3
    assert rows[0]["test_gap_pct"] == 1e6
