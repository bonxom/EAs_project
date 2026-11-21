from unittest.mock import Mock

import pytest
from utils.diagnostics import (
    TrajectoryCollector,
    TrajectoryDiagnostics,
    generate_reflection_summary,
)


def test_trajectory_diagnostics_dataclass():
    diag = TrajectoryDiagnostics(
        best_score=5.2,
        valid_rate=0.85,
        improvement_slope=0.3,
        stagnation_steps=2,
        diversity_score=1.5,
        failure_modes=["SyntaxError", "TimeoutError"],
        reflection_summary="Optimizer converged quickly but got stuck in local minima.",
    )
    assert diag.best_score == 5.2
    assert diag.valid_rate == 0.85
    assert diag.improvement_slope == 0.3
    assert diag.stagnation_steps == 2
    assert diag.diversity_score == 1.5
    assert len(diag.failure_modes) == 2
    assert "local minima" in diag.reflection_summary


def test_collector_empty_fallback():
    collector = TrajectoryCollector()
    diag = collector.finalize()
    assert diag.best_score == 1e6
    assert diag.valid_rate == 0.0
    assert diag.improvement_slope == 0.0
    assert diag.stagnation_steps == 0
    assert diag.diversity_score == 0.0
    assert diag.failure_modes == []
    assert diag.reflection_summary == ""


def test_collector_validity_and_metrics():
    collector = TrajectoryCollector()

    # Step 1: valid heuristic, initial score 10.0
    collector.record_evaluation(
        code="def solve(): return 10",
        score=10.0,
        is_valid=True,
    )
    # Step 2: invalid syntax
    collector.record_evaluation(
        code="def invalid(:",
        score=1e6,
        is_valid=False,
        error="SyntaxError: invalid syntax",
    )
    # Step 3: valid heuristic, improves to 7.0
    collector.record_evaluation(
        code="def solve(): return 7",
        score=7.0,
        is_valid=True,
    )
    # Step 4: valid heuristic, no improvement (8.0)
    collector.record_evaluation(
        code="def solve(): return 8",
        score=8.0,
        is_valid=True,
    )
    # Step 5: timeout
    collector.record_evaluation(
        code="def solve(): while True: pass",
        score=1e6,
        is_valid=False,
        error="TimeoutError: execution exceeded 10s",
    )

    diag = collector.finalize()

    # 3 valid out of 5 total => 60%
    assert diag.valid_rate == pytest.approx(0.6)
    assert diag.best_score == 7.0
    # First best was 10.0, last best was 7.0 over 5 evals => (10.0 - 7.0) / 4 = 0.75
    assert diag.improvement_slope == pytest.approx(0.75)
    # Last improvement was at step 3. Steps 4 and 5 did not improve => stagnation = 2
    assert diag.stagnation_steps == 2
    # Diversity score is std dev of valid scores [10.0, 7.0, 8.0]
    assert diag.diversity_score > 0.0
    # Failure modes should capture SyntaxError and TimeoutError
    assert "SyntaxError" in diag.failure_modes
    assert "TimeoutError" in diag.failure_modes


def test_generate_reflection_summary_with_llm():
    mock_llm = Mock()
    mock_llm.prompt.return_value = (
        "The optimizer repeatedly generated greedy nearest neighbor variations that failed "
        "to escape local optima due to lack of perturbation mechanisms."
    )

    optimizer_code = "def improve_algorithm(...): pass"
    best_heuristics = [("best idea", "def h1(): pass", 5.0)]
    worst_heuristics = [("bad idea", "def h2(): pass", 1e6)]

    summary = generate_reflection_summary(
        mock_llm,
        optimizer_code=optimizer_code,
        best_heuristics=best_heuristics,
        worst_heuristics=worst_heuristics,
        max_words=60,
    )

    assert mock_llm.prompt.called
    assert "local optima" in summary
    assert len(summary.split()) <= 60
