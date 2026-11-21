from utils.diagnostics import TrajectoryDiagnostics
from utils.diagnostic_prompts import (
    build_diagnostic_directive,
    format_reflective_meta_prompt,
)
from utils.meta_population import OptimizerIndividual


def test_build_diagnostic_directive_empty_or_healthy():
    healthy_diag = TrajectoryDiagnostics(
        valid_rate=0.95,
        improvement_slope=0.25,
        stagnation_steps=0,
        diversity_score=2.0,
    )
    directive = build_diagnostic_directive(healthy_diag)
    assert directive == ""


def test_build_diagnostic_directive_all_triggers():
    troubled_diag = TrajectoryDiagnostics(
        valid_rate=0.50,
        improvement_slope=0.0,
        stagnation_steps=4,
        diversity_score=0.0005,
    )
    directive = build_diagnostic_directive(troubled_diag)
    assert "[CRITICAL - LOW VALIDITY]" in directive
    assert "50%" in directive
    assert "[STAGNATION DETECTED]" in directive
    assert "[PREMATURE CONVERGENCE]" in directive


def test_format_reflective_meta_prompt_with_parent_diag():
    diag = TrajectoryDiagnostics(
        valid_rate=0.60,
        improvement_slope=0.0,
        stagnation_steps=3,
        diversity_score=0.01,
        reflection_summary="Greedy search failed to escape early plateau.",
    )
    parent = OptimizerIndividual(
        opt_id="opt_1",
        code="def solve(): pass",
        thought="seed",
        raw_utility=10.0,
        diagnostics=diag,
    )
    base_prompt = "Task: You should design an efficient metaheuristic..."
    formatted = format_reflective_meta_prompt(base_prompt, parent)

    assert "DIAGNOSTIC FEEDBACK FROM PARENT OPTIMIZER" in formatted
    assert "[CRITICAL - LOW VALIDITY]" in formatted
    assert "[STAGNATION DETECTED]" in formatted
    assert "Greedy search failed to escape early plateau." in formatted
    assert "Task: You should design an efficient metaheuristic..." in formatted


def test_format_reflective_meta_prompt_no_diag_fallback():
    parent = OptimizerIndividual(
        opt_id="opt_clean",
        code="def solve(): pass",
        thought="seed",
        raw_utility=10.0,
        diagnostics=None,
    )
    base_prompt = "Task: You should design an efficient metaheuristic..."
    formatted = format_reflective_meta_prompt(base_prompt, parent)
    assert formatted == base_prompt
