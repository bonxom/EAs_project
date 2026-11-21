"""Diagnostic-Guided Self-Invocation Prompting for R-MoH."""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.diagnostics import TrajectoryDiagnostics

if TYPE_CHECKING:
    from utils.meta_population import OptimizerIndividual


def build_diagnostic_directive(diag: TrajectoryDiagnostics | None) -> str:
    """Generate targeted diagnostic directives based on optimizer trajectory metrics."""
    if diag is None:
        return ""

    directives = []
    if diag.valid_rate <= 0.70:
        directives.append(
            f"- [CRITICAL - LOW VALIDITY]: The parent optimizer frequently generates invalid or timed-out heuristics "
            f"(Valid rate: {diag.valid_rate:.0%}). Simplify the code generation template, enforce strict syntax checks, "
            "and avoid nested loops over all node pairs in the prompt instructions."
        )
    if diag.stagnation_steps >= 3 or abs(diag.improvement_slope) <= 1e-4:
        directives.append(
            "- [STAGNATION DETECTED]: The inner evolution plateaued early. Inject stronger exploration mechanisms "
            "(e.g., adaptive mutation rates, structural perturbation, or multi-criteria scoring combining distance and capacity)."
        )
    if diag.diversity_score <= 1e-3:
        directives.append(
            "- [PREMATURE CONVERGENCE]: The inner population collapsed into identical heuristics. Modify the selection/crossover "
            "logic of the Heuristic-Optimizer to maintain niche diversity."
        )

    return "\n".join(directives)


def format_reflective_meta_prompt(base_prompt: str, parent: OptimizerIndividual | None) -> str:
    """Inject diagnostic feedback and reflection summary into the self-invocation prompt."""
    if parent is None or parent.diagnostics is None:
        return base_prompt

    diag = parent.diagnostics
    directives = build_diagnostic_directive(diag)
    reflection = diag.reflection_summary.strip()

    feedback_parts = []
    if directives:
        feedback_parts.append("Key Weaknesses Identified:\n" + directives)
    if reflection:
        feedback_parts.append(f"Reflection Summary: {reflection}")

    if not feedback_parts:
        return base_prompt

    feedback_block = (
        "### DIAGNOSTIC FEEDBACK FROM PARENT OPTIMIZER:\n"
        + "\n\n".join(feedback_parts)
        + "\n\n"
        "Please address these weaknesses when designing the improved optimizer.\n\n"
    )

    return feedback_block + base_prompt
