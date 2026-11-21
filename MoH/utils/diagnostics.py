"""Trajectory Diagnostics and Reflection for R-MoH Inner-Loop."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryDiagnostics:
    best_score: float = 1e6
    valid_rate: float = 0.0
    improvement_slope: float = 0.0
    stagnation_steps: int = 0
    diversity_score: float = 0.0
    failure_modes: list[str] = field(default_factory=list)
    reflection_summary: str = ""


class TrajectoryCollector:
    """Collects heuristic evaluation trajectory during an optimizer's inner loop."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record_evaluation(
        self,
        code: str,
        score: float,
        is_valid: bool,
        error: str | None = None,
        idea: str = "",
    ) -> None:
        self.records.append(
            {
                "code": code,
                "score": float(score) if math.isfinite(score) else 1e6,
                "is_valid": bool(is_valid),
                "error": str(error) if error else None,
                "idea": str(idea),
            }
        )

    def _extract_failure_mode(self, error_str: str | None) -> str | None:
        if not error_str:
            return None
        for category in (
            "SyntaxError",
            "TimeoutError",
            "NameError",
            "TypeError",
            "ValueError",
            "ZeroDivisionError",
            "IndexError",
            "MemoryError",
            "AttributeError",
        ):
            if category in error_str:
                return category
        return "RuntimeError"

    def finalize(self) -> TrajectoryDiagnostics:
        if not self.records:
            return TrajectoryDiagnostics()

        total_evals = len(self.records)
        valid_records = [r for r in self.records if r["is_valid"]]
        valid_count = len(valid_records)
        valid_rate = valid_count / total_evals if total_evals > 0 else 0.0

        # Best score tracking across all evaluations
        best_so_far_series: list[float] = []
        cur_best = 1e6
        last_improved_step = 0

        for idx, r in enumerate(self.records):
            if r["is_valid"] and r["score"] < cur_best:
                cur_best = r["score"]
                last_improved_step = idx
            best_so_far_series.append(cur_best)

        best_score = cur_best

        # Improvement slope: (initial best - final best) / steps (positive when improving)
        if total_evals > 1:
            initial_best = best_so_far_series[0]
            final_best = best_so_far_series[-1]
            if initial_best < 1e5 and final_best < 1e5:
                improvement_slope = (initial_best - final_best) / (total_evals - 1)
            else:
                improvement_slope = 0.0
        else:
            improvement_slope = 0.0

        # Stagnation steps: consecutive evaluations at the end without improving best score
        stagnation_steps = (total_evals - 1 - last_improved_step) if valid_count > 0 else total_evals

        # Diversity score: std of scores for valid solutions
        if len(valid_records) >= 2:
            scores = [r["score"] for r in valid_records if r["score"] < 1e5]
            diversity_score = float(np.std(scores)) if len(scores) >= 2 else 0.0
        else:
            diversity_score = 0.0

        # Failure modes: distinct categorized errors
        failure_modes: list[str] = []
        for r in self.records:
            cat = self._extract_failure_mode(r["error"])
            if cat and cat not in failure_modes:
                failure_modes.append(cat)

        return TrajectoryDiagnostics(
            best_score=best_score,
            valid_rate=valid_rate,
            improvement_slope=improvement_slope,
            stagnation_steps=stagnation_steps,
            diversity_score=diversity_score,
            failure_modes=failure_modes,
            reflection_summary="",
        )


def generate_reflection_summary(
    llm: Any,
    optimizer_code: str,
    best_heuristics: list[tuple[str, str, float]],
    worst_heuristics: list[tuple[str, str, float]],
    max_words: int = 60,
) -> str:
    """Generate a single-shot concise reflection summary diagnosing optimizer stagnation."""
    role = (
        "You are an expert metaheuristics researcher diagnosing algorithm stagnation. "
        f"Provide your answer in at most {max_words} words."
    )

    best_desc = "\n".join(
        f"- [Score: {score:.4f}] Idea: {idea}" for idea, _, score in best_heuristics[:2]
    )
    worst_desc = "\n".join(
        f"- [Score: {score:.4f}] Idea: {idea}" for idea, _, score in worst_heuristics[:2]
    )

    optimizer_snippet = optimizer_code[:1200]
    message = (
        f"The following Heuristic-Optimizer code was run to optimize heuristics:\n"
        f"```python\n{optimizer_snippet}\n```\n\n"
        f"Top performing heuristics produced:\n{best_desc or 'None'}\n\n"
        f"Lowest performing / failed heuristics:\n{worst_desc or 'None'}\n\n"
        f"In at most {max_words} words, explain the core algorithmic bottleneck "
        f"of this optimizer and why it stopped improving."
    )

    try:
        response = llm.prompt(role, message, temperature=0.5)
        # Enforce max words boundary
        words = response.strip().split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return " ".join(words)
    except Exception as e:
        logger.warning(f"Failed to generate reflection summary: {e}")
        return ""
