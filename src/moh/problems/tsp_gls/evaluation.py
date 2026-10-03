"""Trusted minimizing fitness computed from original instance distances."""

import math

from moh.core.programs import TaskOutcome
from moh.execution.optimizer_protocol import number


def gap_percent(cost: float, optimal_cost: float) -> float:
    if not number(cost) or cost <= 0:
        raise ValueError('cost must be positive and finite')
    if not number(optimal_cost) or optimal_cost <= 0:
        raise ValueError('reference cost must be positive and finite')
    tolerance = 1e-8 * max(cost, optimal_cost) + 1e-10
    if cost < optimal_cost - tolerance:
        raise ValueError('reference optimum exceeds the computed tour cost')
    gap = 100 * (max(cost, optimal_cost) - optimal_cost) / optimal_cost
    if not math.isfinite(gap):
        raise ValueError('gap must be finite')
    return gap


def weighted_gap(outcomes: tuple[TaskOutcome, ...], weights: tuple[float, ...]) -> float:
    if not outcomes or len(outcomes) != len(weights):
        raise ValueError('outcomes and weights must have matching nonempty lengths')
    if any(not number(weight) or weight <= 0 for weight in weights):
        raise ValueError('weights must be positive and finite')
    if any(outcome.selected is None or outcome.selected.evaluation.status != 'success'
           for outcome in outcomes):
        raise ValueError('every selected task must succeed')
    # Scale first to avoid overflow when summing large finite weights.
    scale = max(weights)
    normalized = tuple(weight / scale for weight in weights)
    total = math.fsum(normalized)
    return math.fsum(outcome.selected.utility * (weight / total)
                     for outcome, weight in zip(outcomes, normalized, strict=True))
