"""Checkpoint-driven target heuristic initialization and inference."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from utils.elite_reference import EliteReferenceLLM
from utils.heuristic_evaluator import EvalResult
from utils.population import Pop
from utils.utils import clean_code, extract_code, extract_idea

logger = logging.getLogger(__name__)


class TargetInitializationError(RuntimeError):
    """Raised when a target has no valid seed heuristic."""


@dataclass(frozen=True)
class SelectedHeuristic:
    size: int
    idea: str
    code: str
    validation: EvalResult


def initialize_target_population(
    size,
    elites,
    llm,
    evaluate_candidate,
    heuristic_format,
    population_size=10,
    seed_count=5,
    max_attempts=25,
    seed=0,
    reference_count=2,
) -> Pop:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if seed_count <= 0 or seed_count > population_size:
        raise ValueError("seed_count must be positive and no larger than population_size")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    task = f"tsp_gls-{int(size)}"
    population = Pop([task], population_size, allow_equal_utility=True)
    reference_llm = EliteReferenceLLM(
        llm,
        elites,
        size,
        seed=seed,
        reference_count=reference_count,
    )
    expertise = (
        "You are an expert in optimization heuristics and combinatorial "
        "optimization. Design a deterministic target heuristic."
    )
    message = (
        f"Generate one heuristic for TSP size {size}. Follow this function format:\n"
        f"{heuristic_format}"
    )
    attempts = 0
    for attempt in range(max_attempts):
        attempts = attempt + 1
        try:
            response = reference_llm.prompt(expertise, message)
            extracted = extract_code(response)
            if not extracted:
                continue
            code = response if response.strip() == extracted else extracted
            canonical_code = clean_code(code)
            if population.check_solution(task, canonical_code):
                continue
            compile(code, f"<target-seed-{size}-{attempt}>", "exec")
            idea = extract_idea(response) or f"target seed {attempt + 1}"
            result = evaluate_candidate(code, idea)
            if (
                result.status != "ok"
                or result.mean_gap_pct is None
                or not math.isfinite(result.mean_gap_pct)
            ):
                continue
            population.save_solution(task, idea, code, result.mean_gap_pct)
            if population.get_subtask_size(task) >= seed_count:
                break
        except Exception as error:  # noqa: BLE001 -- candidate isolation boundary
            logger.warning("Target %s seed attempt %s failed: %s", size, attempt, error)
            continue

    accepted = population.get_subtask_size(task)
    population.initialization = {
        "requested": seed_count,
        "accepted": accepted,
        "attempts": attempts,
        "partial": 0 < accepted < seed_count,
    }
    if accepted == 0:
        raise TargetInitializationError(
            f"Target {size} has no valid seed after {max_attempts} attempts"
        )
    return population
