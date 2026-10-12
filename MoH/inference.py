"""Checkpoint-driven target heuristic initialization and inference."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from utils.elite_reference import EliteReferenceLLM
from utils.generated_improver import GeneratedImprover
from utils.heuristic_evaluator import EvalResult
from utils.population import Pop
from utils.utils import clean_code, extract_code, extract_idea

logger = logging.getLogger(__name__)


class TargetInitializationError(RuntimeError):
    """Raised when a target has no valid seed heuristic."""


class _BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class SelectedHeuristic:
    size: int
    idea: str
    code: str
    validation: EvalResult
    metadata: dict = field(default_factory=dict)


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
        except _BudgetExhausted:
            raise
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


class InferenceRunner:
    def __init__(
        self,
        checkpoint,
        llm,
        evaluator,
        output_dir,
        heuristic_format,
        iterations=10,
        population_size=10,
        seed_count=5,
        initialization_attempts=25,
        reference_count=2,
        max_evaluations=1000,
        optimizer_timeout=3600,
        seed=0,
        program_factory=GeneratedImprover,
    ):
        positive = {
            "population_size": population_size,
            "seed_count": seed_count,
            "initialization_attempts": initialization_attempts,
            "max_evaluations": max_evaluations,
            "optimizer_timeout": optimizer_timeout,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Positive values required for: {', '.join(invalid)}")
        if iterations < 0:
            raise ValueError("iterations cannot be negative")
        if reference_count < 0:
            raise ValueError("reference_count cannot be negative")
        if seed_count > population_size:
            raise ValueError("seed_count cannot exceed population_size")
        self.checkpoint = checkpoint
        self.llm = llm
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.heuristic_format = heuristic_format
        self.iterations = int(iterations)
        self.population_size = int(population_size)
        self.seed_count = int(seed_count)
        self.initialization_attempts = int(initialization_attempts)
        self.reference_count = int(reference_count)
        self.max_evaluations = int(max_evaluations)
        self.optimizer_timeout = optimizer_timeout
        self.seed = int(seed)
        self.program_factory = program_factory

    def run_target(self, size: int) -> SelectedHeuristic:
        size = int(size)
        task = f"tsp_gls-{size}"
        manifest = self.evaluator.preflight([size], include_test=False)[size]["val"]
        dataset_digest = manifest["dataset_sha256"]
        solver_key = json.dumps(
            self.evaluator.solver_settings, sort_keys=True, separators=(",", ":")
        )
        cache = {}
        evaluated = {}
        evaluation_count = 0
        population = None
        iteration_errors = []
        snapshot_paths = []
        early_stop_reason = None

        def cache_key(code):
            code_digest = hashlib.sha256(code.encode()).hexdigest()
            return (
                size,
                "val",
                code_digest,
                self.evaluator.seed,
                dataset_digest,
                solver_key,
            )

        def evaluate_candidate(code, idea):
            nonlocal evaluation_count
            key = cache_key(code)
            if key in cache:
                result = cache[key]
            else:
                if evaluation_count >= self.max_evaluations:
                    raise _BudgetExhausted(
                        f"Target {size} exhausted {self.max_evaluations} evaluations"
                    )
                evaluation_count += 1
                result = self.evaluator.evaluate(code, size, "val")
                cache[key] = result
            if (
                result.status == "ok"
                and result.mean_gap_pct is not None
                and math.isfinite(result.mean_gap_pct)
            ):
                digest = key[2]
                evaluated.setdefault(digest, (idea or "generated", code, result))
                if population is not None:
                    population.save_solution(
                        task, idea or "generated", code, result.mean_gap_pct
                    )
            return result

        try:
            population = initialize_target_population(
                size,
                self.checkpoint.elites,
                self.llm,
                evaluate_candidate,
                self.heuristic_format,
                population_size=self.population_size,
                seed_count=self.seed_count,
                max_attempts=self.initialization_attempts,
                seed=self.seed,
                reference_count=self.reference_count,
            )
        except _BudgetExhausted as error:
            raise TargetInitializationError(str(error)) from error

        reference_llm = EliteReferenceLLM(
            self.llm,
            self.checkpoint.elites,
            size,
            seed=self.seed,
            reference_count=self.reference_count,
        )
        target_dir = self.output_dir / "populations" / f"tsp{size}"
        target_dir.mkdir(parents=True, exist_ok=True)
        iterations_completed = 0

        def utility(code, idea=None, requested_task=None):
            if requested_task is not None and requested_task != task:
                raise ValueError(
                    f"Utility request target {requested_task!r} does not match {task!r}"
                )
            result = evaluate_candidate(code, idea)
            if (
                result.status != "ok"
                or result.mean_gap_pct is None
                or not math.isfinite(result.mean_gap_pct)
            ):
                return 1e6
            return result.mean_gap_pct

        def save_iteration_snapshot(iteration, error=None):
            best = population.get_best_solution(task)
            snapshot = {
                "iteration": iteration,
                "evaluation_count": evaluation_count,
                "best_validation_gap_pct": best["utility"],
                "best_code_sha256": hashlib.sha256(
                    best["best_sol"].encode()
                ).hexdigest(),
                "population": population.get_population(task),
                "error": error,
            }
            path = target_dir / f"iter_{iteration:03d}.json"
            path.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            snapshot_paths.append(str(path))

        for iteration in range(self.iterations):
            child_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.seed}:{size}:{iteration}".encode()
                ).digest()[:4],
                "big",
            )
            error_text = None
            try:
                program = self.program_factory(
                    self.checkpoint.code,
                    timeout=self.optimizer_timeout,
                    seed=child_seed,
                )
                returned = program(
                    copy.deepcopy(population),
                    utility,
                    reference_llm,
                    self.heuristic_format,
                    task,
                )
                idea, code, _supplied_score = returned
                if code:
                    evaluate_candidate(code, idea)
            except _BudgetExhausted as error:
                early_stop_reason = "evaluation_budget_exhausted"
                error_text = str(error)
            except Exception as error:  # noqa: BLE001 -- generated candidate boundary
                error_text = f"{type(error).__name__}: {error}"
                logger.warning(
                    "Target %s iteration %s failed: %s", size, iteration, error
                )
            iterations_completed += 1
            if error_text:
                iteration_errors.append(
                    {"iteration": iteration, "error": error_text}
                )
            save_iteration_snapshot(iteration, error_text)
            if evaluation_count >= self.max_evaluations:
                early_stop_reason = "evaluation_budget_exhausted"
            if early_stop_reason:
                break

        if not evaluated:
            raise TargetInitializationError(
                f"Target {size} has no finite successful validation result"
            )
        idea, code, validation = min(
            evaluated.values(), key=lambda item: item[2].mean_gap_pct
        )
        metadata = {
            "iterations_completed": iterations_completed,
            "evaluation_count": evaluation_count,
            "early_stop_reason": early_stop_reason,
            "initialization": dict(population.initialization),
            "reference_source_sizes": reference_llm.reference_source_sizes,
            "reference_code_hashes": reference_llm.reference_code_hashes,
            "iteration_errors": iteration_errors,
            "snapshots": snapshot_paths,
            "validation": asdict(validation),
        }
        return SelectedHeuristic(size, idea, code, validation, metadata)
