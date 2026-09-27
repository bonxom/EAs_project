"""Full MoH Offline End-to-End Orchestration."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from moh.core.models import JSONValue, OptimizerProgram
from moh.optimizers.capabilities import CapabilityLimits
from moh.optimizers.evaluator import InnerSearchResult
from moh.optimizers.evolution import (
    EvaluatedOptimizer,
    OuterEvolutionResult,
    initial_optimizer_programs,
    run_outer_evolution,
)
from moh.optimizers.runner import ProgramLimits

type EventSink = Callable[[str, dict[str, JSONValue]], None]


@dataclass(frozen=True)
class FullMoHConfig:
    population_size: int = 2
    generations: int = 1
    capability_limits: CapabilityLimits = field(
        default_factory=lambda: CapabilityLimits(
            max_generate_requests=10, max_evaluate_requests=10
        )
    )
    program_limits: ProgramLimits = field(
        default_factory=lambda: ProgramLimits(
            timeout_seconds=5.0,
            max_message_bytes=1048576,
            max_output_bytes=65536,
        )
    )

    def __post_init__(self):
        if (
            type(self.population_size) is bool
            or not isinstance(self.population_size, int)
            or self.population_size <= 0
        ):
            raise ValueError("population_size must be a positive integer")

        if (
            type(self.generations) is bool
            or not isinstance(self.generations, int)
            or self.generations < 0
        ):
            raise ValueError("generations must be a non-negative integer")

        if not isinstance(self.capability_limits, CapabilityLimits):
            raise TypeError("capability_limits must be a CapabilityLimits instance")

        if not isinstance(self.program_limits, ProgramLimits):
            raise TypeError("program_limits must be a ProgramLimits instance")


@dataclass(frozen=True)
class FullMoHWorkCounts:
    outer_programs_evaluated: int
    outer_offspring_generated: int
    inner_generate_requests: int
    inner_evaluate_requests: int

    def __post_init__(self):
        for name, val in (
            ("outer_programs_evaluated", self.outer_programs_evaluated),
            ("outer_offspring_generated", self.outer_offspring_generated),
            ("inner_generate_requests", self.inner_generate_requests),
            ("inner_evaluate_requests", self.inner_evaluate_requests),
        ):
            if type(val) is bool or not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class FullMoHResult:
    outer_result: OuterEvolutionResult
    work_counts: FullMoHWorkCounts

    def __post_init__(self):
        if not isinstance(self.outer_result, OuterEvolutionResult):
            raise TypeError("outer_result must be an OuterEvolutionResult instance")
        if not isinstance(self.work_counts, FullMoHWorkCounts):
            raise TypeError("work_counts must be a FullMoHWorkCounts instance")

    @property
    def status(self) -> str:
        return self.outer_result.status

    @property
    def winner(self) -> EvaluatedOptimizer | None:
        return self.outer_result.winner

    @property
    def best_program(self) -> OptimizerProgram | None:
        return self.outer_result.best_program

    @property
    def best_inner_result(self) -> InnerSearchResult | None:
        return self.outer_result.best_inner_result

    @property
    def best_utility(self) -> float | None:
        return self.outer_result.best_utility


def run_full_moh(
    *,
    config: FullMoHConfig | None = None,
    seed_programs: Sequence[OptimizerProgram] | None = None,
    meta_llm: Any,
    inner_llm: Any = None,
    evaluator: Callable[[str], Any],
    inner_llm_factory: Callable[[str], Any] | None = None,
    emit: EventSink | None = None,
) -> FullMoHResult:
    cfg = config or FullMoHConfig()
    if not isinstance(cfg, FullMoHConfig):
        raise TypeError("config must be a FullMoHConfig instance")

    if seed_programs is not None:
        if not seed_programs:
            raise ValueError("seed_programs must be non-empty if provided")
        initial_programs = tuple(seed_programs)
    else:
        initial_programs = initial_optimizer_programs(cfg.population_size)

    if inner_llm_factory is not None:
        inner_factory = inner_llm_factory
    elif inner_llm is not None:
        inner_factory = lambda _prog_id: inner_llm
    else:
        inner_factory = lambda _prog_id: meta_llm

    outer_result = run_outer_evolution(
        initial_programs=initial_programs,
        llm=meta_llm,
        evaluator=evaluator,
        inner_llm_factory=inner_factory,
        iterations=cfg.generations,
        outer_capacity=cfg.population_size,
        capability_limits=cfg.capability_limits,
        program_limits=cfg.program_limits,
        emit=emit,
    )

    evaluations_to_count = outer_result.all_evaluated
    if not evaluations_to_count:
        seen_ids: set[str] = set()
        evaluations_list = []
        for gen_record in outer_result.generations:
            for evaluated in gen_record.population:
                if evaluated.program.id not in seen_ids:
                    seen_ids.add(evaluated.program.id)
                    evaluations_list.append(evaluated)
        evaluations_to_count = tuple(evaluations_list)

    total_inner_generate = sum(
        ev.inner_result.generated_count for ev in evaluations_to_count
    )
    total_inner_evaluate = sum(
        ev.inner_result.evaluated_count for ev in evaluations_to_count
    )

    work_counts = FullMoHWorkCounts(
        outer_programs_evaluated=outer_result.programs_evaluated,
        outer_offspring_generated=outer_result.offspring_generated,
        inner_generate_requests=total_inner_generate,
        inner_evaluate_requests=total_inner_evaluate,
    )

    return FullMoHResult(outer_result=outer_result, work_counts=work_counts)
