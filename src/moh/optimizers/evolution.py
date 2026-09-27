"""M4 Outer Evolution over OptimizerProgram."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from moh.core.models import JSONValue, OptimizerProgram
from moh.llm.base import GenerationError
from moh.optimizers.capabilities import CapabilityLimits
from moh.optimizers.evaluator import InnerSearchResult, run_optimizer_inner_search
from moh.optimizers.meta import ProgramMetaOptimizer
from moh.optimizers.programs import parse_optimizer_program
from moh.optimizers.runner import ProgramLimits

type EventSink = Callable[[str, dict[str, JSONValue]], None]


@dataclass(frozen=True)
class EvaluatedOptimizer:
    program: OptimizerProgram
    inner_result: InnerSearchResult
    utility: float | None

    def __post_init__(self):
        if not isinstance(self.program, OptimizerProgram):
            raise TypeError("program must be an OptimizerProgram instance")
        if not isinstance(self.inner_result, InnerSearchResult):
            raise TypeError("inner_result must be an InnerSearchResult instance")
        if self.utility != self.inner_result.utility:
            raise ValueError("utility must match inner_result.utility exactly")
        if self.utility is not None:
            if (
                type(self.utility) is bool
                or not isinstance(self.utility, (int, float))
                or not math.isfinite(self.utility)
            ):
                raise ValueError("utility must be a finite float if present")
            object.__setattr__(self, "utility", float(self.utility))


def outer_rank_key(item: EvaluatedOptimizer) -> tuple[int, float, str]:
    if item.utility is not None:
        return (0, -item.utility, item.program.id)
    return (1, 0.0, item.program.id)


@dataclass(frozen=True)
class OuterGenerationResult:
    generation: int
    population: tuple[EvaluatedOptimizer, ...]
    best_program_id: str | None
    best_utility: float | None

    def __post_init__(self):
        object.__setattr__(self, "population", tuple(self.population))
        if type(self.generation) is bool or not isinstance(self.generation, int):
            raise TypeError("generation must be an integer")
        if self.best_utility is not None:
            if (
                type(self.best_utility) is bool
                or not isinstance(self.best_utility, (int, float))
                or not math.isfinite(self.best_utility)
            ):
                raise ValueError("best_utility must be a finite float if present")
            object.__setattr__(self, "best_utility", float(self.best_utility))
            if (
                not isinstance(self.best_program_id, str)
                or not self.best_program_id.strip()
            ):
                raise ValueError(
                    "best_program_id must be a string when best_utility is present"
                )
        else:
            if self.best_program_id is not None:
                raise ValueError(
                    "best_program_id must be None when best_utility is None"
                )


@dataclass(frozen=True)
class OuterEvolutionResult:
    status: Literal["success", "failed"]
    winner: EvaluatedOptimizer | None
    generations: tuple[OuterGenerationResult, ...]
    best_program: OptimizerProgram | None
    best_inner_result: InnerSearchResult | None
    best_utility: float | None
    programs_evaluated: int
    offspring_generated: int
    all_evaluated: tuple[EvaluatedOptimizer, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "generations", tuple(self.generations))
        object.__setattr__(self, "all_evaluated", tuple(self.all_evaluated))
        if self.status not in ("success", "failed"):
            raise ValueError("status must be 'success' or 'failed'")
        for count_name, count_val in (
            ("programs_evaluated", self.programs_evaluated),
            ("offspring_generated", self.offspring_generated),
        ):
            if (
                type(count_val) is bool
                or not isinstance(count_val, int)
                or count_val < 0
            ):
                raise ValueError(f"{count_name} must be a non-negative integer")

        if self.status == "success":
            if self.winner is None or self.winner.utility is None:
                raise ValueError(
                    "success requires a valid winner with non-null utility"
                )
            if self.best_program is None or self.best_inner_result is None:
                raise ValueError("success requires best_program and best_inner_result")
            if self.best_utility != self.winner.utility:
                raise ValueError("best_utility must match winner utility")
        else:
            if self.winner is not None:
                raise ValueError("failed status must have null winner")


def initial_optimizer_programs(count: int = 2) -> tuple[OptimizerProgram, ...]:
    if type(count) is bool or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    programs = [
        parse_optimizer_program(
            "def improve_algorithm(api):\n"
            "    idea = api.generate('KIND: reflection\\nAnalyze performance.')\n"
            "    code = api.generate('KIND: mutate\\nImprove solution.')\n"
            "    return api.evaluate(code)\n",
            "o000001",
            idea="Initial reflection-based optimizer",
        ),
        parse_optimizer_program(
            "def improve_algorithm(api):\n"
            "    c1 = api.generate('KIND: mutate\\nOption 1.')\n"
            "    s1 = api.evaluate(c1)\n"
            "    c2 = api.generate('KIND: mutate\\nOption 2.')\n"
            "    s2 = api.evaluate(c2)\n"
            "    return max(s1, s2)\n",
            "o000002",
            idea="Initial candidate-comparison optimizer",
        ),
    ]
    while len(programs) < count:
        idx = len(programs) + 1
        programs.append(
            parse_optimizer_program(
                "def improve_algorithm(api):\n"
                f"    c = api.generate('KIND: mutate\\nOption {idx}.')\n"
                "    return api.evaluate(c)\n",
                f"o{idx:06d}",
                idea=f"Initial seed optimizer {idx}",
            )
        )
    return tuple(programs[:count])


def evaluate_optimizer_program(
    program: OptimizerProgram,
    llm: Any,
    evaluator: Callable[[str], Any],
    capability_limits: CapabilityLimits,
    program_limits: ProgramLimits | None = None,
) -> EvaluatedOptimizer:
    _worker_result, _capability_usage, inner_result = run_optimizer_inner_search(
        program=program,
        llm=llm,
        evaluator=evaluator,
        capability_limits=capability_limits,
        program_limits=program_limits,
    )
    return EvaluatedOptimizer(
        program=program,
        inner_result=inner_result,
        utility=inner_result.utility,
    )


def run_outer_evolution(
    *,
    initial_programs: Sequence[OptimizerProgram],
    llm: Any,
    evaluator: Callable[[str], Any],
    inner_llm_factory: Callable[[str], Any] | None = None,
    iterations: int = 1,
    outer_capacity: int = 2,
    capability_limits: CapabilityLimits | None = None,
    program_limits: ProgramLimits | None = None,
    emit: EventSink | None = None,
) -> OuterEvolutionResult:
    if type(iterations) is bool or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if (
        type(outer_capacity) is bool
        or not isinstance(outer_capacity, int)
        or outer_capacity < 1
    ):
        raise ValueError("outer_capacity must be a positive integer")

    if not initial_programs:
        raise ValueError("initial_programs must be non-empty")

    cap_limits = capability_limits or CapabilityLimits(
        max_generate_requests=10, max_evaluate_requests=10
    )

    def safe_emit(event: str, payload: dict[str, Any]):
        if emit is not None:
            emit(event, payload)

    programs_evaluated = 0
    offspring_generated = 0
    population: list[EvaluatedOptimizer] = []
    all_evaluated_list: list[EvaluatedOptimizer] = []

    for prog in initial_programs:
        safe_emit(
            "optimizer_program_generated",
            {
                "id": prog.id,
                "source_code": prog.source_code,
                "idea": prog.idea,
                "generation": -1,
                "parents": [],
            },
        )
        inner_llm = inner_llm_factory(prog.id) if inner_llm_factory else llm
        evaluated = evaluate_optimizer_program(
            prog, inner_llm, evaluator, cap_limits, program_limits
        )
        programs_evaluated += 1
        population.append(evaluated)
        all_evaluated_list.append(evaluated)

    population.sort(key=outer_rank_key)
    population = population[:outer_capacity]

    best_in_pop = (
        population[0] if population and population[0].utility is not None else None
    )
    gen0_result = OuterGenerationResult(
        generation=-1,
        population=tuple(population),
        best_program_id=best_in_pop.program.id if best_in_pop else None,
        best_utility=best_in_pop.utility if best_in_pop else None,
    )
    generations_records = [gen0_result]

    safe_emit(
        "outer_population_updated",
        {
            "generation": -1,
            "ids": [x.program.id for x in population],
            "best_utility": gen0_result.best_utility,
        },
    )

    meta_opt = ProgramMetaOptimizer()

    for gen in range(iterations):
        identity = f"o{gen + len(initial_programs) + 1:06d}"
        safe_emit("outer_generation_started", {"generation": gen, "id": identity})

        pop_records = [
            {
                "id": x.program.id,
                "source_code": x.program.source_code,
                "utility": x.utility,
            }
            for x in population
        ]

        try:
            offspring = meta_opt.propose(pop_records, llm, identity)
            offspring_generated += 1
        except GenerationError as exc:
            safe_emit(
                "outer_generation_failed",
                {"id": identity, "generation": gen, "error": str(exc)[:1024]},
            )
            curr_best = (
                population[0]
                if population and population[0].utility is not None
                else None
            )
            generations_records.append(
                OuterGenerationResult(
                    generation=gen,
                    population=tuple(population),
                    best_program_id=curr_best.program.id if curr_best else None,
                    best_utility=curr_best.utility if curr_best else None,
                )
            )
            continue

        safe_emit(
            "optimizer_program_generated",
            {
                "id": offspring.id,
                "source_code": offspring.source_code,
                "idea": offspring.idea,
                "generation": gen,
                "parents": [x.program.id for x in population],
            },
        )

        inner_llm = inner_llm_factory(offspring.id) if inner_llm_factory else llm
        evaluated_offspring = evaluate_optimizer_program(
            offspring, inner_llm, evaluator, cap_limits, program_limits
        )
        programs_evaluated += 1
        all_evaluated_list.append(evaluated_offspring)

        population = sorted([*population, evaluated_offspring], key=outer_rank_key)[
            :outer_capacity
        ]
        curr_best = (
            population[0] if population and population[0].utility is not None else None
        )

        gen_record = OuterGenerationResult(
            generation=gen,
            population=tuple(population),
            best_program_id=curr_best.program.id if curr_best else None,
            best_utility=curr_best.utility if curr_best else None,
        )
        generations_records.append(gen_record)

        safe_emit(
            "outer_population_updated",
            {
                "generation": gen,
                "ids": [x.program.id for x in population],
                "best_utility": gen_record.best_utility,
            },
        )

    winner = (
        population[0] if population and population[0].utility is not None else None
    )
    status = "success" if winner else "failed"

    safe_emit(
        "outer_evolution_finished",
        {
            "status": status,
            "winner": winner.program.id if winner else None,
            "best_utility": winner.utility if winner else None,
            "programs_evaluated": programs_evaluated,
            "offspring_generated": offspring_generated,
        },
    )

    return OuterEvolutionResult(
        status=status,
        winner=winner,
        generations=tuple(generations_records),
        best_program=winner.program if winner else None,
        best_inner_result=winner.inner_result if winner else None,
        best_utility=winner.utility if winner else None,
        programs_evaluated=programs_evaluated,
        offspring_generated=offspring_generated,
        all_evaluated=tuple(all_evaluated_list),
    )
