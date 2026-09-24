from moh.core.models import RunResult, ScoredOptimizer
from moh.core.populations import rank_key
from moh.llm.base import GenerationError
from moh.optimizers.meta import MetaOptimizer
from moh.optimizers.seed_optimizers import initial_optimizer_candidates


def optimizer_rank(item):
    return rank_key(item.evaluation.status, item.evaluation.utility, item.candidate.id)


def run_outer_loop(
    *, evaluator, llm, iterations, outer_capacity, initial_inner_capacity, emit
):
    if (
        type(iterations) is not int
        or iterations < 0
        or type(outer_capacity) is not int
        or outer_capacity < 2
    ):
        raise ValueError("invalid outer budget or capacity")
    seeds = initial_optimizer_candidates(initial_inner_capacity)
    population = []
    for candidate in seeds:
        emit(
            "optimizer_generated",
            {
                "id": candidate.id,
                "spec": candidate.spec.model_dump(),
                "generation": -1,
                "parents": [],
            },
        )
        population.append(ScoredOptimizer(candidate, evaluator.evaluate(candidate)))
    population.sort(key=optimizer_rank)
    emit(
        "population_updated",
        {
            "level": "outer",
            "generation": -1,
            "ids": [x.candidate.id for x in population],
        },
    )
    for generation in range(iterations):
        identity = f"o{generation + 3:06d}"
        emit("outer_generation_started", {"generation": generation, "id": identity})
        try:
            child = MetaOptimizer().propose(tuple(population), llm, identity)
        except GenerationError as exc:
            emit(
                "generation_failed",
                {"id": identity, "generation": generation, "error": str(exc)[:1024]},
            )
            continue
        emit(
            "optimizer_generated",
            {
                "id": child.id,
                "spec": child.spec.model_dump(),
                "generation": generation,
                "parents": [x.candidate.id for x in population],
            },
        )
        evaluated = ScoredOptimizer(child, evaluator.evaluate(child))
        population = sorted([*population, evaluated], key=optimizer_rank)[
            :outer_capacity
        ]
        emit(
            "population_updated",
            {
                "level": "outer",
                "generation": generation,
                "ids": [x.candidate.id for x in population],
            },
        )
    winner = next((x for x in population if x.evaluation.status == "success"), None)
    status = "success" if winner else "failed"
    emit(
        "outer_finished",
        {
            "status": status,
            "winner": winner.candidate.id if winner else None,
            "error": None if winner else "all_optimizers_failed",
        },
    )
    return RunResult(status, winner, tuple(population))
