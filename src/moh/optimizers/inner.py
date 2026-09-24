import random
from typing import Protocol

from moh.core.models import (
    EvaluationContext,
    Heuristic,
    InnerResult,
    ScoredHeuristic,
    WorkCounts,
)
from moh.core.seeds import derive_seed
from moh.core.specs import OptimizerSpec
from moh.llm.base import GenerationError
from moh.llm.parsing import strip_code_fence
from moh.llm.recording import RecordingLLM
from moh.logging import to_json
from moh.optimizers.selection import select_parents, select_survivors
from moh.problems.baselines import NEAREST_NEIGHBOR_SOURCE
from moh.prompts.heuristic_generation import heuristic_prompt


class HeuristicOptimizer(Protocol):
    def optimize(
        self, *, task, llm, runner, iterations, root_seed, namespace, emit
    ) -> InnerResult: ...


def generate_child(parents, spec, llm, candidate_id):
    idea = (
        llm.generate(heuristic_prompt("reflection", parents, None))
        if spec.use_reflection
        else None
    )
    source = strip_code_fence(
        llm.generate(heuristic_prompt(spec.generation_operator, parents, idea))
    )
    return Heuristic(candidate_id, source, idea)


def evaluate_child(child, task, runner, context):
    return ScoredHeuristic(child, runner.evaluate(child, task, context))


def record_population(members, generation, emit):
    emit(
        "population_updated",
        {
            "level": "inner",
            "generation": generation,
            "ids": [x.heuristic.id for x in members],
        },
    )


class ConfiguredOptimizer:
    def __init__(self, spec):
        self.spec = spec

    def optimize(self, *, task, llm, runner, iterations, root_seed, namespace, emit):
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        derive_seed(root_seed)
        spec = self.spec
        if not isinstance(llm, RecordingLLM):
            llm = RecordingLLM(
                llm, emit, {"optimizer_id": namespace, "task_id": task.id}
            )
        start_calls = llm.calls
        context = EvaluationContext(
            task.id,
            task.instance_seeds,
            tuple(
                derive_seed(root_seed, "task", task.id, "instance", i, "evaluation", 0)
                for i in range(len(task.instances))
            ),
        )
        evaluations, attempts = 0, 0

        def evaluate(child, generation, parents=(), policy_seed=None):
            nonlocal evaluations, attempts
            emit(
                "heuristic_generated",
                {
                    "id": child.id,
                    "source_code": child.source_code,
                    "idea": child.idea,
                    "parents": list(parents),
                    "generation": generation,
                    "task_id": task.id,
                    "optimizer_id": namespace,
                    "policy_seed": policy_seed,
                },
            )
            scored = evaluate_child(child, task, runner, context)
            emit("heuristic_evaluated", to_json(scored.evaluation))
            evaluations += 1
            attempts += scored.evaluation.instances_attempted
            return scored

        members = []
        for slot in range(spec.population_size):
            policy_seed = None
            source = NEAREST_NEIGHBOR_SOURCE
            if slot:
                policy_seed = derive_seed(
                    root_seed, "task", task.id, "candidate", slot, "policy"
                )
                source = (
                    f"_policy_rng = random.Random({policy_seed})\n"
                    "def select_next_node(current_node, unvisited, coordinates):\n"
                    "    return _policy_rng.choice(sorted(unvisited))\n"
                )
            child = Heuristic(f"{namespace}-{task.id}-h{slot + 1:06d}", source)
            members.append(evaluate(child, -1, policy_seed=policy_seed))
        members = select_survivors(
            members, spec.population_size, spec.survivor_selection
        )
        record_population(members, -1, emit)
        for generation in range(iterations):
            seed = derive_seed(
                root_seed, "task", task.id, "search", generation, "parents"
            )
            parents = select_parents(
                members,
                spec.parent_selection,
                2 if spec.generation_operator == "crossover" else 1,
                random.Random(seed),
            )
            identity = (
                f"{namespace}-{task.id}-h{spec.population_size + generation + 1:06d}"
            )
            emit(
                "inner_generation_started",
                {
                    "id": identity,
                    "task_id": task.id,
                    "optimizer_id": namespace,
                    "generation": generation,
                    "seed": seed,
                    "parents": [x.heuristic.id for x in parents],
                },
            )
            try:
                child = generate_child(parents, spec, llm, identity)
            except GenerationError as exc:
                emit(
                    "generation_failed",
                    {
                        "id": identity,
                        "generation": generation,
                        "error": str(exc)[:1024],
                    },
                )
                continue
            scored = evaluate(child, generation, [x.heuristic.id for x in parents])
            members = select_survivors(
                [*members, scored], spec.population_size, spec.survivor_selection
            )
            record_population(members, generation, emit)
        counts = WorkCounts(evaluations, attempts, llm.calls - start_calls)
        emit(
            "inner_finished",
            {"task_id": task.id, "optimizer_id": namespace, "counts": to_json(counts)},
        )
        return InnerResult(tuple(members), counts)


def compile_optimizer(spec):
    return ConfiguredOptimizer(OptimizerSpec.model_validate(spec.model_dump()))
