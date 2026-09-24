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


def run_experiment(config):
    from contextlib import ExitStack

    from moh.config import ExperimentConfig
    from moh.core.models import Heuristic, OptimizerCandidate
    from moh.core.seeds import derive_seed
    from moh.core.specs import OptimizerSpec
    from moh.evaluation import OptimizerEvaluator
    from moh.execution.heuristic_runner import HeuristicRunner
    from moh.execution.protocol import ExecutionLimits
    from moh.llm.fake import FakeLLM
    from moh.llm.recording import RecordingLLM
    from moh.logging import RunRecorder, to_json
    from moh.problems.tsp import TSPTask

    config = ExperimentConfig.model_validate(config.model_dump())
    if config.llm.provider == "openai":
        from moh.llm.openai_client import OpenAILLMClient, validate_environment

        validate_environment()
    with ExitStack() as stack:
        recorder = stack.enter_context(
            RunRecorder.create(config.output_dir, config.model_dump(mode="json"))
        )

        def emit(event, payload):
            if event == "heuristic_generated":
                recorder.save_heuristic(
                    Heuristic(
                        payload["id"], payload["source_code"], payload.get("idea")
                    )
                )
            elif event == "optimizer_generated":
                recorder.save_optimizer(
                    OptimizerCandidate(
                        payload["id"], OptimizerSpec.model_validate(payload["spec"])
                    )
                )
            recorder.emit(event, payload)

        def factory(scope):
            lineage = {"scope": list(scope)}
            seed = (
                derive_seed(config.seed, "outer", "llm")
                if scope == ("outer",)
                else derive_seed(config.seed, "task", scope[1], "llm")
            )
            emit(
                "llm_initialized",
                {**lineage, "provider": config.llm.provider, "seed": seed},
            )
            if config.llm.provider == "fake":
                client = FakeLLM(seed)
            else:

                def observer(metadata):
                    emit(
                        "llm_attempt",
                        {**lineage, "call_id": recording.calls, **to_json(metadata)},
                    )

                client = OpenAILLMClient(
                    config.llm.model, config.llm.timeout_seconds, observer
                )
                stack.callback(client.close)
            recording = RecordingLLM(client, emit, lineage)
            return recording

        try:
            tasks = tuple(
                TSPTask.create(size, config.instances_per_task, config.seed)
                for size in config.tasks.sizes
            )
            runner = HeuristicRunner(ExecutionLimits(**config.execution.model_dump()))
            evaluator = OptimizerEvaluator(
                tasks,
                tuple(config.tasks.weights),
                runner,
                factory,
                config.inner.iterations,
                config.seed,
                emit,
            )
            result = run_outer_loop(
                evaluator=evaluator,
                llm=factory(("outer",)),
                iterations=config.outer.iterations,
                outer_capacity=config.outer.population_size,
                initial_inner_capacity=config.inner.population_size,
                emit=emit,
            )
            recorder.finish(result)
            return result, recorder.path
        except Exception as exc:
            try:
                recorder.fail(type(exc).__name__)
            except (OSError, ValueError):
                pass  # Preserve the original infrastructure error.
            raise
