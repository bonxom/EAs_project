"""Compose validated data, isolated program search and separate held-out scoring."""

import hashlib
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from moh.core.models import Heuristic, WorkCounts
from moh.core.program_population import ProgramPopulation
from moh.core.programs import GapEvaluation, OptimizerProgram, ProgramRunResult
from moh.core.seeds import derive_seed
from moh.execution.budgets import WorkBudget
from moh.execution.gls_runner import GLSRunner
from moh.execution.optimizer_runner import OptimizerRunner
from moh.execution.process import (
    CandidateFailure,
    Deadline,
    ProcessSupervisor,
    ProgramLimits,
)
from moh.experiments.artifacts import ProgramRecorder
from moh.llm.fake import FakeLLM
from moh.llm.program_adapter import ProgramLLM
from moh.llm.recording import RecordingLLM
from moh.logging import to_json
from moh.optimizers.program_inner import ProgramInner, initialize_task_population
from moh.optimizers.program_meta import ProgramEvaluator, ProgramMeta
from moh.optimizers.seeds import (
    BASIC_SOURCE,
    BEST_PARENT_SOURCE,
    MULTI_TEMPERATURE_SOURCE,
)
from moh.problems.tsp_gls.dataset import load_npz_task, synthetic_task
from moh.problems.tsp_gls.solver import GLSOptions
from moh.program_config import ProgramConfig


def _load_tasks(config):
    if config.tasks.source == "synthetic":
        tasks = tuple(
            synthetic_task(
                size,
                config.tasks.validation_count,
                config.tasks.test_count,
                config.seed,
            )
            for size in config.tasks.sizes
        )
    else:
        tasks = tuple(
            load_npz_task(
                dataset.path,
                tuple(dataset.validation_indices),
                tuple(dataset.test_indices),
            )
            for dataset in config.tasks.datasets
        )
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("datasets must describe distinct task sizes/IDs")
    weights = (
        tuple(config.weights)
        if config.weights is not None
        else tuple(task.size / sum(task.size for task in tasks) for task in tasks)
    )
    return tasks, weights


def _limits(config, timeout):
    return ProgramLimits(
        timeout_seconds=timeout,
        **config.execution.model_dump(
            exclude={
                "run_timeout_seconds",
                "optimizer_timeout_seconds",
                "heuristic_timeout_seconds",
            }
        ),
    )


def _initial_programs():
    return (
        OptimizerProgram("seed-basic", BASIC_SOURCE, "upstream basic optimizer"),
        OptimizerProgram(
            "seed-multi-temperature",
            MULTI_TEMPERATURE_SOURCE,
            "upstream optimizer with temperature search",
        ),
        OptimizerProgram(
            "seed-best-parent", BEST_PARENT_SOURCE, "offline best-parent optimizer"
        ),
    )


def _evaluate_held_out(
    result, tasks, runner, budget, *, root_seed, deadline, scope, emit
):
    """Test only winner-selected heuristics; never update validation populations."""
    if result.winner is None:
        return replace(result, counts=budget.counts)
    selected = {
        outcome.task_id: outcome.selected
        for outcome in result.winner.evaluation.task_results
    }
    if set(selected) != {task.id for task in tasks} or any(
        item is None for item in selected.values()
    ):
        raise ValueError("winner must select a validation heuristic for every task")
    outcomes = []
    for task in tasks:
        heuristic = selected[task.id].candidate
        before = budget.counts
        try:
            deadline.check()
            scope.check()
            budget.charge_evaluation()
            emit(
                "heuristic_worker_started",
                {
                    "id": heuristic.id,
                    "source_hash": hashlib.sha256(
                        heuristic.source_code.encode()
                    ).hexdigest(),
                    "task_id": task.id,
                    "split": "test",
                    "optimizer_id": result.winner.id,
                    "invocation_id": f"held-out-{task.id}",
                    "kind": "heuristic",
                },
            )
            evaluation = runner.evaluate(
                heuristic,
                task,
                split="test",
                root_seed=root_seed,
                deadline=deadline,
                scope=scope,
            )
            budget.charge_instances(evaluation.counts.instance_attempts)
        except CandidateFailure as exc:
            evaluation = GapEvaluation(
                heuristic.id,
                task.id,
                "test",
                "failed",
                None,
                (),
                (),
                (),
                exc.code,
                budget.delta(before),
            )
        outcomes.append(evaluation)
        emit(
            "held_out_evaluated",
            {"optimizer_id": result.winner.id, "evaluation": to_json(evaluation)},
        )
    return replace(result, test_results=tuple(outcomes), counts=budget.counts)


def run_program_experiment(config: ProgramConfig) -> tuple[ProgramRunResult, Path]:
    """Run both optimization levels offline by default, retaining every executed source."""
    config = ProgramConfig.model_validate(config.model_dump(mode="python"))
    tasks, weights = _load_tasks(config)
    profiles = (config.heuristic_llm, config.meta_llm)
    if any(profile.provider == "openai" for profile in profiles):
        from moh.llm.openai_client import validate_environment

        validate_environment()
    options = GLSOptions(**config.solver.model_dump())
    optimizer_limits = _limits(config, config.execution.optimizer_timeout_seconds)
    heuristic_limits = _limits(config, config.execution.heuristic_timeout_seconds)
    provenance = {
        "tasks": [
            {"id": task.id, "size": task.size, **task.provenance} for task in tasks
        ],
        "weights": list(weights),
        "solver": config.solver.model_dump(mode="json"),
        "upstream": {
            "revision": "e8c6154911d8182f2a53bcc17b11be0a063a53de",
            "attribution": "third_party/moh/README.md",
            "license": "third_party/moh/LICENSE",
        },
    }
    with ExitStack() as stack:
        recorder = stack.enter_context(
            ProgramRecorder.create(
                config.output_dir, config.model_dump(mode="json"), provenance=provenance
            )
        )
        budget = WorkBudget(
            config.budgets.max_llm_calls, config.budgets.max_heuristic_evaluations
        )
        populations = {
            task.id: ProgramPopulation(config.population_size) for task in tasks
        }
        supervisor = ProcessSupervisor()

        def emit(event, payload):
            if event == "heuristic_generated":
                recorder.save_heuristic(
                    Heuristic(
                        payload["id"], payload["source_code"], payload.get("idea")
                    )
                )
            elif event == "optimizer_generated":
                recorder.save_optimizer(
                    OptimizerProgram(
                        payload["id"], payload["source_code"], payload.get("idea")
                    )
                )
            elif event == "search_checkpoint":
                recorder.checkpoint(
                    payload["label"],
                    payload["populations"],
                    payload["active"],
                    payload["winner"],
                )
            elif event == "llm_requested":
                # Wall time enforces provider bounds but is not a seeded search input.
                payload = {
                    key: value
                    for key, value in payload.items()
                    if key != "timeout_seconds"
                }
            recorder.emit(event, payload)

        def factory(scope):
            profile_name = "meta" if scope[0] == "meta" else "heuristic"
            profile = (
                config.meta_llm if profile_name == "meta" else config.heuristic_llm
            )
            seed = derive_seed(config.seed, "program_llm", *scope)
            lineage = {"scope": list(scope), "profile": profile_name}
            emit(
                "llm_initialized",
                {
                    **lineage,
                    "provider": profile.provider,
                    "model": profile.model,
                    "seed": seed,
                },
            )
            if profile.provider == "fake":
                client = FakeLLM(seed)
            else:
                from moh.llm.openai_client import OpenAILLMClient

                def observe(metadata):
                    emit(
                        "llm_attempt",
                        {**lineage, "call_id": recording.calls, **to_json(metadata)},
                    )

                client = OpenAILLMClient(
                    profile.model, profile.timeout_seconds, observe
                )
            if callable(getattr(client, "close", None)):
                stack.callback(client.close)
            recording = RecordingLLM(client, emit, lineage)
            return ProgramLLM(recording, batch_size=config.execution.batch_size)

        try:
            deadline = Deadline.after(config.execution.run_timeout_seconds)
            try:
                with supervisor.scope(deadline) as scope:
                    gls = GLSRunner(supervisor, heuristic_limits, options)
                    optimizer = OptimizerRunner(supervisor, optimizer_limits)
                    for task in tasks:
                        populations[task.id] = initialize_task_population(
                            task,
                            gls,
                            factory(("initialize", task.id)),
                            budget,
                            capacity=config.population_size,
                            seed_attempts=config.seed_attempts,
                            threshold=config.seed_threshold,
                            root_seed=config.seed,
                            deadline=deadline,
                            scope=scope,
                            emit=emit,
                        )
                    recorder.checkpoint(
                        "initial",
                        {
                            **populations,
                            "meta-optimizer": ProgramPopulation(config.population_size),
                        },
                        None,
                        None,
                    )
                    inner = ProgramInner(optimizer, gls, budget, emit)
                    evaluator = ProgramEvaluator(
                        tasks, weights, inner, factory, budget, emit
                    )
                    result = ProgramMeta(optimizer, evaluator, budget, emit).search(
                        _initial_programs(),
                        populations,
                        factory(("meta",)),
                        iterations=config.outer_iterations,
                        capacity=config.population_size,
                        root_seed=config.seed,
                        deadline=deadline,
                        scope=scope,
                    )
                    emit(
                        "search_finished",
                        {
                            "status": result.status,
                            "winner_id": result.winner.id if result.winner else None,
                            "counts": to_json(budget.counts),
                        },
                    )
                    result = _evaluate_held_out(
                        result,
                        tasks,
                        gls,
                        budget,
                        root_seed=config.seed,
                        deadline=deadline,
                        scope=scope,
                        emit=emit,
                    )
            except CandidateFailure as exc:
                emit(
                    "search_failed",
                    {"error": exc.code, "counts": to_json(budget.counts)},
                )
                result = ProgramRunResult(
                    "failed", None, None, (), populations, (), budget.counts
                )
            recorder.metadata.update(
                {
                    "test_status": (
                        "not_run"
                        if result.winner is None
                        else "success"
                        if all(item.status == "success" for item in result.test_results)
                        else "incomplete"
                    ),
                    "budgets": config.budgets.model_dump(mode="json"),
                    "test_counts": to_json(
                        sum((item.counts for item in result.test_results), WorkCounts())
                    ),
                }
            )
            recorder.finish(result)
            return result, recorder.path
        except Exception as exc:
            try:
                recorder.fail(type(exc).__name__)
            except (OSError, ValueError):
                pass  # Preserve the original infrastructure exception.
            raise
