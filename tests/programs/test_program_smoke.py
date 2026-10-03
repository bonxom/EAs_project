import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from moh.core.models import Heuristic, WorkCounts
from moh.core.program_population import ProgramPopulation
from moh.core.programs import (
    GapEvaluation,
    OptimizerProgram,
    ProgramEvaluation,
    ProgramRunResult,
    ScoredProgram,
    TaskOutcome,
)
from moh.execution.budgets import WorkBudget
from moh.execution.process import Deadline, ProcessSupervisor
from moh.experiments import program_search
from moh.experiments.artifacts import ProgramRecorder
from moh.problems.tsp_gls.dataset import synthetic_task
from moh.program_config import ProgramConfig


def events(path):
    return [
        json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()
    ]


def small_config(tmp_path, **overrides):
    return ProgramConfig(
        output_dir=tmp_path,
        tasks={"sizes": [4], "validation_count": 1},
        outer_iterations=0,
        seed_attempts=0,
        **overrides,
    )


def test_program_smoke_reproducible(tmp_path):
    config = ProgramConfig(output_dir=tmp_path / "a")
    left, lp = program_search.run_program_experiment(config)
    right, rp = program_search.run_program_experiment(
        config.model_copy(update={"output_dir": tmp_path / "b"})
    )
    assert left == right
    assert events(lp) == events(rp)
    assert left.status == "success"
    assert len(left.test_results) == 2
    assert all(item.status == "success" for item in left.test_results)
    assert list((lp / "optimizers").glob("*.py"))
    assert not list((lp / "optimizers").glob("*.json"))
    assert {"initial", "seeds", "round-000000", "round-000001"} <= {
        path.stem for path in (lp / "populations").glob("*.json")
    }
    trace = events(lp)
    assert any(event["event"] == "active_optimizer_changed" for event in trace)
    assert any(event["event"] == "llm_requested" for event in trace)
    assert all("timeout_seconds" not in event for event in trace)
    assert {
        event["profile"] for event in trace if event["event"] == "llm_initialized"
    } == {"heuristic", "meta"}
    data = json.loads((lp / "run.json").read_text())
    assert data["test_status"] == "success"
    assert data["provenance"]["solver"]["iterations"] == 3
    assert data["counts"]["heuristic_evaluations"] == left.counts.heuristic_evaluations
    for event in trace:
        if event["event"] in ("optimizer_worker_started", "heuristic_worker_started"):
            folder = (
                "optimizers" if event["event"].startswith("optimizer") else "heuristics"
            )
            source = (lp / folder / (event["id"] + ".py")).read_bytes()
            assert hashlib.sha256(source).hexdigest() == event["source_hash"]


def test_invalid_config_creates_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ProgramRecorder,
        "create",
        lambda *a, **k: pytest.fail("recorder before validation"),
    )
    monkeypatch.setattr(
        program_search,
        "FakeLLM",
        lambda *a, **k: pytest.fail("provider before validation"),
    )
    config = ProgramConfig(output_dir=tmp_path).model_copy(update={"seed": True})
    with pytest.raises(ValueError):
        program_search.run_program_experiment(config)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("corruption", ["missing", "reference", "duplicate_task"])
def test_dataset_errors_before_side_effects(tmp_path, monkeypatch, corruption):
    task = synthetic_task(4, 2, 1, 42)
    instances = (*task.validation, *task.test)
    arrays = {
        "coordinates": np.array([x.coordinates for x in instances]),
        "distance_matrix": np.array([x.distances for x in instances]),
        "cost": np.array([x.optimal_cost for x in instances]),
        "optimal_tour": np.array([x.optimal_tour[:-1] for x in instances]),
    }
    path = tmp_path / "data.npz"
    if corruption == "reference":
        arrays["cost"][0] *= 2
    if corruption != "missing":
        np.savez(path, **arrays)
    datasets = [{"path": path, "validation_indices": [0, 1], "test_indices": [2]}]
    if corruption == "duplicate_task":
        other = tmp_path / "other.npz"
        np.savez(other, **arrays)
        datasets.append({**datasets[0], "path": other})
    config = ProgramConfig(
        output_dir=tmp_path / "output", tasks={"source": "npz", "datasets": datasets}
    )
    monkeypatch.setattr(
        ProgramRecorder,
        "create",
        lambda *a, **k: pytest.fail("recorder before data validation"),
    )
    monkeypatch.setattr(
        program_search,
        "FakeLLM",
        lambda *a, **k: pytest.fail("provider before data validation"),
    )
    with pytest.raises((ValueError, OSError)):
        program_search.run_program_experiment(config)
    assert not config.output_dir.exists()


def test_npz_provenance_and_default_size_weights(tmp_path):
    task = synthetic_task(4, 1, 1, 42)
    instances = (*task.validation, *task.test)
    path = tmp_path / "data.npz"
    np.savez(
        path,
        coordinates=np.array([x.coordinates for x in instances]),
        distance_matrix=np.array([x.distances for x in instances]),
        cost=np.array([x.optimal_cost for x in instances]),
        optimal_tour=np.array([x.optimal_tour[:-1] for x in instances]),
    )
    config = ProgramConfig(
        output_dir=tmp_path / "output",
        outer_iterations=0,
        seed_attempts=0,
        tasks={
            "source": "npz",
            "datasets": [
                {"path": path, "validation_indices": [0], "test_indices": [1]}
            ],
        },
    )
    result, run_path = program_search.run_program_experiment(config)
    data = json.loads((run_path / "run.json").read_text())
    assert result.status == "success"
    assert (
        data["provenance"]["tasks"][0]["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert data["provenance"]["weights"] == [1.0]


def test_zero_budget_search_fails_cleanly(tmp_path):
    result, path = program_search.run_program_experiment(
        small_config(
            tmp_path, budgets={"max_llm_calls": 0, "max_heuristic_evaluations": 0}
        )
    )
    assert result.status == "failed"
    assert result.winner is None and not result.test_results
    assert result.counts == WorkCounts()
    assert json.loads((path / "run.json").read_text())["test_status"] == "not_run"
    assert list((path / "optimizers").glob("*.py"))


def winner_result():
    task = synthetic_task(4, 1, 1, 42)
    selected = ScoredProgram(
        Heuristic("winner-heuristic", "selected-source"),
        GapEvaluation(
            "winner-heuristic",
            task.id,
            "validation",
            "success",
            0.0,
            (task.validation[0].optimal_cost,),
            (0.0,),
            (task.validation[0].optimal_tour,),
        ),
    )
    newest = ScoredProgram(
        Heuristic("latest-heuristic", "latest-source"),
        replace(selected.evaluation, candidate_id="latest-heuristic"),
    )
    program = OptimizerProgram("winner", "optimizer-source")
    winner = ScoredProgram(
        program,
        ProgramEvaluation("winner", "success", 0.0, (TaskOutcome(task.id, selected),)),
    )
    result = ProgramRunResult(
        "success",
        winner,
        winner,
        (winner,),
        {task.id: ProgramPopulation(1, (newest,))},
        (),
        WorkCounts(),
    )
    return task, selected, result


def test_held_out_uses_winner_selection_without_feedback():
    task, selected, result = winner_result()
    calls = []

    class Runner:
        def evaluate(self, heuristic, task, **kwargs):
            calls.append((heuristic, kwargs["split"]))
            return replace(
                selected.evaluation, split="test", counts=WorkCounts(1, 1, 0)
            )

    supervisor = ProcessSupervisor()
    deadline = Deadline.after(10)
    budget = WorkBudget(10, 10)
    with supervisor.scope(deadline) as scope:
        completed = program_search._evaluate_held_out(
            result,
            (task,),
            Runner(),
            budget,
            root_seed=42,
            deadline=deadline,
            scope=scope,
            emit=lambda *a: None,
        )
    assert calls == [(selected.candidate, "test")]
    assert completed.winner == result.winner
    assert completed.population == result.population
    assert completed.task_populations == result.task_populations
    assert completed.counts == WorkCounts(1, 1, 0)
    assert completed.test_results[0].status == "success"


def test_held_out_budget_failure_preserves_search_winner():
    task, selected, result = winner_result()

    class Runner:
        def evaluate(self, *args, **kwargs):
            pytest.fail("worker ran without budget")

    deadline = Deadline.after(10)
    supervisor = ProcessSupervisor()
    with supervisor.scope(deadline) as scope:
        completed = program_search._evaluate_held_out(
            result,
            (task,),
            Runner(),
            WorkBudget(0, 0),
            root_seed=42,
            deadline=deadline,
            scope=scope,
            emit=lambda *a: None,
        )
    assert completed.status == "success"
    assert completed.winner == result.winner
    assert completed.test_results[0].candidate_id == selected.id
    assert completed.test_results[0].status == "failed"
    assert completed.test_results[0].utility is None
    assert completed.test_results[0].counts == WorkCounts()
    assert completed.counts == WorkCounts()


def test_recording_failure_propagates_and_closes_recorder(tmp_path, monkeypatch):
    paths = []
    original = ProgramRecorder.create.__func__

    def create(cls, *args, **kwargs):
        recorder = original(cls, *args, **kwargs)
        paths.append(recorder)
        return recorder

    monkeypatch.setattr(ProgramRecorder, "create", classmethod(create))
    monkeypatch.setattr(
        ProgramRecorder,
        "save_heuristic",
        lambda *a: (_ for _ in ()).throw(OSError("disk failure")),
    )
    with pytest.raises(OSError, match="disk failure"):
        program_search.run_program_experiment(small_config(tmp_path))
    assert paths[0].events.closed
    assert json.loads((paths[0].path / "run.json").read_text())["status"] == "error"


def test_selected_provider_environment_validated_before_artifacts(
    tmp_path, monkeypatch
):
    from moh.llm import openai_client

    monkeypatch.setattr(
        openai_client,
        "validate_environment",
        lambda: (_ for _ in ()).throw(ValueError("missing key")),
    )
    monkeypatch.setattr(
        ProgramRecorder,
        "create",
        lambda *a, **k: pytest.fail("artifacts before environment validation"),
    )
    config = small_config(
        tmp_path, meta_llm={"provider": "openai", "model": "explicit"}
    )
    with pytest.raises(ValueError, match="missing key"):
        program_search.run_program_experiment(config)


def test_held_out_incomplete_metadata_keeps_successful_search(tmp_path):
    result, path = program_search.run_program_experiment(
        small_config(
            tmp_path, budgets={"max_llm_calls": 100, "max_heuristic_evaluations": 5}
        )
    )
    assert result.status == "success"
    assert result.winner is not None
    assert result.test_results and result.test_results[0].status == "failed"
    assert result.test_results[0].error == "evaluation_budget"
    assert result.counts.heuristic_evaluations == 5
    data = json.loads((path / "run.json").read_text())
    assert data["test_status"] == "incomplete"
    assert data["test_counts"] == {
        "heuristic_evaluations": 0,
        "instance_attempts": 0,
        "llm_calls": 0,
    }


def test_expired_search_deadline_is_failed_not_held_out(tmp_path, monkeypatch):
    def expired_search(self, seeds, populations, meta_llm, **kwargs):
        object.__setattr__(kwargs["deadline"], "expires_at", 0.0)
        return ProgramRunResult(
            "failed", None, None, (), populations, (), self.budget.counts
        )

    monkeypatch.setattr(program_search.ProgramMeta, "search", expired_search)
    result, path = program_search.run_program_experiment(small_config(tmp_path))
    assert result.status == "failed"
    assert result.winner is None and not result.test_results
    assert json.loads((path / "run.json").read_text())["test_status"] == "not_run"


def test_initializer_receives_threshold_and_sources_exist_before_worker(
    tmp_path, monkeypatch
):
    initializer = program_search.initialize_task_population
    runner = program_search.GLSRunner.evaluate
    thresholds = []

    def initialize(*args, **kwargs):
        thresholds.append(kwargs["threshold"])
        return initializer(*args, **kwargs)

    def evaluate(self, heuristic, task, **kwargs):
        paths = list(tmp_path.glob("*/heuristics/" + heuristic.id + ".py"))
        assert len(paths) == 1
        assert paths[0].read_text() == heuristic.source_code
        return runner(self, heuristic, task, **kwargs)

    monkeypatch.setattr(program_search, "initialize_task_population", initialize)
    monkeypatch.setattr(program_search.GLSRunner, "evaluate", evaluate)
    result, _ = program_search.run_program_experiment(
        small_config(tmp_path, seed_threshold=0.0)
    )
    assert result.status == "success"
    assert thresholds == [0.0]


def test_independent_real_profile_routing_uses_fake_transport_only(
    tmp_path, monkeypatch
):
    from moh.llm import openai_client
    from moh.llm.fake import FakeLLM

    constructed, closed = [], []
    monkeypatch.setattr(openai_client, "validate_environment", lambda: None)

    class AdapterStub(FakeLLM):
        def __init__(self, model, timeout_seconds, observer):
            constructed.append((model, timeout_seconds))
            self.model = model
            super().__init__(42)

        def close(self):
            closed.append(self.model)

    monkeypatch.setattr(openai_client, "OpenAILLMClient", AdapterStub)
    result, path = program_search.run_program_experiment(
        small_config(
            tmp_path,
            heuristic_llm={
                "provider": "openai",
                "model": "heuristic-explicit",
                "timeout_seconds": 3.0,
            },
            meta_llm={
                "provider": "openai",
                "model": "meta-explicit",
                "timeout_seconds": 7.0,
            },
        )
    )
    assert result.status == "success"
    assert ("heuristic-explicit", 3.0) in constructed
    assert ("meta-explicit", 7.0) in constructed
    assert len(closed) == len(constructed)
    assert {
        event["model"] for event in events(path) if event["event"] == "llm_initialized"
    } == {"heuristic-explicit", "meta-explicit"}


def test_deadline_after_completed_search_preserves_winner(tmp_path, monkeypatch):
    _task, _selected, completed = winner_result()

    def search(self, seeds, populations, meta_llm, **kwargs):
        object.__setattr__(kwargs["deadline"], "expires_at", 0.0)
        return replace(completed, counts=self.budget.counts)

    monkeypatch.setattr(program_search.ProgramMeta, "search", search)
    result, path = program_search.run_program_experiment(small_config(tmp_path))
    assert result.status == "success"
    assert result.winner == completed.winner
    assert result.task_populations == completed.task_populations
    assert result.test_results[0].status == "failed"
    assert result.test_results[0].error == "timeout"
    assert json.loads((path / "run.json").read_text())["test_status"] == "incomplete"


def test_failed_search_preserves_committed_task_populations(tmp_path, monkeypatch):
    task, _selected, completed = winner_result()
    expected = ProgramRunResult(
        "failed", None, None, (), completed.task_populations, (), WorkCounts()
    )

    def search(self, seeds, populations, meta_llm, **kwargs):
        object.__setattr__(kwargs["deadline"], "expires_at", 0.0)
        return replace(expected, counts=self.budget.counts)

    monkeypatch.setattr(program_search.ProgramMeta, "search", search)
    result, path = program_search.run_program_experiment(small_config(tmp_path))
    assert result.status == "failed"
    assert result.task_populations == expected.task_populations
    persisted = json.loads((path / "run.json").read_text())["task_populations"]
    assert persisted[task.id]["members"][0]["candidate"]["id"] == "latest-heuristic"
