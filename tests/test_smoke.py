import json

import pytest

from moh.config import ExperimentConfig
from moh.core.models import EvaluationResult
from moh.experiment import run_experiment


def events(path):
    return [
        json.loads(x, parse_constant=lambda _: pytest.fail("nonfinite JSON"))
        for x in (path / "events.jsonl").read_text().splitlines()
    ]


def test_smoke_reproducible(tmp_path):
    config = ExperimentConfig(output_dir=tmp_path / "a")
    left, lp = run_experiment(config)
    right, rp = run_experiment(config.model_copy(update={"output_dir": tmp_path / "b"}))
    assert left == right
    assert events(lp) == events(rp)
    assert left.status == "success"
    assert len([x for x in events(lp) if x["event"] == "outer_generation_started"]) == 2
    evaluated = set()
    for event in events(lp):
        if event["event"] == "heuristic_evaluated":
            assert (lp / "heuristics" / f"{event['candidate_id']}.py").is_file()
        if event["event"] == "optimizer_evaluated":
            assert (lp / "optimizers" / f"{event['optimizer_id']}.json").is_file()
            evaluated.add(event["optimizer_id"])
        if event["event"] == "population_updated" and event["level"] == "outer":
            assert set(event["ids"]) <= evaluated
    assert json.loads((lp / "run.json").read_text())["status"] == "success"
    assert (lp / "config.yaml").is_file()
    for event in events(lp):
        if event["event"] == "heuristic_generated":
            assert (lp / "heuristics" / f"{event['id']}.py").read_text() == event[
                "source_code"
            ]
        if event["event"] == "optimizer_generated":
            assert (
                json.loads((lp / "optimizers" / f"{event['id']}.json").read_text())
                == event["spec"]
            )


def test_invalid_child_continues(tmp_path, monkeypatch):
    from moh.llm.fake import FakeLLM

    original = FakeLLM.generate

    def bad_code(self, prompt):
        return (
            "invalid code !"
            if prompt.startswith(("KIND: mutate", "KIND: crossover"))
            else original(self, prompt)
        )

    monkeypatch.setattr(FakeLLM, "generate", bad_code)
    result, path = run_experiment(
        ExperimentConfig(output_dir=tmp_path, instances_per_task=1)
    )
    assert result.status == "success"
    assert any(
        x["event"] == "heuristic_evaluated" and x["status"] == "failed"
        for x in events(path)
    )


def test_all_failed_and_zero(tmp_path, monkeypatch):
    from moh.execution.heuristic_runner import HeuristicRunner

    def fail(self, heuristic, task, context):
        return EvaluationResult(
            heuristic.id, context, "failed", None, (), "test_failure", 1
        )

    monkeypatch.setattr(HeuristicRunner, "evaluate", fail)
    result, path = run_experiment(
        ExperimentConfig.model_validate(
            {
                "output_dir": tmp_path,
                "inner": {"iterations": 0},
                "outer": {"iterations": 0},
            }
        )
    )
    assert result.status == "failed"
    assert not any(x["event"] == "llm_requested" for x in events(path))
    assert json.loads((path / "run.json").read_text())["status"] == "failed"


def test_invalid_config_has_no_side_effects(tmp_path, monkeypatch):
    from moh.logging import RunRecorder

    def forbid(*args, **kwargs):
        pytest.fail("created artifacts before validation")

    monkeypatch.setattr(RunRecorder, "create", forbid)
    invalid = ExperimentConfig().model_copy(update={"seed": True})
    with pytest.raises(ValueError):
        run_experiment(invalid)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    with pytest.raises(ValueError):
        run_experiment(
            ExperimentConfig.model_validate(
                {"llm": {"provider": "openai", "model": "test"}}
            )
        )


def test_output_failure(tmp_path):
    path = tmp_path / "file"
    path.write_text("occupied")
    with pytest.raises(OSError):
        run_experiment(ExperimentConfig(output_dir=path))
