import os
import subprocess
import sys
from pathlib import Path

import pytest
from deployment_fakes import (
    HEURISTIC,
    OPTIMIZER,
    FakeEvaluator,
    FakeLLM,
    training_population,
)
from omegaconf import OmegaConf
from utils.checkpoint import export_checkpoint, load_checkpoint


def _cfg(checkpoint="unused", targets=(500,), **overrides):
    values = {
        "checkpoint": checkpoint,
        "target_sizes": list(targets),
        "inference_iterations": 1,
        "population_size": 2,
        "seed_count": 1,
        "initialization_attempts": 2,
        "reference_count": 1,
        "val_n_instances": 8,
        "test_n_instances": 100,
        "max_evaluations_per_target": 20,
        "seed": 0,
        "optimizer_timeout": 10,
        "instance_timeout": 2,
        "evaluation_timeout": 10,
        "workers": 2,
        "evaluate_test": True,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _checkpoint_dir(tmp_path):
    return export_checkpoint(
        tmp_path / "checkpoint",
        OPTIMIZER,
        training_population((20,)),
        {"training_sizes": [20], "seed": 0},
    )


def test_paths_resolve_from_original_cwd(tmp_path, monkeypatch):
    from deploy import run_deployment

    original = tmp_path / "original"
    original.mkdir()
    checkpoint_dir = _checkpoint_dir(original)
    output = tmp_path / "hydra-output"
    output.mkdir()
    monkeypatch.chdir(output)
    monkeypatch.setattr(
        "hydra.utils.to_absolute_path",
        lambda path: str(original / path),
    )
    llm = FakeLLM([HEURISTIC, HEURISTIC])
    rows = run_deployment(
        _cfg(checkpoint=checkpoint_dir.name),
        llm,
        evaluator=FakeEvaluator({HEURISTIC: 1.0}),
    )
    assert rows[0]["status"] == "ok"
    assert (output / "report.json").is_file()
    assert llm.messages


def test_bad_checkpoint_prevents_llm_calls(tmp_path, monkeypatch):
    from deploy import run_deployment

    checkpoint_dir = _checkpoint_dir(tmp_path)
    (checkpoint_dir / "meta_optimizer.py").write_text("# tampered")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.chdir(output)
    llm = FakeLLM()
    with pytest.raises(ValueError, match="hash"):
        run_deployment(
            _cfg(checkpoint=str(checkpoint_dir)),
            llm,
            evaluator=FakeEvaluator(),
        )
    assert llm.messages == []


def test_missing_test_data_prevents_llm_calls(tmp_path, monkeypatch):
    from deploy import run_deployment

    class MissingTestEvaluator(FakeEvaluator):
        def preflight(self, sizes, include_test):
            assert include_test is True
            raise ValueError("missing test instances")

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.chdir(output)
    llm = FakeLLM()
    with pytest.raises(ValueError, match="missing test"):
        run_deployment(
            _cfg(), llm, evaluator=MissingTestEvaluator(), checkpoint=load_checkpoint(_checkpoint_dir(tmp_path))
        )
    assert llm.messages == []


def test_two_target_run_exports_both_reports(tmp_path, monkeypatch):
    from deploy import run_deployment

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.chdir(output)
    checkpoint = load_checkpoint(_checkpoint_dir(tmp_path))
    rows = run_deployment(
        _cfg(targets=(500, 1000)),
        FakeLLM([HEURISTIC] * 6),
        evaluator=FakeEvaluator({HEURISTIC: 1.0}),
        checkpoint=checkpoint,
    )
    assert [row["size"] for row in rows] == [500, 1000]
    assert (output / "heuristics" / "tsp500.py").is_file()
    assert (output / "heuristics" / "tsp1000.py").is_file()


def test_cli_config_can_request_128_without_silent_reduction(tmp_path, monkeypatch):
    from deploy import run_deployment

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.chdir(output)
    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    evaluator.test_count = 128
    rows = run_deployment(
        _cfg(test_n_instances=128),
        FakeLLM([HEURISTIC, HEURISTIC]),
        evaluator=evaluator,
        checkpoint=load_checkpoint(_checkpoint_dir(tmp_path)),
    )
    assert rows[0]["test_n_instances"] == 128
    assert len(evaluator.preflight([500], include_test=True)[500]["test"]["indices"]) == 128


def test_validation_only_preflight_never_requires_test(tmp_path, monkeypatch):
    from deploy import run_deployment

    class ValidationOnlyEvaluator(FakeEvaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.preflight_modes = []

        def preflight(self, sizes, include_test):
            self.preflight_modes.append(include_test)
            return super().preflight(sizes, include_test)

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.chdir(output)
    evaluator = ValidationOnlyEvaluator({HEURISTIC: 1.0})
    rows = run_deployment(
        _cfg(evaluate_test=False),
        FakeLLM([HEURISTIC, HEURISTIC]),
        evaluator=evaluator,
        checkpoint=load_checkpoint(_checkpoint_dir(tmp_path)),
    )
    assert rows[0]["status"] == "validation_only"
    assert evaluator.preflight_modes[0] is False
    assert all(call[2] == "val" for call in evaluator.calls)


def test_offline_subprocess_smoke_writes_artifacts(tmp_path):
    checkpoint_dir = _checkpoint_dir(tmp_path)
    output = tmp_path / "smoke-output"
    output.mkdir()
    before = (checkpoint_dir / "meta_optimizer.py").read_bytes()
    script = f'''from omegaconf import OmegaConf
from deploy import run_deployment
from deployment_fakes import FakeEvaluator, FakeLLM, HEURISTIC
cfg = OmegaConf.create({{
    "checkpoint": {str(checkpoint_dir)!r}, "target_sizes": [500],
    "inference_iterations": 1, "population_size": 2, "seed_count": 1,
    "initialization_attempts": 2, "reference_count": 1,
    "val_n_instances": 8, "test_n_instances": 100,
    "max_evaluations_per_target": 20, "seed": 0,
    "optimizer_timeout": 10, "instance_timeout": 2,
    "evaluation_timeout": 10, "workers": 2, "evaluate_test": False,
}})
rows = run_deployment(cfg, FakeLLM([HEURISTIC, HEURISTIC]), evaluator=FakeEvaluator({{HEURISTIC: 1.0}}))
assert rows[0]["status"] == "validation_only"
'''
    environment = os.environ.copy()
    tests_path = str(Path(__file__).resolve().parent)
    project_path = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        [tests_path, project_path, environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=output,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "heuristics" / "tsp500.py").is_file()
    assert (output / "report.json").is_file()
    assert (checkpoint_dir / "meta_optimizer.py").read_bytes() == before


@pytest.mark.parametrize('fraction', [0, .3])
def test_cli_passes_diversity_fraction(tmp_path, monkeypatch, fraction):
    from deploy import run_deployment

    checkpoint = load_checkpoint(_checkpoint_dir(tmp_path))
    monkeypatch.chdir(tmp_path)
    rows = run_deployment(_cfg(population={'diversity_fraction': fraction}), FakeLLM(),
                          evaluator=FakeEvaluator(), checkpoint=checkpoint)
    assert rows[0]['selection_metadata']['population_policy']['diversity_fraction'] == fraction


def test_cli_rejects_bad_fraction_before_llm():
    from deploy import run_deployment

    llm = FakeLLM()
    with pytest.raises(ValueError, match='diversity_fraction'):
        run_deployment(_cfg(population={'diversity_fraction': float('nan')}), llm)
    assert llm.messages == []
