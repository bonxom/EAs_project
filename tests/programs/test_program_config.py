from pathlib import Path

import pytest

from moh.program_config import ProgramConfig, load_program_config


def test_defaults_and_normalized_weights():
    config = ProgramConfig()
    assert config.tasks.sizes == [4, 6]
    assert config.weights == [0.4, 0.6]
    assert config.heuristic_llm.provider == config.meta_llm.provider == "fake"
    assert config.solver.iterations == 3
    assert config.execution.batch_size == 2
    assert config.budgets.max_llm_calls == 100
    assert (
        ProgramConfig(
            outer_iterations=0,
            seed_attempts=0,
            budgets={"max_llm_calls": 0, "max_heuristic_evaluations": 0},
        ).outer_iterations
        == 0
    )


@pytest.mark.parametrize(
    "values",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"unknown": 1},
        {"population_size": 0},
        {"outer_iterations": -1},
        {"seed_attempts": False},
        {"weights": [0.0, 1.0]},
        {"weights": [1.0]},
        {"weights": [float("inf"), 1.0]},
        {"tasks": {"sizes": [4, 4]}},
        {"tasks": {"sizes": [13]}},
        {"tasks": {"sizes": [True]}},
        {"tasks": {"validation_count": 0}},
        {"execution": {"source_bytes": 1.5}},
        {"execution": {"batch_size": True}},
        {"execution": {"run_timeout_seconds": 10**400}},
        {"execution": {"optimizer_timeout_seconds": 1e30}},
        {"execution": {"heuristic_timeout_seconds": float("nan")}},
        {"heuristic_llm": {"timeout_seconds": 10**400}},
        {"meta_llm": {"provider": "openai"}},
        {"solver": {"iterations": -1}},
        {"budgets": {"max_llm_calls": -1}},
    ],
)
def test_invalid_fields(values):
    with pytest.raises(ValueError):
        ProgramConfig(**values)


def test_model_copy_and_nested_mutation_are_revalidated():
    config = ProgramConfig().model_copy(update={"seed": True})
    with pytest.raises(ValueError):
        ProgramConfig.model_validate(config.model_dump(mode="python"))
    config = ProgramConfig()
    config.tasks.sizes.append(13)
    with pytest.raises(ValueError):
        ProgramConfig.model_validate(config.model_dump(mode="python"))


def test_dataset_settings_and_independent_model_names(tmp_path):
    config = ProgramConfig(
        tasks={
            "source": "npz",
            "datasets": [
                {
                    "path": str(tmp_path / "100.npz"),
                    "validation_indices": [0, 1],
                    "test_indices": [2],
                }
            ],
        },
        heuristic_llm={"provider": "openai", "model": "inner"},
        meta_llm={"provider": "openai", "model": "outer"},
        weights=[2.0],
    )
    assert config.weights == [1.0]
    assert config.tasks.datasets[0].path == tmp_path / "100.npz"
    assert config.meta_llm.model == "outer"


@pytest.mark.parametrize(
    "dataset",
    [
        {"validation_indices": [0], "test_indices": [0]},
        {"validation_indices": [], "test_indices": [1]},
        {"validation_indices": [0, 0], "test_indices": [1]},
        {"validation_indices": [True], "test_indices": [1]},
    ],
)
def test_invalid_dataset_splits(dataset):
    with pytest.raises(ValueError):
        ProgramConfig(
            tasks={"source": "npz", "datasets": [{"path": "x.npz", **dataset}]}
        )


def test_yaml_loading_and_templates(tmp_path):
    config = load_program_config(Path("configs/moh_smoke.yaml"))
    assert config.model_dump() == ProgramConfig().model_dump()
    config = load_program_config(Path("configs/moh_dataset.yaml"))
    assert config.tasks.source == "npz"
    assert len(config.tasks.datasets) == 2
    assert config.solver.iterations == 1000
    assert config.heuristic_llm.provider == config.meta_llm.provider == "fake"
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\nseed: 2\n")
    with pytest.raises(ValueError):
        load_program_config(bad)


def test_weight_numeric_limits_and_round_trip():
    config = ProgramConfig(weights=[1e308, 1e308])
    assert config.weights == [0.5, 0.5]
    assert (
        ProgramConfig.model_validate(config.model_dump()).model_dump()
        == config.model_dump()
    )
    config = ProgramConfig(
        budgets={"max_llm_calls": None, "max_heuristic_evaluations": None}
    )
    assert config.budgets.max_llm_calls is None
    with pytest.raises(ValueError):
        ProgramConfig(weights=[1e-300, 1e300])


@pytest.mark.parametrize(
    "values",
    [
        {"tasks": {"source": "npz"}},
        {
            "tasks": {
                "datasets": [
                    {"path": "x", "validation_indices": [0], "test_indices": [1]}
                ]
            }
        },
        {"execution": {"source_bytes": 0}},
        {"execution": {"request_bytes": 2**31}},
        {"execution": {"optimizer_timeout_seconds": False}},
        {"solver": {"reset_interval": 0}},
        {"solver": {"iterations": True}},
        {"meta_llm": {"timeout_seconds": 0}},
    ],
)
def test_additional_boundary_values(values):
    with pytest.raises(ValueError):
        ProgramConfig(**values)
