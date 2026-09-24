from pathlib import Path

import pytest
from pydantic import ValidationError

from moh.config import ExperimentConfig, load_config


@pytest.mark.parametrize(
    "update",
    [
        {"unknown": 1},
        {"seed": True},
        {"seed": -1},
        {"outer": {"iterations": -1}},
        {"outer": {"population_size": 1}},
        {"inner": {"population_size": 11}},
        {"inner": {"iterations": True}},
        {"tasks": {"sizes": [10, 10]}},
        {"tasks": {"sizes": []}},
        {"tasks": {"sizes": [30]}},
        {"tasks": {"sizes": [True]}},
        {"tasks": {"weights": [1.0]}},
        {"tasks": {"weights": [1.0, float("nan")]}},
        {"tasks": {"weights": [0.0, 1.0]}},
        {"instances_per_task": 0},
        {"execution": {"timeout_seconds": 0}},
        {"execution": {"result_bytes": True}},
        {"llm": {"timeout_seconds": float("inf")}},
        {"llm": {"provider": "openai"}},
        {"llm": {"provider": "openai", "model": " "}},
    ],
)
def test_invalid_config(update):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(update)


def test_defaults_and_duplicates(tmp_path):
    config = ExperimentConfig()
    assert config.tasks.weights == [1.0, 1.0]
    assert config.seed == 42
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nseed: 2\n")
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text("output_dir: other\n")
    assert load_config(path).output_dir == Path("other")
