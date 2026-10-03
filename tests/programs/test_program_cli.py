import json

import pytest

from moh.main import main


def test_program_cli_success(tmp_path, capsys):
    config = tmp_path / "smoke.yaml"
    config.write_text(
        f"output_dir: {tmp_path / 'output'}\nouter_iterations: 0\nseed_attempts: 0\ntasks:\n  sizes: [4]\n  validation_count: 1\n"
    )
    assert main(["--mode", "moh", "--config", str(config)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "success"
    assert data["objective"] == "mean_gap_percent"
    assert data["winner"] and data["utility"] >= 0
    assert data["run_dir"]


def test_program_cli_failed_search(tmp_path, capsys):
    config = tmp_path / "failed.yaml"
    config.write_text(
        f"output_dir: {tmp_path / 'output'}\nouter_iterations: 0\nbudgets:\n  max_llm_calls: 0\n  max_heuristic_evaluations: 0\n"
    )
    assert main(["--mode", "moh", "--config", str(config)]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "failed"
    assert data["winner"] is data["utility"] is None


@pytest.mark.parametrize(
    "text", ["seed: true\n", "seed: 1\nseed: 2\n", "tasks: [invalid\n"]
)
def test_program_cli_invalid_config(tmp_path, capsys, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    assert main(["--mode", "moh", "--config", str(path)]) == 2
    output = capsys.readouterr()
    assert "error:" in output.err
    assert not output.out


def test_program_cli_missing_dataset(tmp_path, capsys):
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "tasks:\n  source: npz\n  datasets:\n    - path: nonexistent.npz\n      validation_indices: [0]\n      test_indices: [1]\n"
    )
    assert main(["--mode", "moh", "--config", str(config)]) == 2
    assert "error:" in capsys.readouterr().err
