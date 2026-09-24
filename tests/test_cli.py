import subprocess
import sys

from moh.main import main


def test_invalid_config(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: true\n")
    assert main(["--config", str(path)]) == 2
    assert capsys.readouterr().err


def test_demo(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text("instances_per_task: 1\n")
    result = subprocess.run(
        [sys.executable, "-m", "moh.main", "--config", str(path), "--mode", "tsp-demo"],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "nearest" in result.stdout
        and "random" in result.stdout
        and "success" in result.stdout
    )
