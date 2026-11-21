import os
import subprocess
from pathlib import Path

import pytest

MOH_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = "run_controller_experiment.sh"
SCRIPT = MOH_ROOT / SCRIPT_NAME

FAKE_UV = r"""#!/usr/bin/env bash
set -euo pipefail
{
  printf '%s\n' "$@"
  printf '%s\n' '---'
} >> "$FAKE_UV_LOG"
if [ "${1:-}" = "run" ] && [ "${2:-}" = "python" ]; then
  case "${3:-}" in
    main.py)
      for arg in "$@"; do
        case "$arg" in
          hydra.run.dir=*)
            dir="${arg#hydra.run.dir=}"
            mkdir -p "$dir/checkpoint"
            printf '{"training_meta_utility": 0.1}\n' > "$dir/checkpoint/metadata.json"
            ;;
        esac
      done
      ;;
    deploy.py)
      for arg in "$@"; do
        case "$arg" in
          hydra.run.dir=*)
            dir="${arg#hydra.run.dir=}"
            mkdir -p "$dir"
            printf '[]\n' > "$dir/report.json"
            ;;
        esac
      done
      ;;
  esac
fi
exit 0
"""


def _project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    # The repository stores LF; a core.autocrlf checkout may materialise CRLF,
    # which bash 3.2 cannot parse. Exercise the committed script content.
    script_text = SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n")
    (project / SCRIPT_NAME).write_text(script_text, encoding="utf-8")
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(FAKE_UV, encoding="utf-8")
    uv.chmod(0o755)
    return project, fake_bin


def _env(tmp_path, fake_bin):
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment.get('PATH', '')}"
    environment["FAKE_UV_LOG"] = str(tmp_path / "uv.log")
    environment["LOCAL_LLM_API_KEY"] = "offline-test-key"
    return environment


def _run(project, tmp_path, fake_bin, *arguments, environment=None):
    return subprocess.run(
        ["bash", str(project / SCRIPT_NAME), *arguments],
        cwd=project,
        env=environment if environment is not None else _env(tmp_path, fake_bin),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _calls(log_path, script_name):
    log_path = Path(log_path)
    if not log_path.is_file():
        return []
    entries, current = [], []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line == "---":
            if current:
                entries.append(current)
            current = []
        else:
            current.append(line)
    return [entry for entry in entries if script_name in entry]


def _make_checkpoint(out_dir, mode, seed):
    checkpoint = out_dir / f"{mode}_seed{seed}" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")
    return checkpoint


@pytest.mark.skipif(os.name != "posix", reason="the experiment runner is a bash script")
def test_deploy_stage_runs_tsp500_1000_for_each_checkpoint(tmp_path):
    project, fake_bin = _project(tmp_path)
    out_dir = tmp_path / "controller-exp"
    modes = (
        "dual_hifo_independent",
        "dual_hifo_global_rule",
        "dual_hifo_global_bandit",
    )
    for mode in modes:
        _make_checkpoint(out_dir, mode, 0)

    result = _run(
        project,
        tmp_path,
        fake_bin,
        "--stage",
        "deploy",
        "--seeds",
        "0",
        "--out-dir",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    deploy_calls = _calls(tmp_path / "uv.log", "deploy.py")
    assert len(deploy_calls) == len(modes)
    checkpoints = set()
    for call in deploy_calls:
        assert "target_sizes=[500,1000]" in call
        assert "inference_iterations=10" in call
        assert "val_n_instances=8" in call
        assert "test_n_instances=100" in call
        assert "workers=8" in call
        assert "evaluate_test=true" in call
        assert "seed=0" in call
        assert "llm_client@heu=local" in call
        assert "heu.base_url=http://172.25.16.1:20128/v1" in call
        checkpoint = next(arg for arg in call if arg.startswith("checkpoint="))
        hydra_dir = next(arg for arg in call if arg.startswith("hydra.run.dir="))
        checkpoints.add(Path(checkpoint.split("=", 1)[1]).parent.name)
        assert Path(hydra_dir.split("=", 1)[1]).parent == out_dir / "deployment"
    assert checkpoints == {f"{mode}_seed0" for mode in modes}

    for mode in modes:
        assert (
            out_dir / f"{mode}_seed0" / "checkpoint" / "metadata.json"
        ).is_file(), "the runner must not delete trained checkpoints"
        assert (out_dir / "deployment" / f"{mode}_seed0" / "report.json").is_file()


@pytest.mark.skipif(os.name != "posix", reason="the experiment runner is a bash script")
def test_deploy_stage_skips_completed_runs_and_missing_checkpoints(tmp_path):
    project, fake_bin = _project(tmp_path)
    out_dir = tmp_path / "controller-exp"
    _make_checkpoint(out_dir, "dual_hifo_global_rule", 0)
    arguments = (
        "--stage",
        "deploy",
        "--seeds",
        "0",
        "--modes",
        "dual_hifo_global_rule dual_hifo_global_bandit",
        "--out-dir",
        str(out_dir),
    )

    first = _run(project, tmp_path, fake_bin, *arguments)
    assert first.returncode == 0, first.stderr
    assert "checkpoint" in first.stderr
    assert len(_calls(tmp_path / "uv.log", "deploy.py")) == 1

    second = _run(project, tmp_path, fake_bin, *arguments)
    assert second.returncode == 0, second.stderr
    assert "already complete" in second.stdout
    assert len(_calls(tmp_path / "uv.log", "deploy.py")) == 1


@pytest.mark.skipif(os.name != "posix", reason="the experiment runner is a bash script")
def test_incomplete_deploy_directory_is_not_deleted(tmp_path):
    project, fake_bin = _project(tmp_path)
    out_dir = tmp_path / "controller-exp"
    _make_checkpoint(out_dir, "dual_hifo_global_rule", 0)
    marker = out_dir / "deployment" / "dual_hifo_global_rule_seed0" / "partial.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("inspect me", encoding="utf-8")

    result = _run(
        project,
        tmp_path,
        fake_bin,
        "--stage",
        "deploy",
        "--seeds",
        "0",
        "--modes",
        "dual_hifo_global_rule",
        "--out-dir",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "inspect it before re-running" in result.stderr
    assert marker.read_text(encoding="utf-8") == "inspect me"
    assert _calls(tmp_path / "uv.log", "deploy.py") == []


@pytest.mark.skipif(os.name != "posix", reason="the experiment runner is a bash script")
def test_train_stage_keeps_equal_budget_and_never_deploys(tmp_path):
    project, fake_bin = _project(tmp_path)
    out_dir = tmp_path / "controller-exp"

    result = _run(
        project,
        tmp_path,
        fake_bin,
        "--stage",
        "train",
        "--seeds",
        "1",
        "--modes",
        "dual_hifo_global_rule",
        "--out-dir",
        str(out_dir),
        "--max-requests",
        "123",
        "--n-iterations",
        "4",
    )

    assert result.returncode == 0, result.stderr
    train_calls = _calls(tmp_path / "uv.log", "main.py")
    assert len(train_calls) == 1
    call = train_calls[0]
    assert "dual_hifo.mode=dual_hifo_global_rule" in call
    assert "seed=1" in call
    assert "n_iterations=4" in call
    assert "provider_budget.max_requests=123" in call
    assert _calls(tmp_path / "uv.log", "deploy.py") == []
    assert (out_dir / "dual_hifo_global_rule_seed1" / "checkpoint" / "metadata.json").is_file()


@pytest.mark.skipif(os.name != "posix", reason="the experiment runner is a bash script")
def test_missing_local_api_key_fails_before_creating_run_directories(tmp_path):
    project, fake_bin = _project(tmp_path)
    out_dir = tmp_path / "controller-exp"
    environment = _env(tmp_path, fake_bin)
    environment.pop("LOCAL_LLM_API_KEY", None)

    result = _run(
        project,
        tmp_path,
        fake_bin,
        "--stage",
        "deploy",
        "--out-dir",
        str(out_dir),
        environment=environment,
    )

    assert result.returncode != 0
    assert "LOCAL_LLM_API_KEY" in result.stderr
    assert not out_dir.exists()
    assert _calls(tmp_path / "uv.log", "deploy.py") == []


@pytest.mark.skipif(os.name != "posix", reason="the experiment runner is a bash script")
def test_unknown_stage_is_rejected(tmp_path):
    project, fake_bin = _project(tmp_path)

    result = _run(project, tmp_path, fake_bin, "--stage", "nonsense")

    assert result.returncode != 0
    assert "--stage" in result.stderr
