import json

import pytest

from moh.core.models import Heuristic, OptimizerCandidate, RunResult
from moh.core.specs import OptimizerSpec
from moh.logging import RunRecorder


def test_artifacts_and_sequence(tmp_path, base_spec, monkeypatch):
    monkeypatch.setenv("SENTINEL_SECRET", "do-not-persist-me")
    with RunRecorder.create(tmp_path, {"seed": 42}) as recorder:
        other = RunRecorder.create(tmp_path, {"seed": 42})
        assert other.path != recorder.path
        other.close()
        recorder.save_heuristic(Heuristic("h1", "source"))
        recorder.save_optimizer(OptimizerCandidate("o1", OptimizerSpec(**base_spec)))
        recorder.emit("hello", {"seed": 42})
        recorder.emit("next", {})
        recorder.finish(RunResult("failed", None, ()))
    assert (recorder.path / "heuristics/h1.py").read_text() == "source"
    assert json.loads((recorder.path / "optimizers/o1.json").read_text()) == base_spec
    events = [
        json.loads(line)
        for line in (recorder.path / "events.jsonl").read_text().splitlines()
    ]
    assert [event["sequence"] for event in events] == [0, 1]
    assert json.loads((recorder.path / "run.json").read_text())["status"] == "failed"
    assert "do-not-persist-me" not in "".join(
        p.read_text() for p in recorder.path.rglob("*") if p.is_file()
    )


@pytest.mark.parametrize(
    "payload", [{"utility": float("nan")}, {"sequence": 3}, {"event": "fake"}]
)
def test_bad_event(tmp_path, payload):
    with RunRecorder.create(tmp_path, {}) as recorder, pytest.raises(ValueError):
        recorder.emit("bad", payload)


def test_errors_and_redaction(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("x")
    with pytest.raises(OSError):
        RunRecorder.create(blocked, {})
    with RunRecorder.create(
        tmp_path / "good", {}, redactions=("secret-token",)
    ) as recorder:
        recorder.emit("error", {"error": "secret-token failed"})
        assert "secret-token" not in (recorder.path / "events.jsonl").read_text()
        with pytest.raises(ValueError):
            recorder.save_heuristic(Heuristic("../escape", "bad"))
        recorder.close()
        with pytest.raises(ValueError):
            recorder.emit("closed", {})
