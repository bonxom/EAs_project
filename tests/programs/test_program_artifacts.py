import json
from types import MappingProxyType

import pytest

from moh.core.models import Heuristic, WorkCounts
from moh.core.program_population import ProgramPopulation
from moh.core.programs import OptimizerProgram, ProgramRunResult
from moh.experiments.artifacts import ProgramRecorder
from moh.logging import to_json


def test_serialization_immutable_record_mapping():
    result = ProgramRunResult(
        "failed", None, None, (), {"task": ProgramPopulation(3)}, (), WorkCounts()
    )
    assert to_json(result)["task_populations"] == {
        "task": {"capacity": 3, "members": []}
    }
    assert to_json(MappingProxyType({"nested": (1, 2)})) == {"nested": [1, 2]}


def test_source_artifacts_checkpoint_and_provenance(tmp_path):
    with ProgramRecorder.create(
        tmp_path, {"seed": 42}, provenance={"tasks": ["tsp4"]}
    ) as recorder:
        optimizer = OptimizerProgram("optimizer.v1", "source")
        recorder.save_optimizer(optimizer)
        recorder.save_heuristic(Heuristic("heuristic.v1", "heuristic"))
        recorder.save_optimizer(optimizer)
        recorder.checkpoint("initial", {"tsp4": ProgramPopulation(3)}, None, None)
        recorder.emit("attempt", {"source": "source"})
        result = ProgramRunResult(
            "failed", None, None, (), {"tsp4": ProgramPopulation(3)}, (), WorkCounts()
        )
        recorder.finish(result)
    assert (recorder.path / "optimizers/optimizer.v1.py").read_text() == "source"
    assert not list((recorder.path / "optimizers").glob("*.json"))
    assert (
        json.loads((recorder.path / "populations/initial.json").read_text())["active"]
        is None
    )
    data = json.loads((recorder.path / "run.json").read_text())
    assert data["schema_version"] == 2
    assert data["provenance"] == {"tasks": ["tsp4"]}
    events = [
        json.loads(x) for x in (recorder.path / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["schema_version"] == 2
    assert "timestamp" not in events[0]


def test_safe_ids_collisions_validation_and_errors(tmp_path):
    with ProgramRecorder.create(tmp_path, {}) as recorder:
        with pytest.raises(ValueError):
            recorder.save_heuristic(Heuristic("../escape", "source"))
        recorder.save_optimizer(OptimizerProgram("o", "first"))
        with pytest.raises(ValueError, match="collision"):
            recorder.save_optimizer(OptimizerProgram("o", "second"))
        with pytest.raises(ValueError):
            recorder.checkpoint("../escape", {}, None, None)
        with pytest.raises(ValueError):
            recorder.emit("bad", {"sequence": 1})
        with pytest.raises(ValueError):
            recorder.emit("bad", {"utility": float("nan")})
        recorder.fail(ValueError("problem"))
    assert json.loads((recorder.path / "run.json").read_text())["error"] == "problem"
    blocked = tmp_path / "file"
    blocked.write_text("file")
    with pytest.raises(OSError):
        ProgramRecorder.create(blocked, {})
