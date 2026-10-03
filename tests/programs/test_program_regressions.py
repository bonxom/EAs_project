"""Repository-wide boundaries and retained program lineage."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_generated_code_execution_stays_in_worker_modules():
    allowed = {
        "execution/worker.py",
        "execution/optimizer_worker.py",
        "execution/gls_worker.py",
    }
    calls = set()
    for path in (ROOT / "src/moh").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"exec", "compile"}
            ):
                relative = path.relative_to(ROOT / "src/moh").as_posix()
                assert relative in allowed, relative
                calls.add(relative)
    assert calls == allowed


def test_seed_loader_reads_source_without_importing_program_modules():
    path = ROOT / "src/moh/optimizers/seeds/__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.rsplit(".", 1)[-1]
        in {"basic", "multi_temperature", "best_parent"}
        for node in ast.walk(tree)
    )


def test_worker_sources_and_population_entries_are_retained_and_assessed(tmp_path):
    from moh.experiments import program_search
    from moh.program_config import ProgramConfig

    result, path = program_search.run_program_experiment(ProgramConfig(output_dir=tmp_path))
    assert result.status == "success"
    events = [
        json.loads(line)
        for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assessed = {"inner": set(), "outer": set()}
    worker_events = []
    for event in events:
        if (
            event["event"] in {"heuristic_evaluated", "optimizer_evaluated"}
            and event["evaluation"]["status"] == "success"
        ):
            level = "inner" if event["event"] == "heuristic_evaluated" else "outer"
            assessed[level].add(event["id"])
        if event["event"] in {"heuristic_worker_started", "optimizer_worker_started"}:
            worker_events.append(event)
            folder = (
                "heuristics"
                if event["event"] == "heuristic_worker_started"
                else "optimizers"
            )
            code = (path / folder / f"{event['id']}.py").read_text(encoding="utf-8")
            assert hashlib.sha256(code.encode("utf-8")).hexdigest() == event["source_hash"]
        if event["event"] == "population_updated":
            assert set(event["ids"]) <= assessed[event["level"]]
    assert worker_events
    for selected in (*result.population, result.active):
        assert selected.id in assessed["outer"]
        assert (path / "optimizers" / f"{selected.id}.py").read_text() == selected.source_code
        assert all(
            outcome.selected.evaluation.status == "success"
            and outcome.selected.evaluation.split == "validation"
            for outcome in selected.evaluation.task_results
        )
    assert all(
        member.evaluation.split == "validation"
        for population in result.task_populations.values()
        for member in population.members
    )
