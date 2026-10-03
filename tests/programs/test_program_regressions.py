"""Repository-wide boundaries and retained program lineage."""

import ast
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
