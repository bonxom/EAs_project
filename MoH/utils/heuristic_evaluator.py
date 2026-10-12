"""Parent-owned isolated evaluation service for generated heuristics."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from problems.tsp_gls.dataset_selection import load_eval_dataset


@dataclass(frozen=True)
class EvalResult:
    size: int
    mode: str
    mean_gap_pct: float | None
    status: str
    failures: int
    indices: tuple[int, ...]
    code_sha256: str
    dataset_sha256: str
    elapsed_seconds: float
    stdout_path: str
    error: str | None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attribute_path(node) -> tuple[str, ...]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _validate_deterministic_source(code: str, filename: str) -> None:
    tree = ast.parse(code, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(name.split(".")[0] in {"random", "secrets"} for name in names):
                raise ValueError("Heuristic must be deterministic; random APIs are forbidden")
        if isinstance(node, ast.Attribute):
            path = _attribute_path(node)
            if "random" in path or (path and path[0] == "secrets"):
                raise ValueError("Heuristic must be deterministic; random APIs are forbidden")


class HeuristicEvaluator:
    def __init__(
        self,
        root_dir,
        output_dir,
        seed=0,
        val_count=8,
        test_count=100,
        workers=8,
        instance_timeout=90,
        evaluation_timeout=None,
    ):
        if val_count <= 0 or test_count <= 0:
            raise ValueError("Evaluation counts must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if instance_timeout <= 0:
            raise ValueError("instance_timeout must be positive")
        if evaluation_timeout is not None and evaluation_timeout <= 0:
            raise ValueError("evaluation_timeout must be positive")
        self.root_dir = Path(root_dir).resolve()
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.val_count = int(val_count)
        self.test_count = int(test_count)
        self.workers = min(int(workers), 8)
        self.instance_timeout = int(instance_timeout)
        self.evaluation_timeout = evaluation_timeout
        self.solver_settings = {
            "workers": self.workers,
            "instance_timeout": self.instance_timeout,
            "time_limit": 20,
            "iteration_limit": 1000,
            "perturbation_moves": 1,
        }

    def _count(self, mode: str) -> int:
        if mode == "val":
            return self.val_count
        if mode == "test":
            return self.test_count
        raise ValueError(f"Unsupported evaluation mode: {mode!r}")

    def preflight(self, sizes: list[int], include_test: bool) -> dict:
        modes = ("val", "test") if include_test else ("val",)
        manifest = {}
        for size in sizes:
            mode_entries = {}
            for mode in modes:
                _data, indices, path = load_eval_dataset(
                    self.root_dir,
                    int(size),
                    mode,
                    self.seed,
                    self._count(mode),
                )
                mode_entries[mode] = {
                    "dataset_path": str(path.resolve()),
                    "dataset_sha256": _sha256_file(path),
                    "indices": tuple(indices),
                }
            manifest[int(size)] = mode_entries
        return manifest

    def _failed(
        self,
        *,
        size: int,
        mode: str,
        code_sha256: str,
        dataset_sha256: str,
        indices: tuple[int, ...],
        stdout_path: Path,
        started: float,
        error: str,
    ) -> EvalResult:
        return EvalResult(
            size=size,
            mode=mode,
            mean_gap_pct=None,
            status="failed",
            failures=0,
            indices=indices,
            code_sha256=code_sha256,
            dataset_sha256=dataset_sha256,
            elapsed_seconds=time.monotonic() - started,
            stdout_path=str(stdout_path),
            error=error,
        )

    @staticmethod
    def _stop_group(process, sig) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, sig)
                return
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    def evaluate(self, code: str, size: int, mode: str) -> EvalResult:
        started = time.monotonic()
        code_bytes = code.encode()
        code_digest = _sha256_bytes(code_bytes)
        count = self._count(mode)
        evaluations_dir = self.output_dir / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(
            tempfile.mkdtemp(prefix=f"tsp{size}-{mode}-", dir=evaluations_dir)
        )
        code_path = run_dir / "heuristic.py"
        result_path = run_dir / "result.json"
        stdout_path = run_dir / "stdout.txt"
        code_path.write_bytes(code_bytes)
        try:
            _validate_deterministic_source(code, str(code_path))
            manifest = self.preflight([size], include_test=mode == "test")[size][mode]
        except (OSError, ValueError, SyntaxError, TypeError) as error:
            stdout_path.write_text(str(error), encoding="utf-8")
            return self._failed(
                size=size,
                mode=mode,
                code_sha256=code_digest,
                dataset_sha256="",
                indices=(),
                stdout_path=stdout_path,
                started=started,
                error=str(error),
            )

        indices = manifest["indices"]
        dataset_digest = manifest["dataset_sha256"]
        workers = min(self.workers, count, os.cpu_count() or 1, 8)
        eval_path = self.root_dir / "problems" / "tsp_gls" / "eval.py"
        command = [
            sys.executable,
            "-u",
            str(eval_path),
            str(size),
            str(self.root_dir),
            mode,
            str(self.instance_timeout),
            str(self.seed),
            "--count",
            str(count),
            "--heuristic-path",
            str(code_path),
            "--result-path",
            str(result_path),
            "--workers",
            str(workers),
        ]
        timeout = self.evaluation_timeout or (
            math.ceil(count / workers) * self.instance_timeout + 30
        )
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=os.name == "posix",
            )
            try:
                stdout, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._stop_group(process, signal.SIGTERM)
                try:
                    stdout, _ = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    self._stop_group(process, signal.SIGKILL)
                    stdout, _ = process.communicate()
                stdout_path.write_text(stdout or "", encoding="utf-8")
                return self._failed(
                    size=size,
                    mode=mode,
                    code_sha256=code_digest,
                    dataset_sha256=dataset_digest,
                    indices=indices,
                    stdout_path=stdout_path,
                    started=started,
                    error=f"Evaluation timed out after {timeout}s",
                )
        except OSError as error:
            stdout_path.write_text(str(error), encoding="utf-8")
            return self._failed(
                size=size,
                mode=mode,
                code_sha256=code_digest,
                dataset_sha256=dataset_digest,
                indices=indices,
                stdout_path=stdout_path,
                started=started,
                error=f"Failed to launch evaluator: {error}",
            )

        stdout_path.write_text(stdout or "", encoding="utf-8")
        if process.returncode != 0:
            return self._failed(
                size=size,
                mode=mode,
                code_sha256=code_digest,
                dataset_sha256=dataset_digest,
                indices=indices,
                stdout_path=stdout_path,
                started=started,
                error=f"Evaluator exited with code {process.returncode}",
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            returned_indices = tuple(payload["indices"])
            score = float(payload["mean_gap_pct"])
            elapsed = float(payload["elapsed_seconds"])
            failures = int(payload["failures"])
            if payload["status"] != "ok":
                raise ValueError("Evaluator reported failure")
            if payload["size"] != size or payload["mode"] != mode:
                raise ValueError("Evaluator result request mismatch")
            if payload["count"] != count or returned_indices != indices:
                raise ValueError("Evaluator result count or indices mismatch")
            if payload["code_sha256"] != code_digest:
                raise ValueError("Evaluator result code hash mismatch")
            if payload["dataset_sha256"] != dataset_digest:
                raise ValueError("Evaluator result dataset hash mismatch")
            if not math.isfinite(score) or not math.isfinite(elapsed):
                raise ValueError("Evaluator result contains nonfinite values")
            if failures < 0 or failures > count:
                raise ValueError("Evaluator result has invalid failure count")
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            return self._failed(
                size=size,
                mode=mode,
                code_sha256=code_digest,
                dataset_sha256=dataset_digest,
                indices=indices,
                stdout_path=stdout_path,
                started=started,
                error=f"Invalid evaluator result: {error}",
            )
        return EvalResult(
            size=size,
            mode=mode,
            mean_gap_pct=score,
            status="ok",
            failures=failures,
            indices=returned_indices,
            code_sha256=code_digest,
            dataset_sha256=dataset_digest,
            elapsed_seconds=elapsed,
            stdout_path=str(stdout_path),
            error=None,
        )
