"""Immutable training checkpoints for deployment inference."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from utils.population import Pop

SCHEMA_VERSION = 1
_SECRET_MARKERS = ("api_key", "token", "secret", "password")
_ITERATION_RE = re.compile(r"^iter_(\d+)\.json$")
_VALIDATION_HEADER_RE = re.compile(
    r"\[\*\] mode=val, data=(.+?), seed=(-?\d+), n_instances=(\d+)"
)
_INDICES_RE = re.compile(r"\[\*\] instance_indices=(\[[^\n]*\])")
_DATASET_SIZE_RE = re.compile(r"val_instance_data_sol_(\d+)\.pkl$")


@dataclass(frozen=True)
class Elite:
    source_size: int
    idea: str
    code: str
    code_sha256: str


@dataclass(frozen=True)
class TrainingCheckpoint:
    code: str
    elites: tuple[Elite, ...]
    metadata: dict


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).lower()
    return any(marker in normalized for marker in _SECRET_MARKERS)


def _remove_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _remove_secrets(child)
            for key, child in value.items()
            if not _is_secret_key(key)
        }
    if isinstance(value, list):
        return [_remove_secrets(child) for child in value]
    return value


def _resolve_sanitized(config: DictConfig | ListConfig, raw: Any, path: str = "") -> Any:
    if isinstance(raw, dict):
        return {
            key: _resolve_sanitized(config, value, f"{path}.{key}" if path else key)
            for key, value in raw.items()
        }
    if isinstance(raw, list):
        return [
            _resolve_sanitized(config, value, f"{path}[{index}]")
            for index, value in enumerate(raw)
        ]
    try:
        return OmegaConf.select(config, path, throw_on_missing=True)
    except (KeyError, OmegaConfBaseException):
        return raw


def redacted_config(cfg: Any) -> dict:
    """Return resolved non-secret configuration without reading secret values."""
    if OmegaConf.is_config(cfg):
        unresolved = OmegaConf.to_container(cfg, resolve=False)
    elif isinstance(cfg, dict):
        unresolved = cfg
    else:
        unresolved = vars(cfg)
    sanitized = _remove_secrets(unresolved)
    sanitized_cfg = OmegaConf.create(sanitized)
    resolved = _resolve_sanitized(sanitized_cfg, sanitized)
    if not isinstance(resolved, dict):
        raise TypeError("Checkpoint configuration must be a mapping")
    return resolved


def _safe_relative_path(root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Artifact path must be a nonempty relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Artifact path must stay relative to checkpoint: {raw_path}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Artifact path escapes checkpoint: {raw_path}")
    return resolved


def _validation_provenance(metadata: dict, training_sizes: list[int]) -> dict:
    supplied = metadata.get("validation_provenance")
    if not isinstance(supplied, dict):
        source_run = metadata.get("source_run")
        supplied = (
            _recover_validation_provenance(Path(source_run), training_sizes)
            if source_run
            else {}
        )
    result = {}
    for size in training_sizes:
        entry = supplied.get(str(size), supplied.get(size, {"status": "unknown"}))
        entry = dict(entry) if isinstance(entry, dict) else {"status": "unknown"}
        dataset_path = entry.get("dataset_path")
        if dataset_path:
            path = Path(dataset_path)
            if path.is_file():
                entry["dataset_sha256"] = _sha256_file(path)
        result[str(size)] = entry
    return result


def _recover_validation_provenance(run_dir: Path, training_sizes: list[int]) -> dict:
    recovered = {}
    logs_dir = run_dir / "logs"
    for path in sorted(logs_dir.glob("*_val.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        header = _VALIDATION_HEADER_RE.search(text)
        indices_match = _INDICES_RE.search(text)
        if not header or not indices_match:
            continue
        dataset_path = Path(header.group(1))
        size_match = _DATASET_SIZE_RE.search(dataset_path.name)
        if not size_match:
            continue
        size = int(size_match.group(1))
        if size not in training_sizes:
            continue
        try:
            indices = ast.literal_eval(indices_match.group(1))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(indices, list) or not all(
            isinstance(index, int) for index in indices
        ):
            continue
        entry = {
            "status": "known",
            "dataset_path": str(dataset_path),
            "seed": int(header.group(2)),
            "n_instances": int(header.group(3)),
            "indices": indices,
        }
        if dataset_path.is_file():
            entry["dataset_sha256"] = _sha256_file(dataset_path)
        recovered[str(size)] = entry
    return recovered


def _recover_run_config(run_dir: Path) -> tuple[dict, int]:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        return {}, 0
    try:
        cfg = OmegaConf.load(config_path)
        config = redacted_config(cfg)
        seed = int(config.get("seed", 0))
    except (OSError, TypeError, ValueError, OmegaConfBaseException) as error:
        raise ValueError("Legacy Hydra configuration is malformed") from error
    return config, seed


def export_checkpoint(
    output_dir: Path,
    code: str,
    population: Pop,
    metadata: dict,
    elite_count: int = 3,
) -> Path:
    """Atomically export an optimizer and training elites to a new directory."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {destination}")
    if elite_count <= 0:
        raise ValueError("elite_count must be positive")
    compile(code, "<checkpoint-meta-optimizer>", "exec")

    training_sizes = [int(size) for size in metadata.get("training_sizes", ())]
    if not training_sizes:
        raise ValueError("Checkpoint metadata requires training_sizes")
    if len(set(training_sizes)) != len(training_sizes):
        raise ValueError("training_sizes must be unique")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        optimizer_path = temporary / "meta_optimizer.py"
        optimizer_path.write_text(code, encoding="utf-8")
        optimizer_digest = _sha256_file(optimizer_path)
        elites_dir = temporary / "elites"
        elites_dir.mkdir()
        elite_artifacts = []
        for size in training_sizes:
            task = f"tsp_gls-{size}"
            try:
                candidates = population.get_population(task)
            except KeyError as error:
                raise ValueError(f"Missing training population for {task}") from error
            ordered = sorted(candidates, key=lambda item: float(item["utility"]))
            entries = []
            for candidate in ordered[:elite_count]:
                elite_code = str(candidate["best_sol"])
                compile(elite_code, f"<checkpoint-elite-{size}>", "exec")
                entries.append(
                    {
                        "source_size": size,
                        "idea": str(candidate.get("idea", "")),
                        "code": elite_code,
                        "code_sha256": _sha256_bytes(elite_code.encode()),
                        "training_utility": float(candidate["utility"]),
                    }
                )
            if not entries:
                raise ValueError(f"Training population for {task} has no elites")
            relative_path = Path("elites") / f"tsp{size}.json"
            artifact_path = temporary / relative_path
            artifact_path.write_bytes(_json_bytes({"elites": entries}))
            elite_artifacts.append(
                {
                    "source_size": size,
                    "path": relative_path.as_posix(),
                    "sha256": _sha256_file(artifact_path),
                }
            )

        exported_metadata = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "problem_name": metadata.get("problem_name", "tsp_gls"),
            "training_sizes": training_sizes,
            "seed": int(metadata.get("seed", 0)),
            "source_run": metadata.get("source_run"),
            "config": metadata.get("config", {}),
            "validation_provenance": _validation_provenance(
                metadata, training_sizes
            ),
            "optimizer": {
                "path": "meta_optimizer.py",
                "sha256": optimizer_digest,
            },
            "elite_artifacts": elite_artifacts,
        }
        (temporary / "metadata.json").write_bytes(_json_bytes(exported_metadata))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _require_hash(path: Path, expected: Any) -> None:
    if not isinstance(expected, str) or _sha256_file(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path.name}")


def load_checkpoint(checkpoint_dir: Path) -> TrainingCheckpoint:
    """Validate and load checkpoint source without executing it."""
    root = Path(checkpoint_dir)
    metadata_path = root / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint metadata: {error}") from error
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    if metadata.get("problem_name") != "tsp_gls":
        raise ValueError("Unsupported checkpoint problem")

    optimizer = metadata.get("optimizer")
    if not isinstance(optimizer, dict):
        raise TypeError("Checkpoint optimizer metadata is missing")
    optimizer_path = _safe_relative_path(root, optimizer.get("path"))
    _require_hash(optimizer_path, optimizer.get("sha256"))
    code = optimizer_path.read_text(encoding="utf-8")
    try:
        compile(code, str(optimizer_path), "exec")
    except SyntaxError as error:
        raise ValueError("Checkpoint optimizer has invalid Python syntax") from error

    training_sizes = metadata.get("training_sizes")
    if (
        not isinstance(training_sizes, list)
        or not training_sizes
        or any(not isinstance(size, int) for size in training_sizes)
        or len(set(training_sizes)) != len(training_sizes)
    ):
        raise ValueError("Invalid checkpoint training sizes")
    artifacts = metadata.get("elite_artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("Checkpoint elite artifact metadata is missing")
    by_size = {artifact.get("source_size"): artifact for artifact in artifacts}
    if set(by_size) != set(training_sizes) or len(by_size) != len(artifacts):
        raise ValueError("Checkpoint elite artifact sizes do not match training sizes")

    elites = []
    for size in training_sizes:
        artifact = by_size[size]
        artifact_path = _safe_relative_path(root, artifact.get("path"))
        _require_hash(artifact_path, artifact.get("sha256"))
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            entries = payload["elites"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid elite artifact for size {size}") from error
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Elite artifact for size {size} is empty")
        for entry in entries:
            if entry.get("source_size") != size:
                raise ValueError("Elite source size does not match artifact")
            elite_code = entry.get("code")
            expected_digest = entry.get("code_sha256")
            if not isinstance(elite_code, str) or _sha256_bytes(
                elite_code.encode()
            ) != expected_digest:
                raise ValueError("Elite code hash mismatch")
            try:
                compile(elite_code, str(artifact_path), "exec")
            except SyntaxError as error:
                raise ValueError("Elite has invalid Python syntax") from error
            elites.append(
                Elite(
                    source_size=size,
                    idea=str(entry.get("idea", "")),
                    code=elite_code,
                    code_sha256=expected_digest,
                )
            )
    return TrainingCheckpoint(code=code, elites=tuple(elites), metadata=metadata)


def _latest_subtask_snapshot(run_dir: Path) -> tuple[int, Path]:
    snapshots = []
    for path in (run_dir / "pop" / "subtask").glob("iter_*.json"):
        match = _ITERATION_RE.match(path.name)
        if match:
            snapshots.append((int(match.group(1)), path))
    if not snapshots:
        raise ValueError("Legacy run has no completed subtask snapshot")
    return max(snapshots, key=lambda item: item[0])


def _legacy_optimizer_code(run_dir: Path, completed_iteration: int) -> str:
    meta_path = run_dir / "logs" / "meta_utility.csv"
    try:
        with meta_path.open(newline="") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        raise ValueError("Legacy run is missing meta_utility.csv") from error
    accepted = []
    for row in rows:
        try:
            iteration = int(row["iteration"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed legacy meta utility log") from error
        if iteration <= completed_iteration and str(row.get("accepted", "")).lower() == "true":
            accepted.append(iteration)
    if accepted:
        iteration = accepted[-1]
        path = run_dir / "code" / "improver" / f"iter_{iteration}.py"
        if not path.is_file():
            raise ValueError(f"Missing accepted optimizer artifact for iteration {iteration}")
        return path.read_text(encoding="utf-8")

    seed_path = run_dir / "pop" / "improver" / "pre.json"
    try:
        seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
        entries = seed_payload["meta-optimizer"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Legacy run has no usable seed optimizer") from error
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("Legacy seed optimizer artifact must contain exactly one entry")
    code = entries[0].get("best_sol")
    if not isinstance(code, str):
        raise TypeError("Legacy seed optimizer code is missing")
    return code


def export_run_checkpoint(
    run_dir: Path, output_dir: Path, elite_count: int = 3
) -> Path:
    """Convert canonical artifacts from an existing training run."""
    source_run = Path(run_dir)
    completed_iteration, snapshot_path = _latest_subtask_snapshot(source_run)
    code = _legacy_optimizer_code(source_run, completed_iteration)
    config, seed = _recover_run_config(source_run)
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Malformed legacy subtask snapshot") from error
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("Legacy subtask snapshot is empty")
    task_pattern = re.compile(r"^tsp_gls-(\d+)$")
    sizes = []
    policies = []
    population = Pop(list(snapshot), max(elite_count, 1))
    for task, entries in snapshot.items():
        match = task_pattern.match(task)
        if not match or not isinstance(entries, list):
            raise ValueError(f"Unsupported legacy subtask snapshot entry: {task}")
        sizes.append(int(match.group(1)))
        for entry in entries:
            policies.append(entry.get("population_policy"))
            population.save_solution(
                task,
                str(entry.get("idea", "")),
                entry.get("best_sol"),
                float(entry.get("utility")),
            )
    policy = policies[0] if policies else None
    if any(value != policy for value in policies):
        raise ValueError("Mixed population policies in snapshot")
    if policy is not None:
        from utils.behavior_population import POLICY_VERSION, validate_fraction
        if not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION:
            raise ValueError("Unsupported population policy")
        validate_fraction(policy.get("diversity_fraction"))
    return export_checkpoint(
        Path(output_dir),
        code,
        population,
        {
            **({"population_policy": policy} if policy is not None else {}),
            "training_sizes": sorted(sizes),
            "seed": seed,
            "config": config,
            "source_run": str(source_run.resolve()),
            "validation_provenance": _recover_validation_provenance(
                source_run, sizes
            ),
        },
        elite_count=elite_count,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export a deployment checkpoint")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--elite-count", type=int, default=3)
    args = parser.parse_args()
    export_run_checkpoint(args.run, args.output, args.elite_count)


if __name__ == "__main__":
    _main()
