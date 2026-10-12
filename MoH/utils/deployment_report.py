"""Freeze selected heuristics and report independent test evaluations."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

REPORT_FIELDS = (
    "size",
    "status",
    "validation_gap_pct",
    "test_gap_pct",
    "val_n_instances",
    "test_n_instances",
    "test_failures",
    "elapsed_seconds",
    "heuristic_sha256",
    "checkpoint_sha256",
    "val_dataset_sha256",
    "test_dataset_sha256",
    "seed",
    "error",
    "selection_metadata",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_selected_heuristic(selected, output_dir: Path) -> Path:
    directory = Path(output_dir) / "heuristics"
    directory.mkdir(parents=True, exist_ok=True)
    code_path = directory / f"tsp{selected.size}.py"
    code_path.write_text(selected.code, encoding="utf-8")
    digest = _sha256_bytes(code_path.read_bytes())
    metadata = {
        "size": selected.size,
        "idea": selected.idea,
        "code_sha256": digest,
        "validation": asdict(selected.validation),
        "selection_metadata": selected.metadata,
    }
    (directory / f"tsp{selected.size}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return code_path


def _failed_target_row(size, error, evaluator, checkpoint_digest):
    return {
        "size": size,
        "status": "inference_failed",
        "validation_gap_pct": None,
        "test_gap_pct": None,
        "val_n_instances": 0,
        "test_n_instances": 0,
        "test_failures": 0,
        "elapsed_seconds": 0.0,
        "heuristic_sha256": None,
        "checkpoint_sha256": checkpoint_digest,
        "val_dataset_sha256": None,
        "test_dataset_sha256": None,
        "seed": evaluator.seed,
        "error": error,
        "selection_metadata": {},
    }


def evaluate_selected_heuristics(
    selected,
    evaluator,
    output_dir: Path,
    checkpoint,
    target_errors: dict[int, str] | None = None,
    evaluate_test: bool = True,
) -> list[dict]:
    checkpoint_digest = _sha256_bytes(checkpoint.code.encode())
    rows = []
    successful_sizes = set()
    for choice in selected:
        successful_sizes.add(choice.size)
        code_path = save_selected_heuristic(choice, output_dir)
        code = code_path.read_text(encoding="utf-8")
        code_digest = _sha256_bytes(code_path.read_bytes())
        validation = choice.validation
        row = {
            "size": choice.size,
            "status": "validation_only" if not evaluate_test else "ok",
            "validation_gap_pct": validation.mean_gap_pct,
            "test_gap_pct": None,
            "val_n_instances": len(validation.indices),
            "test_n_instances": 0,
            "test_failures": 0,
            "elapsed_seconds": validation.elapsed_seconds,
            "heuristic_sha256": code_digest,
            "checkpoint_sha256": checkpoint_digest,
            "val_dataset_sha256": validation.dataset_sha256,
            "test_dataset_sha256": None,
            "seed": evaluator.seed,
            "error": None,
            "selection_metadata": choice.metadata,
        }
        if evaluate_test:
            test_result = evaluator.evaluate(code, choice.size, "test")
            row["test_n_instances"] = len(test_result.indices)
            row["test_failures"] = test_result.failures
            row["elapsed_seconds"] += test_result.elapsed_seconds
            row["test_dataset_sha256"] = test_result.dataset_sha256
            if test_result.status == "ok":
                row["test_gap_pct"] = test_result.mean_gap_pct
            else:
                row["status"] = "test_failed"
                row["error"] = test_result.error or "Test evaluation failed"
        rows.append(row)

    for size, error in (target_errors or {}).items():
        if size not in successful_sizes:
            rows.append(
                _failed_target_row(size, error, evaluator, checkpoint_digest)
            )
    return sorted(rows, key=lambda row: row["size"])


def write_report(rows: list[dict], output_dir: Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (destination / "report.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["selection_metadata"] = json.dumps(
                row.get("selection_metadata", {}), sort_keys=True
            )
            writer.writerow(csv_row)
