"""M7B-A / M7B-A2 Experiment Artifact Management.

Handles atomic serialization of experiment manifests, content-addressed preservation
of generated outer optimizer programs and inner candidate heuristics (Policy SOURCE-A),
and raw trajectory records without HiFo interpretation.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from moh.experiments.protocol import ExperimentManifest, validate_manifest_safety


def save_program_artifact(output_dir: Path, program_id: str, source_code: str) -> dict[str, str]:
    """Save outer OptimizerProgram source code as a content-addressed artifact."""
    programs_dir = output_dir / "programs"
    programs_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = programs_dir / f"{program_id}.py"
    file_path.write_text(source_code, encoding="utf-8")
    
    sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    return {
        "program_id": program_id,
        "filename": f"programs/{program_id}.py",
        "sha256": sha256,
        "size_bytes": str(len(source_code.encode("utf-8"))),
    }


def save_candidate_artifact(output_dir: Path, source_code: str, max_bytes: int = 65536) -> dict[str, str]:
    """Save inner candidate heuristic source code as a content-addressed artifact (SOURCE-A policy)."""
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = source_code.encode("utf-8")[:max_bytes]
    sha256 = hashlib.sha256(source_bytes).hexdigest()
    file_path = candidates_dir / f"{sha256}.py"
    if not file_path.exists():
        file_path.write_bytes(source_bytes)

    return {
        "filename": f"candidates/{sha256}.py",
        "sha256": sha256,
        "size_bytes": str(len(source_bytes)),
    }


def save_trajectory_artifact(output_dir: Path, trajectory_records: list[dict[str, Any]]) -> dict[str, str]:
    """Save raw uninterpreted trajectory records (facts only, no HiFo labels)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "trajectory.json"
    
    dumped = json.dumps(trajectory_records, indent=2, sort_keys=True)
    file_path.write_text(dumped, encoding="utf-8")
    
    sha256 = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    return {
        "type": "trajectory",
        "filename": "trajectory.json",
        "sha256": sha256,
        "record_count": str(len(trajectory_records)),
    }


def save_manifest_atomic(output_dir: Path, manifest: ExperimentManifest) -> Path:
    """Atomically save manifest.json after strict safety validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    temp_path = output_dir / "manifest.json.tmp"
    
    manifest_dict = asdict(manifest)
    validate_manifest_safety(manifest_dict)
    
    dumped = json.dumps(manifest_dict, indent=2, sort_keys=True)
    temp_path.write_text(dumped, encoding="utf-8")
    temp_path.replace(manifest_path)
    return manifest_path
