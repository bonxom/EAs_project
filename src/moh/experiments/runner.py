"""M7B-A Experiment Runner and CLI.

Orchestrates Full MoH experiment protocol execution, enforces real API opt-in guards,
provides dry-run budget calculation previews, and produces paper-ready manifests/artifacts.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from moh.core.models import EvaluationContext, Heuristic
from moh.core.seeds import derive_seed
from moh.execution.heuristic_runner import HeuristicRunner
from moh.execution.protocol import ExecutionLimits
from moh.experiments.artifacts import (
    save_manifest_atomic,
    save_program_artifact,
    save_trajectory_artifact,
)
from moh.experiments.protocol import (
    ExperimentManifest,
    ExperimentProtocolConfig,
    FullMoHAlgorithmConfig,
    LLMRuntimeConfig,
    TaskConfig,
    calculate_budget_maxima,
    calculate_campaign_budget,
    compute_prompt_fingerprints,
    hash_config,
)
from moh.full_moh import FullMoHConfig, run_full_moh
from moh.llm.fake import FakeLLM
from moh.optimizers.capabilities import CapabilityLimits
from moh.optimizers.evolution import initial_optimizer_programs
from moh.optimizers.runner import ProgramLimits
from moh.problems.tsp import TSPTask


def get_git_commit_info(cwd: Path | None = None) -> tuple[str, bool]:
    """Return (commit_sha, is_dirty) using git CLI."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True
        ).strip()
        return commit, len(status) > 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown", True


def load_experiment_config(path: str | Path) -> ExperimentProtocolConfig:
    """Load ExperimentProtocolConfig from a YAML file."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    
    task_dict = data.get("task", {})
    algo_dict = data.get("algorithm", {})
    llm_dict = data.get("llm", {})
    
    task_cfg = TaskConfig(
        family=task_dict.get("family", "tsp"),
        sizes=tuple(task_dict.get("sizes", [10])),
        instances_per_task=task_dict.get("instances_per_task", 3),
        root_seed=task_dict.get("root_seed", 42),
    )
    
    prog_limits_dict = algo_dict.get("program_limits", {})
    prog_limits = ProgramLimits(
        timeout_seconds=prog_limits_dict.get("timeout_seconds", 5.0),
        max_message_bytes=prog_limits_dict.get("max_message_bytes", 1048576),
        max_output_bytes=prog_limits_dict.get("max_output_bytes", 65536),
    )
    
    algo_cfg = FullMoHAlgorithmConfig(
        method=algo_dict.get("method", "full_moh"),
        population_size=algo_dict.get("population_size", 2),
        generations=algo_dict.get("generations", 2),
        max_inner_generate_requests=algo_dict.get("max_inner_generate_requests", 5),
        max_inner_evaluate_requests=algo_dict.get("max_inner_evaluate_requests", 5),
        program_limits=prog_limits,
    )
    
    llm_cfg = LLMRuntimeConfig(
        requested_model=llm_dict.get("requested_model", "ag/gemini-3.6-flash-low"),
        provider=llm_dict.get("provider", "openai"),
        proxy=llm_dict.get("proxy", "9router"),
        proxy_version=llm_dict.get("proxy_version", "0.5.81"),
        api_mode=llm_dict.get("api_mode", "chat_completions"),
        outer_max_output_tokens=llm_dict.get("outer_max_output_tokens", 2048),
        inner_max_output_tokens=llm_dict.get("inner_max_output_tokens", 2048),
        timeout_seconds=llm_dict.get("timeout_seconds", 30.0),
        sdk_retries=llm_dict.get("sdk_retries", 0),
        provider_attempt_limit_per_request=llm_dict.get("provider_attempt_limit_per_request", 1),
    )
    
    return ExperimentProtocolConfig(
        protocol_version=data.get("protocol_version", "1.0.0"),
        run_seed=data.get("run_seed", 42),
        replicate_index=data.get("replicate_index", 0),
        task=task_cfg,
        algorithm=algo_cfg,
        llm=llm_cfg,
        output_dir=Path(data.get("output_dir", "outputs/experiments")),
    )


def generate_dry_run_preview(config: ExperimentProtocolConfig, replicates: int = 3) -> dict[str, Any]:
    """Generate dry-run protocol and budget summary without network execution."""
    git_commit, git_dirty = get_git_commit_info()
    budget = calculate_budget_maxima(config)
    campaign = calculate_campaign_budget(config, replicates)
    fingerprints = compute_prompt_fingerprints()
    initial_seeds = initial_optimizer_programs(config.algorithm.population_size)

    return {
        "status": "dry_run_preview",
        "allow_real_api": False,
        "method": config.algorithm.method,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": hash_config(config),
        "prompt_fingerprints": fingerprints,
        "initial_optimizer_programs": [
            {"id": p.id, "idea": p.idea} for p in initial_seeds
        ],
        "per_run_budget_maxima": {
            "max_optimizer_executions": budget.max_optimizer_executions,
            "max_outer_meta_generations": budget.max_outer_meta_generations,
            "max_inner_generate_requests": budget.max_inner_generate_requests,
            "max_inner_evaluate_requests": budget.max_inner_evaluate_requests,
            "max_task_instance_evaluations": budget.max_task_instance_evaluations,
            "max_provider_attempts": budget.max_provider_attempts,
        },
        "campaign_sanity_budget_maxima": {
            "sanity_replicates": replicates,
            "campaign_max_provider_attempts": campaign.campaign_max_provider_attempts,
            "campaign_max_task_instance_evaluations": campaign.campaign_max_task_instance_evaluations,
        },
        "llm_runtime": {
            "requested_model": config.llm.requested_model,
            "provider": config.llm.provider,
            "proxy": config.llm.proxy,
            "proxy_version": config.llm.proxy_version,
            "api_mode": config.llm.api_mode,
            "outer_max_output_tokens": config.llm.outer_max_output_tokens,
            "inner_max_output_tokens": config.llm.inner_max_output_tokens,
            "sdk_retries": config.llm.sdk_retries,
        },
    }


def run_experiment(
    config: ExperimentProtocolConfig,
    allow_real_api: bool = False,
    meta_llm: Any = None,
    inner_llm: Any = None,
) -> dict[str, Any]:
    """Execute Full MoH experiment runner or return dry-run preview if network disallowed."""
    # Guard real API execution
    if config.llm.provider != "fake" and not allow_real_api:
        return generate_dry_run_preview(config)

    git_commit, git_dirty = get_git_commit_info()
    cfg_hash = hash_config(config)
    fingerprints = compute_prompt_fingerprints()
    run_id = f"run_{config.algorithm.method}_{cfg_hash[:8]}_rep{config.replicate_index:02d}"
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Use FakeLLM if provider is fake and LLM instance not provided
    if meta_llm is None:
        if config.llm.provider == "fake":
            meta_llm = FakeLLM(seed=config.run_seed)
        else:
            raise ValueError("Real API provider requires explicit meta_llm instance and --allow-real-api flag")

    inner_llm = inner_llm or meta_llm

    # Instantiate task and evaluator
    size = config.task.sizes[0]
    task = TSPTask.create(size, config.task.instances_per_task, config.task.root_seed)
    runner = HeuristicRunner(limits=ExecutionLimits())

    def evaluator(source_code: str) -> float:
        if not isinstance(source_code, str):
            raise TypeError("source_code must be a string")
        heuristic_obj = Heuristic("h_candidate", source_code)
        context = EvaluationContext(
            task.id,
            task.instance_seeds,
            tuple(
                derive_seed(config.task.root_seed, "task", size, "instance", i, "evaluation", 0)
                for i in range(len(task.instances))
            ),
        )
        eval_res = runner.evaluate(heuristic_obj, task, context)
        if eval_res.status != "success" or eval_res.utility is None:
            raise ValueError(f"heuristic evaluation failed: {eval_res.error}")
        return float(eval_res.utility)

    full_moh_config = FullMoHConfig(
        population_size=config.algorithm.population_size,
        generations=config.algorithm.generations,
        capability_limits=CapabilityLimits(
            max_generate_requests=config.algorithm.max_inner_generate_requests,
            max_evaluate_requests=config.algorithm.max_inner_evaluate_requests,
        ),
        program_limits=config.algorithm.program_limits,
    )

    trajectory_records: list[dict[str, Any]] = []
    
    def emit_event(event: str, payload: dict[str, Any]):
        trajectory_records.append({"event": event, "payload": payload})

    res = run_full_moh(
        config=full_moh_config,
        meta_llm=meta_llm,
        inner_llm=inner_llm,
        evaluator=evaluator,
        emit=emit_event,
    )

    # Classification of status
    if res.best_utility is not None:
        status_label = "COMPLETED_VALID"
    elif res.work_counts.outer_programs_evaluated > 0:
        status_label = "COMPLETED_NO_VALID_UTILITY"
    else:
        status_label = "EXECUTION_FAILURE"

    # Save program artifacts
    artifact_refs = []
    for evaluated in res.outer_result.all_evaluated:
        art_info = save_program_artifact(run_dir, evaluated.program.id, evaluated.program.source_code)
        artifact_refs.append(art_info)

    traj_info = save_trajectory_artifact(run_dir, trajectory_records)
    artifact_refs.append(traj_info)

    # Manifest creation
    manifest = ExperimentManifest(
        run_id=run_id,
        timestamp=timestamp,
        git_commit=git_commit,
        git_dirty=git_dirty,
        method=config.algorithm.method,
        config_hash=cfg_hash,
        requested_model=config.llm.requested_model,
        route_provider=config.llm.provider,
        proxy=config.llm.proxy,
        proxy_version=config.llm.proxy_version,
        api_mode=config.llm.api_mode,
        resolved_proxy_host=os.environ.get("PROXY_HOST", "wsl_default_gateway"),
        task_config={
            "family": config.task.family,
            "sizes": list(config.task.sizes),
            "instances_per_task": config.task.instances_per_task,
            "root_seed": config.task.root_seed,
        },
        algorithm_config={
            "method": config.algorithm.method,
            "population_size": config.algorithm.population_size,
            "generations": config.algorithm.generations,
            "max_inner_generate_requests": config.algorithm.max_inner_generate_requests,
            "max_inner_evaluate_requests": config.algorithm.max_inner_evaluate_requests,
        },
        llm_config={
            "requested_model": config.llm.requested_model,
            "provider": config.llm.provider,
            "sdk_retries": config.llm.sdk_retries,
            "outer_max_output_tokens": config.llm.outer_max_output_tokens,
            "inner_max_output_tokens": config.llm.inner_max_output_tokens,
        },
        prompt_fingerprints=fingerprints,
        work_counts={
            "outer_programs_evaluated": res.work_counts.outer_programs_evaluated,
            "outer_offspring_generated": res.work_counts.outer_offspring_generated,
            "inner_generate_requests": res.work_counts.inner_generate_requests,
            "inner_evaluate_requests": res.work_counts.inner_evaluate_requests,
        },
        token_counts={
            "outer_input_tokens": getattr(meta_llm, "input_tokens", 0),
            "outer_output_tokens": getattr(meta_llm, "output_tokens", 0),
            "total_tokens": getattr(meta_llm, "total_tokens", 0),
        },
        status=status_label,
        best_utility=res.best_utility,
        best_program_id=res.best_program.id if res.best_program else None,
        artifact_references=tuple(artifact_refs),
    )

    manifest_path = save_manifest_atomic(run_dir, manifest)

    return {
        "status": status_label,
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "best_utility": res.best_utility,
        "best_program_id": res.best_program.id if res.best_program else None,
        "work_counts": manifest.work_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Full MoH M7B-A Experiment Harness")
    parser.add_argument("--config", default="configs/full_moh_experiment.yaml", help="Path to experiment config YAML")
    parser.add_argument("--allow-real-api", action="store_true", help="Opt-in flag for real network execution")
    parser.add_argument("--dry-run", action="store_true", help="Print dry-run protocol & budget preview")
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    if args.dry_run or not args.allow_real_api:
        preview = generate_dry_run_preview(config)
        print(json.dumps(preview, indent=2))
        sys.exit(0)

    result = run_experiment(config, allow_real_api=args.allow_real_api)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
