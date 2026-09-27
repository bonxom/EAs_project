import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from moh.core.models import EvaluationContext, Heuristic
from moh.core.seeds import derive_seed
from moh.execution.heuristic_runner import HeuristicRunner
from moh.execution.protocol import ExecutionLimits
from moh.experiments.artifacts import (
    save_campaign_manifest_atomic,
    save_candidate_artifact,
    save_manifest_atomic,
    save_program_artifact,
    save_trajectory_artifact,
)
from moh.experiments.protocol import (
    CampaignManifest,
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
from moh.llm.base import GenerationError
from moh.llm.budget import (
    ProviderAttemptBudget,
    ProviderAttemptBudgetExceeded,
    ProviderAttemptLimits,
    ProviderUsageAccountant,
)
from moh.llm.fake import FakeLLM
from moh.llm.openai_client import OpenAILLMClient
from moh.optimizers.capabilities import CapabilityLimits
from moh.optimizers.evolution import initial_optimizer_programs
from moh.optimizers.runner import ProgramLimits
from moh.problems.tsp import TSPTask
from moh.real_pilot import TSP_CANDIDATE_CONTRACT_PROMPT, TaskCandidateLLMAdapter


class CampaignMasterGuard:
    """Campaign-level ceiling guard ensuring total attempts across all replicates <= max_attempts."""

    def __init__(self, max_attempts: int = 66):
        if type(max_attempts) is bool or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        self.max_attempts = max_attempts
        self._attempts = 0
        self._lock = threading.Lock()

    @property
    def attempts(self) -> int:
        with self._lock:
            return self._attempts

    def reserve(self) -> None:
        with self._lock:
            if self._attempts >= self.max_attempts:
                raise ProviderAttemptBudgetExceeded("campaign_attempt_budget_exhausted")
            self._attempts += 1


class ReplicateAttemptBudget(ProviderAttemptBudget):
    """Replicate-level attempt budget enforcing <=22 replicate attempts AND <=66 campaign attempts."""

    def __init__(self, campaign_guard: CampaignMasterGuard, max_replicate_attempts: int = 22):
        if not isinstance(campaign_guard, CampaignMasterGuard):
            raise TypeError("campaign_guard must be a CampaignMasterGuard instance")
        limits = ProviderAttemptLimits(max_attempts=max_replicate_attempts)
        super().__init__(limits)
        self._campaign_guard = campaign_guard
        self._max_replicate_attempts = max_replicate_attempts

    def reserve(self) -> None:
        with self._lock:
            if self._attempts >= self._max_replicate_attempts:
                raise ProviderAttemptBudgetExceeded("replicate_attempt_budget_exhausted")
            # Reserve on campaign master guard before incrementing replicate attempt count
            self._campaign_guard.reserve()
            self._attempts += 1


def validate_real_provider_environment(llm_config: LLMRuntimeConfig) -> None:
    """Validate environment variables for real API execution without making network calls."""
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Policy violation: OPENAI_API_KEY must be UNSET during real provider execution")

    api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_COMPAT_API_KEY must be configured for real provider execution")

    compat_model = os.environ.get("OPENAI_COMPAT_MODEL", "").strip()
    if compat_model and compat_model != llm_config.requested_model:
        raise ValueError(
            f"OPENAI_COMPAT_MODEL mismatch: env has '{compat_model}', config requests '{llm_config.requested_model}'"
        )

    base_url = os.environ.get("OPENAI_COMPAT_BASE_URL", "").strip()
    if not base_url:
        raise ValueError("OPENAI_COMPAT_BASE_URL must be configured for real provider execution")
    if ":20128/v1" not in base_url and not base_url.endswith("/v1"):
        raise ValueError(f"OPENAI_COMPAT_BASE_URL ({base_url}) invalid endpoint path or port (expected port 20128 and /v1)")


def build_real_provider_llms(
    config: ExperimentProtocolConfig,
    replicate_budget: ProviderAttemptBudget,
    usage_accountant: ProviderUsageAccountant | None = None,
    transport_factory: Callable[[], Any] | None = None,
) -> tuple[Any, Any]:
    """Construct production outer meta-LLM and inner task-candidate LLM adapter."""
    validate_real_provider_environment(config.llm)

    if usage_accountant is None:
        usage_accountant = ProviderUsageAccountant()

    meta_transport = transport_factory() if transport_factory is not None else None
    inner_transport = transport_factory() if transport_factory is not None else None

    meta_llm = OpenAILLMClient(
        model=config.llm.requested_model,
        timeout_seconds=config.llm.timeout_seconds,
        observer=lambda *_: None,
        transport=meta_transport,
        attempt_budget=replicate_budget,
        usage_accountant=usage_accountant,
        max_output_tokens=config.llm.outer_max_output_tokens,
        api_mode=config.llm.api_mode,
        max_attempts_per_request=config.llm.provider_attempt_limit_per_request,
    )

    inner_client = OpenAILLMClient(
        model=config.llm.requested_model,
        timeout_seconds=config.llm.timeout_seconds,
        observer=lambda *_: None,
        transport=inner_transport,
        attempt_budget=replicate_budget,
        usage_accountant=usage_accountant,
        max_output_tokens=config.llm.inner_max_output_tokens,
        api_mode=config.llm.api_mode,
        max_attempts_per_request=config.llm.provider_attempt_limit_per_request,
    )

    inner_llm = TaskCandidateLLMAdapter(inner_client, TSP_CANDIDATE_CONTRACT_PROMPT)

    return meta_llm, inner_llm


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
    campaign_guard: CampaignMasterGuard | None = None,
    replicate_budget: ProviderAttemptBudget | None = None,
    transport_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Execute Full MoH single replicate experiment runner or return dry-run preview if network disallowed."""
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

    # Construct LLMs if not provided
    if meta_llm is None:
        if config.llm.provider == "fake":
            meta_llm = FakeLLM(seed=config.run_seed + config.replicate_index)
            inner_llm = inner_llm or meta_llm
        else:
            if replicate_budget is None:
                c_guard = campaign_guard or CampaignMasterGuard(max_attempts=66)
                replicate_budget = ReplicateAttemptBudget(c_guard, max_replicate_attempts=22)
            meta_llm, inner_llm = build_real_provider_llms(
                config=config,
                replicate_budget=replicate_budget,
                transport_factory=transport_factory,
            )

    inner_llm = inner_llm or meta_llm

    # Instantiate task and evaluator
    size = config.task.sizes[0]
    task = TSPTask.create(size, config.task.instances_per_task, config.task.root_seed)
    runner = HeuristicRunner(limits=ExecutionLimits())
    candidate_artifact_refs: list[dict[str, str]] = []

    def evaluator(source_code: str) -> float:
        if not isinstance(source_code, str):
            raise TypeError("source_code must be a string")
        
        # Policy SOURCE-A: Save candidate source artifact
        cand_ref = save_candidate_artifact(run_dir, source_code)
        if cand_ref not in candidate_artifact_refs:
            candidate_artifact_refs.append(cand_ref)

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

    failure_stage: str | None = None
    error_code: str | None = None

    try:
        res = run_full_moh(
            config=full_moh_config,
            meta_llm=meta_llm,
            inner_llm=inner_llm,
            evaluator=evaluator,
            emit=emit_event,
        )
        if res.best_utility is not None:
            status_label = "COMPLETED_VALID"
        elif res.work_counts.outer_programs_evaluated > 0:
            status_label = "COMPLETED_NO_VALID_UTILITY"
        else:
            status_label = "EXECUTION_FAILURE"
        best_utility = res.best_utility
        best_program_id = res.best_program.id if res.best_program else None
        work_counts = {
            "outer_programs_evaluated": res.work_counts.outer_programs_evaluated,
            "outer_offspring_generated": res.work_counts.outer_offspring_generated,
            "inner_generate_requests": res.work_counts.inner_generate_requests,
            "inner_evaluate_requests": res.work_counts.inner_evaluate_requests,
        }
        all_evaluated_programs = res.outer_result.all_evaluated
    except ProviderAttemptBudgetExceeded as exc:
        status_label = "SAFETY_FAILURE" if "campaign" in str(exc) or "replicate" in str(exc) else "PROVIDER_FAILURE"
        failure_stage = "llm_generation"
        error_code = str(exc.code) if hasattr(exc, "code") else "provider_attempt_budget_exhausted"
        best_utility = None
        best_program_id = None
        work_counts = {
            "outer_programs_evaluated": 0,
            "outer_offspring_generated": 0,
            "inner_generate_requests": 0,
            "inner_evaluate_requests": 0,
        }
        all_evaluated_programs = []
    except GenerationError as exc:
        status_label = "PROVIDER_FAILURE"
        failure_stage = "llm_generation"
        error_code = getattr(exc, "code", type(exc).__name__)
        best_utility = None
        best_program_id = None
        work_counts = {
            "outer_programs_evaluated": 0,
            "outer_offspring_generated": 0,
            "inner_generate_requests": 0,
            "inner_evaluate_requests": 0,
        }
        all_evaluated_programs = []

    # Save program artifacts
    artifact_refs = []
    for evaluated in all_evaluated_programs:
        art_info = save_program_artifact(run_dir, evaluated.program.id, evaluated.program.source_code)
        artifact_refs.append(art_info)

    artifact_refs.extend(candidate_artifact_refs)
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
        work_counts=work_counts,
        token_counts={
            "outer_input_tokens": getattr(meta_llm, "input_tokens", 0),
            "outer_output_tokens": getattr(meta_llm, "output_tokens", 0),
            "total_tokens": getattr(meta_llm, "total_tokens", 0),
        },
        status=status_label,
        failure_stage=failure_stage,
        error_code=error_code,
        best_utility=best_utility,
        best_program_id=best_program_id,
        artifact_references=tuple(artifact_refs),
    )

    manifest_path = save_manifest_atomic(run_dir, manifest)

    return {
        "status": status_label,
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "best_utility": best_utility,
        "best_program_id": best_program_id,
        "work_counts": manifest.work_counts,
        "failure_stage": failure_stage,
        "error_code": error_code,
    }


def calculate_quality_summary(replicate_results: list[dict[str, Any]], planned_n: int = 3) -> dict[str, Any]:
    """Compute summary statistics for COMPLETED_VALID runs."""
    valid_utilities = [
        r["best_utility"]
        for r in replicate_results
        if r.get("status") == "COMPLETED_VALID" and r.get("best_utility") is not None
    ]
    n_valid = len(valid_utilities)
    if n_valid == 0:
        return {
            "planned_n": planned_n,
            "valid_n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    mean_val = float(statistics.mean(valid_utilities))
    median_val = float(statistics.median(valid_utilities))
    std_val = float(statistics.stdev(valid_utilities)) if n_valid >= 2 else None
    min_val = float(min(valid_utilities))
    max_val = float(max(valid_utilities))

    return {
        "planned_n": planned_n,
        "valid_n": n_valid,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }


def run_campaign(
    config: ExperimentProtocolConfig,
    allow_real_api: bool = False,
    transport_factory: Callable[[], Any] | None = None,
    replicates: int = 3,
) -> dict[str, Any]:
    """Execute a governed multi-replicate campaign sequentially."""
    if config.llm.provider != "fake" and not allow_real_api:
        return generate_dry_run_preview(config, replicates=replicates)

    # Validate environment if real provider
    if config.llm.provider != "fake":
        validate_real_provider_environment(config.llm)

    git_commit, git_dirty = get_git_commit_info()
    cfg_hash = hash_config(config)
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    campaign_id = f"campaign_{config.algorithm.method}_{cfg_hash[:8]}"

    campaign_dir = config.output_dir / "campaigns" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    campaign_budget = calculate_campaign_budget(config, replicates=replicates)
    campaign_guard = CampaignMasterGuard(max_attempts=campaign_budget.campaign_max_provider_attempts)

    planned_replicates = tuple(range(replicates))
    attempted_replicates: list[int] = []
    completed_replicates: list[int] = []
    replicate_results: list[dict[str, Any]] = []

    status_counts: dict[str, int] = {
        "COMPLETED_VALID": 0,
        "COMPLETED_NO_VALID_UTILITY": 0,
        "PROVIDER_FAILURE": 0,
        "EXECUTION_FAILURE": 0,
        "SAFETY_FAILURE": 0,
    }

    total_work_counts = {
        "outer_programs_evaluated": 0,
        "outer_offspring_generated": 0,
        "inner_generate_requests": 0,
        "inner_evaluate_requests": 0,
        "task_instance_evaluations": 0,
    }

    total_provider_attempts = {
        "outer_provider_attempts": 0,
        "inner_provider_attempts": 0,
        "total_provider_attempts": 0,
    }

    total_token_usage = {
        "outer_input_tokens": 0,
        "outer_output_tokens": 0,
        "outer_reasoning_tokens": 0,
        "outer_total_tokens": 0,
        "inner_input_tokens": 0,
        "inner_output_tokens": 0,
        "inner_reasoning_tokens": 0,
        "inner_total_tokens": 0,
        "combined_input_tokens": 0,
        "combined_output_tokens": 0,
        "combined_reasoning_tokens": 0,
        "combined_total_tokens": 0,
    }

    campaign_status = "SANITY_CAMPAIGN_COMPLETE"
    stop_reason: str | None = None

    for rep_idx in planned_replicates:
        # Check campaign guard before starting replicate
        if campaign_guard.attempts >= campaign_guard.max_attempts:
            campaign_status = "SANITY_CAMPAIGN_SAFETY_STOP"
            stop_reason = "campaign_master_attempt_budget_exhausted"
            break

        attempted_replicates.append(rep_idx)

        rep_config = ExperimentProtocolConfig(
            protocol_version=config.protocol_version,
            run_seed=config.run_seed,
            replicate_index=rep_idx,
            task=config.task,
            algorithm=config.algorithm,
            llm=config.llm,
            output_dir=config.output_dir,
        )

        replicate_budget = ReplicateAttemptBudget(
            campaign_guard=campaign_guard,
            max_replicate_attempts=22,
        )

        try:
            rep_res = run_experiment(
                config=rep_config,
                allow_real_api=allow_real_api,
                campaign_guard=campaign_guard,
                replicate_budget=replicate_budget,
                transport_factory=transport_factory,
            )
            completed_replicates.append(rep_idx)
            replicate_results.append(rep_res)

            st = rep_res.get("status", "EXECUTION_FAILURE")
            status_counts[st] = status_counts.get(st, 0) + 1

            if st == "SAFETY_FAILURE":
                campaign_status = "SANITY_CAMPAIGN_SAFETY_STOP"
                stop_reason = rep_res.get("error_code", "replicate_safety_stop")
                break

            # Aggregate metrics from replicate manifest
            manifest_path = Path(rep_res["manifest_path"])
            if manifest_path.exists():
                m_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
                w_counts = m_dict.get("work_counts", {})
                for k in ("outer_programs_evaluated", "outer_offspring_generated", "inner_generate_requests", "inner_evaluate_requests"):
                    total_work_counts[k] += w_counts.get(k, 0)

                total_work_counts["task_instance_evaluations"] += w_counts.get("inner_evaluate_requests", 0) * config.task.instances_per_task

                t_counts = m_dict.get("token_counts", {})
                total_token_usage["outer_input_tokens"] += t_counts.get("outer_input_tokens", 0)
                total_token_usage["outer_output_tokens"] += t_counts.get("outer_output_tokens", 0)
                total_token_usage["outer_total_tokens"] += t_counts.get("total_tokens", 0)

                rep_attempts = replicate_budget.usage.attempts
                total_provider_attempts["total_provider_attempts"] += rep_attempts

        except Exception as exc:  # noqa: BLE001
            campaign_status = "SANITY_CAMPAIGN_INFRASTRUCTURE_FAILURE"
            stop_reason = f"infrastructure_exception: {exc}"
            break

    total_token_usage["combined_input_tokens"] = total_token_usage["outer_input_tokens"] + total_token_usage["inner_input_tokens"]
    total_token_usage["combined_output_tokens"] = total_token_usage["outer_output_tokens"] + total_token_usage["inner_output_tokens"]
    total_token_usage["combined_reasoning_tokens"] = total_token_usage["outer_reasoning_tokens"] + total_token_usage["inner_reasoning_tokens"]
    total_token_usage["combined_total_tokens"] = total_token_usage["outer_total_tokens"] + total_token_usage["inner_total_tokens"]

    quality_summary = calculate_quality_summary(replicate_results, planned_n=replicates)
    replicate_run_ids = tuple(r.get("run_id", "") for r in replicate_results)
    replicate_statuses = tuple(r.get("status", "") for r in replicate_results)

    manifest = CampaignManifest(
        campaign_id=campaign_id,
        timestamp=timestamp,
        git_commit=git_commit,
        git_dirty=git_dirty,
        campaign_kind="sanity",
        method=config.algorithm.method,
        config_hash=cfg_hash,
        requested_model=config.llm.requested_model,
        planned_replicates=planned_replicates,
        attempted_replicates=tuple(attempted_replicates),
        completed_replicates=tuple(completed_replicates),
        replicate_run_ids=replicate_run_ids,
        replicate_statuses=replicate_statuses,
        status_counts=status_counts,
        total_work_counts=total_work_counts,
        total_provider_attempts=total_provider_attempts,
        total_token_usage=total_token_usage,
        quality_summary=quality_summary,
        campaign_status=campaign_status,
        stop_reason=stop_reason,
    )

    manifest_path = save_campaign_manifest_atomic(campaign_dir, manifest)

    return {
        "status": campaign_status,
        "campaign_id": campaign_id,
        "manifest_path": str(manifest_path),
        "planned_replicates": list(planned_replicates),
        "attempted_replicates": attempted_replicates,
        "completed_replicates": completed_replicates,
        "status_counts": status_counts,
        "quality_summary": quality_summary,
        "total_work_counts": total_work_counts,
        "total_provider_attempts": total_provider_attempts,
        "stop_reason": stop_reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Full MoH M7B-A / M7B-B-FIX Experiment Harness")
    parser.add_argument("--config", default="configs/full_moh_experiment.yaml", help="Path to experiment config YAML")
    parser.add_argument("--allow-real-api", action="store_true", help="Opt-in flag for real network execution")
    parser.add_argument("--dry-run", action="store_true", help="Print dry-run protocol & budget preview")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)

        if args.dry_run or not args.allow_real_api:
            preview = generate_dry_run_preview(config, replicates=3)
            print(json.dumps(preview, indent=2))
            sys.exit(0)

        result = run_campaign(config, allow_real_api=args.allow_real_api)
        print(json.dumps(result, indent=2))

        if result.get("status") in ("SANITY_CAMPAIGN_COMPLETE", "COMPLETED_VALID"):
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

