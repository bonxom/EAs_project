"""M7A/M7B-A Experiment Protocol, Fair Budget Contract, and Campaign Calculator.

Defines immutable experiment configurations, pure budget calculations,
task-instance evaluation bounds, campaign budget calculations, deterministic
config hashing, prompt fingerprinting, and run manifest schemas.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from moh.optimizers.runner import ProgramLimits


@dataclass(frozen=True)
class LLMRuntimeConfig:
    requested_model: str = "ag/gemini-3.6-flash-low"
    provider: str = "openai"
    proxy: str = "9router"
    proxy_version: str = "0.5.81"
    api_mode: str = "chat_completions"
    outer_max_output_tokens: int = 2048
    inner_max_output_tokens: int = 2048
    timeout_seconds: float = 30.0
    sdk_retries: int = 0
    provider_attempt_limit_per_request: int = 1

    def __post_init__(self):
        if not isinstance(self.requested_model, str) or not self.requested_model.strip():
            raise ValueError("requested_model must be a non-empty string")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(self.proxy, str) or not self.proxy.strip():
            raise ValueError("proxy must be a non-empty string")
        if not isinstance(self.proxy_version, str) or not self.proxy_version.strip():
            raise ValueError("proxy_version must be a non-empty string")
        if not isinstance(self.api_mode, str) or not self.api_mode.strip():
            raise ValueError("api_mode must be a non-empty string")

        if type(self.outer_max_output_tokens) is bool or not isinstance(self.outer_max_output_tokens, int) or self.outer_max_output_tokens <= 0:
            raise ValueError("outer_max_output_tokens must be a positive integer")
        if type(self.inner_max_output_tokens) is bool or not isinstance(self.inner_max_output_tokens, int) or self.inner_max_output_tokens <= 0:
            raise ValueError("inner_max_output_tokens must be a positive integer")

        if type(self.timeout_seconds) is bool or not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be a positive float")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

        if type(self.sdk_retries) is bool or self.sdk_retries != 0:
            raise ValueError("sdk_retries must be strictly 0 for reproducible experiment contract")

        if type(self.provider_attempt_limit_per_request) is bool or not isinstance(self.provider_attempt_limit_per_request, int) or self.provider_attempt_limit_per_request <= 0:
            raise ValueError("provider_attempt_limit_per_request must be a positive integer")


@dataclass(frozen=True)
class TaskConfig:
    family: str = "tsp"
    sizes: tuple[int, ...] = (10,)
    instances_per_task: int = 3
    root_seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "sizes", tuple(self.sizes))
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("family must be a non-empty string")

        if not self.sizes or any(type(s) is bool or not isinstance(s, int) or s not in (10, 20, 50) for s in self.sizes):
            raise ValueError("sizes must contain supported TSP sizes (10, 20, 50)")

        if type(self.instances_per_task) is bool or not isinstance(self.instances_per_task, int) or self.instances_per_task <= 0:
            raise ValueError("instances_per_task must be a positive integer")

        if type(self.root_seed) is bool or not isinstance(self.root_seed, int) or self.root_seed < 0:
            raise ValueError("root_seed must be a non-negative integer")


@dataclass(frozen=True)
class FullMoHAlgorithmConfig:
    method: Literal["full_moh"] = "full_moh"
    population_size: int = 2
    generations: int = 2
    max_inner_generate_requests: int = 5
    max_inner_evaluate_requests: int = 5
    program_limits: ProgramLimits = field(default_factory=ProgramLimits)

    def __post_init__(self):
        if self.method != "full_moh":
            raise ValueError("method must be 'full_moh' for baseline configuration")

        if type(self.population_size) is bool or not isinstance(self.population_size, int) or self.population_size <= 0:
            raise ValueError("population_size must be a positive integer")

        if type(self.generations) is bool or not isinstance(self.generations, int) or self.generations < 0:
            raise ValueError("generations must be a non-negative integer")

        if type(self.max_inner_generate_requests) is bool or not isinstance(self.max_inner_generate_requests, int) or self.max_inner_generate_requests < 0:
            raise ValueError("max_inner_generate_requests must be a non-negative integer")

        if type(self.max_inner_evaluate_requests) is bool or not isinstance(self.max_inner_evaluate_requests, int) or self.max_inner_evaluate_requests < 0:
            raise ValueError("max_inner_evaluate_requests must be a non-negative integer")

        if not isinstance(self.program_limits, ProgramLimits):
            raise TypeError("program_limits must be a ProgramLimits instance")


@dataclass(frozen=True)
class ExperimentProtocolConfig:
    protocol_version: str = "1.0.0"
    run_seed: int = 42
    replicate_index: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    algorithm: FullMoHAlgorithmConfig = field(default_factory=FullMoHAlgorithmConfig)
    llm: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)
    output_dir: Path = field(default_factory=lambda: Path("outputs/experiments"))

    def __post_init__(self):
        if not isinstance(self.protocol_version, str) or not self.protocol_version.strip():
            raise ValueError("protocol_version must be a non-empty string")

        if type(self.run_seed) is bool or not isinstance(self.run_seed, int) or self.run_seed < 0:
            raise ValueError("run_seed must be a non-negative integer")

        if type(self.replicate_index) is bool or not isinstance(self.replicate_index, int) or self.replicate_index < 0:
            raise ValueError("replicate_index must be a non-negative integer")

        if not isinstance(self.task, TaskConfig):
            raise TypeError("task must be a TaskConfig instance")

        if not isinstance(self.algorithm, FullMoHAlgorithmConfig):
            raise TypeError("algorithm must be a FullMoHAlgorithmConfig instance")

        if not isinstance(self.llm, LLMRuntimeConfig):
            raise TypeError("llm must be an LLMRuntimeConfig instance")

        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        elif not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a Path or path string")


@dataclass(frozen=True)
class BudgetMaxima:
    max_optimizer_executions: int
    max_outer_meta_generations: int
    max_inner_generate_requests: int
    max_inner_evaluate_requests: int
    max_task_instance_evaluations: int
    max_provider_attempts: int

    def __post_init__(self):
        for name, val in (
            ("max_optimizer_executions", self.max_optimizer_executions),
            ("max_outer_meta_generations", self.max_outer_meta_generations),
            ("max_inner_generate_requests", self.max_inner_generate_requests),
            ("max_inner_evaluate_requests", self.max_inner_evaluate_requests),
            ("max_task_instance_evaluations", self.max_task_instance_evaluations),
            ("max_provider_attempts", self.max_provider_attempts),
        ):
            if type(val) is bool or not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class CampaignBudgetMaxima:
    replicates: int
    per_run: BudgetMaxima
    campaign_max_optimizer_executions: int
    campaign_max_outer_meta_generations: int
    campaign_max_inner_generate_requests: int
    campaign_max_inner_evaluate_requests: int
    campaign_max_task_instance_evaluations: int
    campaign_max_provider_attempts: int


def calculate_budget_maxima(config: ExperimentProtocolConfig) -> BudgetMaxima:
    """Calculate theoretical work and attempt upper bounds for a given experiment config."""
    pop = config.algorithm.population_size
    gen = config.algorithm.generations
    inner_gen_per_opt = config.algorithm.max_inner_generate_requests
    inner_eval_per_opt = config.algorithm.max_inner_evaluate_requests
    attempt_limit = config.llm.provider_attempt_limit_per_request
    instances_per_task = config.task.instances_per_task

    max_optimizer_executions = pop + gen
    max_outer_meta_generations = gen
    max_inner_generate_requests = max_optimizer_executions * inner_gen_per_opt
    max_inner_evaluate_requests = max_optimizer_executions * inner_eval_per_opt
    max_task_instance_evaluations = max_inner_evaluate_requests * instances_per_task

    max_provider_attempts = (
        max_outer_meta_generations * attempt_limit
        + max_inner_generate_requests * attempt_limit
    )

    return BudgetMaxima(
        max_optimizer_executions=max_optimizer_executions,
        max_outer_meta_generations=max_outer_meta_generations,
        max_inner_generate_requests=max_inner_generate_requests,
        max_inner_evaluate_requests=max_inner_evaluate_requests,
        max_task_instance_evaluations=max_task_instance_evaluations,
        max_provider_attempts=max_provider_attempts,
    )


def calculate_campaign_budget(config: ExperimentProtocolConfig, replicates: int) -> CampaignBudgetMaxima:
    """Calculate campaign-level ceilings across N independent replicates."""
    if type(replicates) is bool or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")

    per_run = calculate_budget_maxima(config)
    return CampaignBudgetMaxima(
        replicates=replicates,
        per_run=per_run,
        campaign_max_optimizer_executions=per_run.max_optimizer_executions * replicates,
        campaign_max_outer_meta_generations=per_run.max_outer_meta_generations * replicates,
        campaign_max_inner_generate_requests=per_run.max_inner_generate_requests * replicates,
        campaign_max_inner_evaluate_requests=per_run.max_inner_evaluate_requests * replicates,
        campaign_max_task_instance_evaluations=per_run.max_task_instance_evaluations * replicates,
        campaign_max_provider_attempts=per_run.max_provider_attempts * replicates,
    )


def compare_protocol_fairness(
    config_a: ExperimentProtocolConfig, config_b: ExperimentProtocolConfig
) -> tuple[bool, list[str]]:
    """Verify that two experiment configurations share identical fairness-locked parameters."""
    mismatches = []

    if config_a.task.family != config_b.task.family:
        mismatches.append(f"task family mismatch: {config_a.task.family} vs {config_b.task.family}")
    if config_a.task.sizes != config_b.task.sizes:
        mismatches.append(f"task sizes mismatch: {config_a.task.sizes} vs {config_b.task.sizes}")
    if config_a.task.instances_per_task != config_b.task.instances_per_task:
        mismatches.append(f"instances_per_task mismatch: {config_a.task.instances_per_task} vs {config_b.task.instances_per_task}")
    if config_a.task.root_seed != config_b.task.root_seed:
        mismatches.append(f"task root_seed mismatch: {config_a.task.root_seed} vs {config_b.task.root_seed}")

    if config_a.algorithm.population_size != config_b.algorithm.population_size:
        mismatches.append(f"population_size mismatch: {config_a.algorithm.population_size} vs {config_b.algorithm.population_size}")
    if config_a.algorithm.generations != config_b.algorithm.generations:
        mismatches.append(f"generations mismatch: {config_a.algorithm.generations} vs {config_b.algorithm.generations}")
    if config_a.algorithm.max_inner_generate_requests != config_b.algorithm.max_inner_generate_requests:
        mismatches.append(f"max_inner_generate_requests mismatch: {config_a.algorithm.max_inner_generate_requests} vs {config_b.algorithm.max_inner_generate_requests}")
    if config_a.algorithm.max_inner_evaluate_requests != config_b.algorithm.max_inner_evaluate_requests:
        mismatches.append(f"max_inner_evaluate_requests mismatch: {config_a.algorithm.max_inner_evaluate_requests} vs {config_b.algorithm.max_inner_evaluate_requests}")

    if config_a.llm.requested_model != config_b.llm.requested_model:
        mismatches.append(f"requested_model mismatch: {config_a.llm.requested_model} vs {config_b.llm.requested_model}")
    if config_a.llm.outer_max_output_tokens != config_b.llm.outer_max_output_tokens:
        mismatches.append(f"outer_max_output_tokens mismatch: {config_a.llm.outer_max_output_tokens} vs {config_b.llm.outer_max_output_tokens}")
    if config_a.llm.inner_max_output_tokens != config_b.llm.inner_max_output_tokens:
        mismatches.append(f"inner_max_output_tokens mismatch: {config_a.llm.inner_max_output_tokens} vs {config_b.llm.inner_max_output_tokens}")
    if config_a.llm.provider_attempt_limit_per_request != config_b.llm.provider_attempt_limit_per_request:
        mismatches.append(f"provider_attempt_limit_per_request mismatch: {config_a.llm.provider_attempt_limit_per_request} vs {config_b.llm.provider_attempt_limit_per_request}")
    if config_a.llm.sdk_retries != config_b.llm.sdk_retries:
        mismatches.append(f"sdk_retries mismatch: {config_a.llm.sdk_retries} vs {config_b.llm.sdk_retries}")

    return len(mismatches) == 0, mismatches


def _to_canonical_dict(obj: Any) -> Any:
    if isinstance(obj, (ExperimentProtocolConfig, LLMRuntimeConfig, TaskConfig, FullMoHAlgorithmConfig, ProgramLimits)):
        return {k: _to_canonical_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, Path):
        return str(obj.as_posix())
    elif isinstance(obj, (tuple, list)):
        return [_to_canonical_dict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _to_canonical_dict(v) for k, v in sorted(obj.items())}
    return obj


def hash_config(config: ExperimentProtocolConfig) -> str:
    """Produce deterministic SHA256 hex digest of ExperimentProtocolConfig."""
    canonical_dict = _to_canonical_dict(config)
    dumped = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def compute_prompt_fingerprints() -> dict[str, str]:
    """Return SHA256 fingerprints of outer and inner prompt templates."""
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    outer_prompt_path = prompts_dir / "optimizer_program_generation.py"
    inner_prompt_path = prompts_dir / "heuristic_generation.py"

    res = {}
    if outer_prompt_path.exists():
        content = outer_prompt_path.read_text(encoding="utf-8")
        res["outer_program_prompt_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if inner_prompt_path.exists():
        content = inner_prompt_path.read_text(encoding="utf-8")
        res["inner_heuristic_prompt_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return res


VALID_MANIFEST_STATUSES = (
    "COMPLETED_VALID",
    "COMPLETED_NO_VALID_UTILITY",
    "PROVIDER_FAILURE",
    "EXECUTION_FAILURE",
    "SAFETY_FAILURE",
)


@dataclass(frozen=True)
class ExperimentManifest:
    run_id: str
    timestamp: str
    git_commit: str
    git_dirty: bool
    method: str
    config_hash: str
    requested_model: str
    route_provider: str
    proxy: str
    proxy_version: str
    api_mode: str
    resolved_proxy_host: str
    task_config: dict[str, Any]
    algorithm_config: dict[str, Any]
    llm_config: dict[str, Any]
    prompt_fingerprints: dict[str, str]
    work_counts: dict[str, int]
    token_counts: dict[str, Any]
    status: Literal[
        "COMPLETED_VALID",
        "COMPLETED_NO_VALID_UTILITY",
        "PROVIDER_FAILURE",
        "EXECUTION_FAILURE",
        "SAFETY_FAILURE",
    ]
    failure_stage: str | None = None
    error_code: str | None = None
    best_utility: float | None = None
    best_program_id: str | None = None
    artifact_references: tuple[dict[str, str], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "artifact_references", tuple(self.artifact_references))
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError("timestamp must be a non-empty string")
        if not isinstance(self.git_commit, str) or not self.git_commit.strip():
            raise ValueError("git_commit must be a non-empty string")
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be a boolean")
        if self.status not in VALID_MANIFEST_STATUSES:
            raise ValueError(f"status must be one of {VALID_MANIFEST_STATUSES}")

        if self.best_utility is not None:
            if type(self.best_utility) is bool or not isinstance(self.best_utility, (int, float)) or not math.isfinite(self.best_utility):
                raise ValueError("best_utility must be a finite float if present")
            object.__setattr__(self, "best_utility", float(self.best_utility))

        validate_manifest_safety(_to_canonical_dict(self))


FORBIDDEN_SECRET_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "secret",
    "password",
    "private_key",
)

INTEGER_COUNT_KEYS = (
    "outer_programs_evaluated",
    "outer_offspring_generated",
    "inner_generate_requests",
    "inner_evaluate_requests",
    "max_optimizer_executions",
    "max_outer_meta_generations",
    "max_inner_generate_requests",
    "max_inner_evaluate_requests",
    "max_task_instance_evaluations",
    "max_provider_attempts",
    "outer_provider_attempts",
    "inner_provider_attempts",
    "total_provider_attempts",
)


def validate_manifest_safety(data: Any) -> None:
    """Validate manifest data for absence of secret keys and strict numerical types."""
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(k, str):
                lower_k = k.lower()
                for secret in FORBIDDEN_SECRET_KEYS:
                    if secret in lower_k:
                        raise ValueError(f"Manifest safety violation: forbidden secret key '{k}' found")
                if lower_k in INTEGER_COUNT_KEYS:
                    if type(v) is bool:
                        raise ValueError(f"Manifest safety violation: boolean value not allowed for integer count '{k}'")
                    if not isinstance(v, int) or v < 0:
                        raise ValueError(f"Manifest safety violation: non-integer or negative count for '{k}'")
            validate_manifest_safety(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            validate_manifest_safety(item)
    elif isinstance(data, float) and not math.isfinite(data):
        raise ValueError("Manifest safety violation: non-finite float value found")
