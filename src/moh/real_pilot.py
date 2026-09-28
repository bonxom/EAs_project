"""Full MoH Real Pilot Harness and Budget Contract (M6A)."""

import argparse
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from moh.core.models import EvaluationContext, Heuristic
from moh.core.seeds import derive_seed
from moh.execution.heuristic_runner import HeuristicRunner
from moh.execution.protocol import ExecutionLimits
from moh.full_moh import FullMoHConfig, FullMoHResult, run_full_moh
from moh.llm.base import GenerationError
from moh.llm.budget import (
    ProviderAttemptBudget,
    ProviderAttemptLimits,
    ProviderUsageAccountant,
)
from moh.llm.openai_client import OpenAILLMClient, resolve_base_url
from moh.llm.parsing import strip_code_fence
from moh.optimizers.capabilities import CapabilityLimits
from moh.optimizers.programs import parse_optimizer_program
from moh.optimizers.runner import ProgramLimits
from moh.problems.tsp import TSPTask

EXPECTED_PILOT_MODEL = "ag/gemini-3.6-flash-low"

PILOT_SEED_OPTIMIZER = parse_optimizer_program(
    "def improve_algorithm(api):\n"
    "    c = api.generate('KIND: mutate\\nOption 1.')\n"
    "    return api.evaluate(c)\n",
    "o000001",
    idea="Initial pilot seed optimizer",
)

TSP_CANDIDATE_CONTRACT_PROMPT = """[TASK CONTRACT]
Generate a Python TSP heuristic function implementing exactly:

def select_next_node(current_node, unvisited, coordinates):

Parameters:
- current_node: int or coordinate representing the current node index.
- unvisited: list of unvisited node indices.
- coordinates: dict or list mapping node index to (x, y) coordinates.

Requirements:
- Must define a function named `select_next_node`.
- Must return a single node index from `unvisited`.
- Output MUST be executable Python source code ONLY.
- Do NOT include markdown code blocks or fences (no ```python or ```).
- Do NOT include any explanations, prose, or commentary.
"""

TEXT_KINDS = {"reflection"}
CANDIDATE_KINDS = {"mutate", "crossover"}


def parse_generation_kind(prompt: str) -> str:
    lines = prompt.splitlines()
    first_line = lines[0].strip() if lines else ""
    if not first_line.startswith("KIND: "):
        raise GenerationError("missing_prompt_kind")
    kind = first_line[6:].strip()
    if kind not in TEXT_KINDS and kind not in CANDIDATE_KINDS:
        raise GenerationError("unsupported_generation_kind")
    return kind


class TaskCandidateLLMAdapter:
    """Wraps an LLM client to route generation requests by intent (TEXT vs CANDIDATE)."""

    def __init__(self, llm: Any, task_contract_prompt: str):
        self._llm = llm
        self._task_contract_prompt = task_contract_prompt

    def generate(self, prompt: str) -> str:
        kind = parse_generation_kind(prompt)
        if kind in TEXT_KINDS:
            return self._llm.generate(prompt)
        elif kind in CANDIDATE_KINDS:
            combined_prompt = (
                f"{self._task_contract_prompt}\n\n[OPTIMIZER INSTRUCTION]\n{prompt}"
            )
            return self._llm.generate(combined_prompt)
        else:
            raise GenerationError("unsupported_generation_kind")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


@dataclass(frozen=True)
class InnerSearchSettings:
    max_generate_requests: int = 1
    max_evaluate_requests: int = 1

    def __post_init__(self):
        if (
            type(self.max_generate_requests) is bool
            or not isinstance(self.max_generate_requests, int)
            or self.max_generate_requests != 1
        ):
            raise ValueError("max_generate_requests must be exactly 1")
        if (
            type(self.max_evaluate_requests) is bool
            or not isinstance(self.max_evaluate_requests, int)
            or self.max_evaluate_requests != 1
        ):
            raise ValueError("max_evaluate_requests must be exactly 1")


@dataclass(frozen=True)
class OuterLLMSettings:
    requested_model: str = EXPECTED_PILOT_MODEL
    max_output_tokens: int = 2048
    provider_attempt_limit: int = 1

    def __post_init__(self):
        if not isinstance(self.requested_model, str) or not self.requested_model.strip():
            raise ValueError("requested_model must be a non-empty string")
        if (
            type(self.max_output_tokens) is bool
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 8
            or self.max_output_tokens > 2048
        ):
            raise ValueError("max_output_tokens must be an integer between 9 and 2048")
        if (
            type(self.provider_attempt_limit) is bool
            or not isinstance(self.provider_attempt_limit, int)
            or self.provider_attempt_limit != 1
        ):
            raise ValueError("provider_attempt_limit must be exactly 1")


@dataclass(frozen=True)
class InnerLLMSettings:
    requested_model: str = EXPECTED_PILOT_MODEL
    max_output_tokens: int = 2048
    provider_attempt_limit_per_optimizer: int = 1

    def __post_init__(self):
        if not isinstance(self.requested_model, str) or not self.requested_model.strip():
            raise ValueError("requested_model must be a non-empty string")
        if (
            type(self.max_output_tokens) is bool
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 8
            or self.max_output_tokens > 2048
        ):
            raise ValueError("max_output_tokens must be an integer between 9 and 2048")
        if (
            type(self.provider_attempt_limit_per_optimizer) is bool
            or not isinstance(self.provider_attempt_limit_per_optimizer, int)
            or self.provider_attempt_limit_per_optimizer != 1
        ):
            raise ValueError("provider_attempt_limit_per_optimizer must be exactly 1")


@dataclass(frozen=True)
class RuntimeSettings:
    timeout_seconds: float = 30.0
    sdk_max_retries: int = 0
    route_provider: str = "antigravity"
    proxy: str = "9router"
    proxy_version: str = "0.5.81"
    api_mode: str = "chat_completions"

    def __post_init__(self):
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds != 30.0
        ):
            raise ValueError("timeout_seconds must be exactly 30.0")
        if (
            type(self.sdk_max_retries) is bool
            or not isinstance(self.sdk_max_retries, int)
            or self.sdk_max_retries != 0
        ):
            raise ValueError("sdk_max_retries must be exactly 0")


@dataclass(frozen=True)
class TaskSettings:
    name: str = "tsp10"
    size: int = 10
    count: int = 1
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if type(self.size) is bool or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("size must be a positive integer")
        if type(self.count) is bool or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("count must be a positive integer")
        if type(self.seed) is bool or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class RealPilotConfig:
    population_size: int = 1
    generations: int = 1
    inner: InnerSearchSettings = field(default_factory=InnerSearchSettings)
    outer_llm: OuterLLMSettings = field(default_factory=OuterLLMSettings)
    inner_llm: InnerLLMSettings = field(default_factory=InnerLLMSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    task: TaskSettings = field(default_factory=TaskSettings)

    def __post_init__(self):
        if (
            type(self.population_size) is bool
            or not isinstance(self.population_size, int)
            or self.population_size != 1
        ):
            raise ValueError("population_size must be exactly 1 for pilot")
        if (
            type(self.generations) is bool
            or not isinstance(self.generations, int)
            or self.generations != 1
        ):
            raise ValueError("generations must be exactly 1 for pilot")
        if not isinstance(self.inner, InnerSearchSettings):
            raise TypeError("inner must be an InnerSearchSettings instance")
        if not isinstance(self.outer_llm, OuterLLMSettings):
            raise TypeError("outer_llm must be an OuterLLMSettings instance")
        if not isinstance(self.inner_llm, InnerLLMSettings):
            raise TypeError("inner_llm must be an InnerLLMSettings instance")
        if not isinstance(self.runtime, RuntimeSettings):
            raise TypeError("runtime must be a RuntimeSettings instance")
        if not isinstance(self.task, TaskSettings):
            raise TypeError("task must be a TaskSettings instance")


@dataclass(frozen=True)
class RealPilotResult:
    status: Literal["dry_run_refused", "success", "failed"]
    reason: str | None
    requested_model: str
    proxy: str
    proxy_version: str
    population_size: int
    generations: int
    outer_programs_evaluated: int
    outer_offspring_generated: int
    inner_generate_requests: int
    inner_evaluate_requests: int
    outer_provider_attempts: int
    inner_provider_attempts: int
    total_provider_attempts: int
    outer_input_tokens: int
    outer_output_tokens: int
    outer_reasoning_tokens: int
    outer_total_tokens: int
    inner_input_tokens: int
    inner_output_tokens: int
    inner_reasoning_tokens: int
    inner_total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    total_tokens: int
    best_utility: float | None
    best_program_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pilot_config(path: str | Path) -> RealPilotConfig:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Pilot config file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    inner_dict = data.get("inner", {})
    outer_dict = data.get("outer_llm", {})
    inner_llm_dict = data.get("inner_llm", {})
    runtime_dict = data.get("runtime", {})
    task_dict = data.get("task", {})

    inner_cfg = InnerSearchSettings(
        max_generate_requests=inner_dict.get("max_generate_requests", 1),
        max_evaluate_requests=inner_dict.get("max_evaluate_requests", 1),
    )
    outer_cfg = OuterLLMSettings(
        requested_model=outer_dict.get("requested_model", EXPECTED_PILOT_MODEL),
        max_output_tokens=outer_dict.get("max_output_tokens", 2048),
        provider_attempt_limit=outer_dict.get("provider_attempt_limit", 1),
    )
    inner_llm_cfg = InnerLLMSettings(
        requested_model=inner_llm_dict.get("requested_model", EXPECTED_PILOT_MODEL),
        max_output_tokens=inner_llm_dict.get("max_output_tokens", 2048),
        provider_attempt_limit_per_optimizer=inner_llm_dict.get(
            "provider_attempt_limit_per_optimizer", 1
        ),
    )
    runtime_cfg = RuntimeSettings(
        timeout_seconds=float(runtime_dict.get("timeout_seconds", 30.0)),
        sdk_max_retries=runtime_dict.get("sdk_max_retries", 0),
        route_provider=runtime_dict.get("route_provider", "antigravity"),
        proxy=runtime_dict.get("proxy", "9router"),
        proxy_version=runtime_dict.get("proxy_version", "0.5.81"),
        api_mode=runtime_dict.get("api_mode", "chat_completions"),
    )
    task_cfg = TaskSettings(
        name=task_dict.get("name", "tsp10"),
        size=task_dict.get("size", 10),
        count=task_dict.get("count", 1),
        seed=task_dict.get("seed", 42),
    )

    return RealPilotConfig(
        population_size=data.get("population_size", 1),
        generations=data.get("generations", 1),
        inner=inner_cfg,
        outer_llm=outer_cfg,
        inner_llm=inner_llm_cfg,
        runtime=runtime_cfg,
        task=task_cfg,
    )


def validate_real_pilot_environment(config: RealPilotConfig) -> None:
    api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_COMPAT_API_KEY must be set in environment for real mode")

    legacy_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if legacy_key:
        raise ValueError("OPENAI_API_KEY must be inactive/unset in real mode")

    base_url = resolve_base_url()
    if not base_url:
        raise ValueError(
            "OPENAI_COMPAT_BASE_URL must be set in environment for real mode"
        )

    if config.outer_llm.requested_model != EXPECTED_PILOT_MODEL:
        raise ValueError(
            f"outer requested_model must be '{EXPECTED_PILOT_MODEL}', got {config.outer_llm.requested_model}"
        )
    if config.inner_llm.requested_model != EXPECTED_PILOT_MODEL:
        raise ValueError(
            f"inner requested_model must be '{EXPECTED_PILOT_MODEL}', got {config.inner_llm.requested_model}"
        )


def make_tsp_evaluator(
    size: int = 10, count: int = 1, seed: int = 42
) -> Callable[[str], float]:
    task = TSPTask.create(size, count, seed)
    context = EvaluationContext(
        task.id,
        task.instance_seeds,
        tuple(
            derive_seed(seed, "task", size, "instance", i, "evaluation", 0)
            for i in range(len(task.instances))
        ),
    )
    runner = HeuristicRunner(ExecutionLimits())

    def evaluate_heuristic(source_code: str) -> float:
        cleaned_source = strip_code_fence(source_code)
        heuristic = Heuristic("cand", cleaned_source)
        res = runner.evaluate(heuristic, task, context)
        if res.status != "success" or res.utility is None:
            raise ValueError(f"Heuristic execution failed: {res.error or 'unknown error'}")
        return res.utility

    return evaluate_heuristic


def run_real_pilot_harness(
    config: RealPilotConfig | None = None,
    *,
    allow_real_api: bool = False,
    transport: Any = None,
    evaluator: Callable[[str], float] | None = None,
    observer: Any = None,
) -> RealPilotResult:
    cfg = config or RealPilotConfig()
    if not isinstance(cfg, RealPilotConfig):
        raise TypeError("config must be a RealPilotConfig instance")

    if not allow_real_api and transport is None:
        return RealPilotResult(
            status="dry_run_refused",
            reason="--allow-real-api flag required to execute network requests",
            requested_model=cfg.outer_llm.requested_model,
            proxy=cfg.runtime.proxy,
            proxy_version=cfg.runtime.proxy_version,
            population_size=cfg.population_size,
            generations=cfg.generations,
            outer_programs_evaluated=0,
            outer_offspring_generated=0,
            inner_generate_requests=0,
            inner_evaluate_requests=0,
            outer_provider_attempts=0,
            inner_provider_attempts=0,
            total_provider_attempts=0,
            outer_input_tokens=0,
            outer_output_tokens=0,
            outer_reasoning_tokens=0,
            outer_total_tokens=0,
            inner_input_tokens=0,
            inner_output_tokens=0,
            inner_reasoning_tokens=0,
            inner_total_tokens=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_reasoning_tokens=0,
            total_tokens=0,
            best_utility=None,
            best_program_id=None,
        )

    if allow_real_api and transport is None:
        validate_real_pilot_environment(cfg)

    obs = observer or (lambda *_: None)

    outer_accountant = ProviderUsageAccountant()
    outer_attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=cfg.outer_llm.provider_attempt_limit)
    )

    outer_llm = OpenAILLMClient(
        model=cfg.outer_llm.requested_model,
        timeout_seconds=cfg.runtime.timeout_seconds,
        observer=obs,
        transport=transport,
        attempt_budget=outer_attempt_budget,
        usage_accountant=outer_accountant,
        max_output_tokens=cfg.outer_llm.max_output_tokens,
        api_mode=cfg.runtime.api_mode,
    )

    inner_accountants: list[ProviderUsageAccountant] = []
    inner_budgets: list[ProviderAttemptBudget] = []

    def inner_llm_factory(_prog_id: str) -> Any:
        acct = ProviderUsageAccountant()
        budg = ProviderAttemptBudget(
            ProviderAttemptLimits(
                max_attempts=cfg.inner_llm.provider_attempt_limit_per_optimizer
            )
        )
        inner_accountants.append(acct)
        inner_budgets.append(budg)
        raw_client = OpenAILLMClient(
            model=cfg.inner_llm.requested_model,
            timeout_seconds=cfg.runtime.timeout_seconds,
            observer=obs,
            transport=transport,
            attempt_budget=budg,
            usage_accountant=acct,
            max_output_tokens=cfg.inner_llm.max_output_tokens,
            api_mode=cfg.runtime.api_mode,
        )
        return TaskCandidateLLMAdapter(raw_client, TSP_CANDIDATE_CONTRACT_PROMPT)

    eval_fn = evaluator or make_tsp_evaluator(
        size=cfg.task.size, count=cfg.task.count, seed=cfg.task.seed
    )

    full_moh_cfg = FullMoHConfig(
        population_size=cfg.population_size,
        generations=cfg.generations,
        capability_limits=CapabilityLimits(
            max_generate_requests=cfg.inner.max_generate_requests,
            max_evaluate_requests=cfg.inner.max_evaluate_requests,
        ),
        program_limits=ProgramLimits(
            timeout_seconds=5.0,
        ),
    )

    try:
        moh_res: FullMoHResult = run_full_moh(
            config=full_moh_cfg,
            seed_programs=[PILOT_SEED_OPTIMIZER],
            meta_llm=outer_llm,
            evaluator=eval_fn,
            inner_llm_factory=inner_llm_factory,
        )
        status = moh_res.status
        reason = None
        best_utility = moh_res.best_utility
        best_prog_id = moh_res.best_program.id if moh_res.best_program else None
        outer_prog_eval = moh_res.work_counts.outer_programs_evaluated
        outer_offspring_gen = moh_res.work_counts.outer_offspring_generated
        inner_gen_req = moh_res.work_counts.inner_generate_requests
        inner_eval_req = moh_res.work_counts.inner_evaluate_requests
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        reason = f"Execution error: {type(exc).__name__}: {exc}"
        best_utility = None
        best_prog_id = None
        outer_prog_eval = 0
        outer_offspring_gen = 0
        inner_gen_req = 0
        inner_eval_req = 0

    outer_attempts = outer_attempt_budget.usage.attempts
    inner_attempts = sum(b.usage.attempts for b in inner_budgets)
    total_attempts = outer_attempts + inner_attempts

    o_usage = outer_accountant.usage
    o_in = o_usage.input_tokens
    o_out = o_usage.output_tokens
    o_reas = o_usage.reasoning_tokens
    o_tot = o_usage.total_tokens

    i_usages = [acct.usage for acct in inner_accountants]
    i_in = sum(u.input_tokens for u in i_usages)
    i_out = sum(u.output_tokens for u in i_usages)
    i_reas = sum(u.reasoning_tokens for u in i_usages)
    i_tot = sum(u.total_tokens for u in i_usages)

    t_in = o_in + i_in
    t_out = o_out + i_out
    t_reas = o_reas + i_reas
    t_tot = o_tot + i_tot

    return RealPilotResult(
        status=status,
        reason=reason,
        requested_model=cfg.outer_llm.requested_model,
        proxy=cfg.runtime.proxy,
        proxy_version=cfg.runtime.proxy_version,
        population_size=cfg.population_size,
        generations=cfg.generations,
        outer_programs_evaluated=outer_prog_eval,
        outer_offspring_generated=outer_offspring_gen,
        inner_generate_requests=inner_gen_req,
        inner_evaluate_requests=inner_eval_req,
        outer_provider_attempts=outer_attempts,
        inner_provider_attempts=inner_attempts,
        total_provider_attempts=total_attempts,
        outer_input_tokens=o_in,
        outer_output_tokens=o_out,
        outer_reasoning_tokens=o_reas,
        outer_total_tokens=o_tot,
        inner_input_tokens=i_in,
        inner_output_tokens=i_out,
        inner_reasoning_tokens=i_reas,
        inner_total_tokens=i_tot,
        total_input_tokens=t_in,
        total_output_tokens=t_out,
        total_reasoning_tokens=t_reas,
        total_tokens=t_tot,
        best_utility=best_utility,
        best_program_id=best_prog_id,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Full MoH Bounded Real Pilot Entry Point (M6A/M6B)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_moh_real_pilot.yaml",
        help="Path to pilot configuration YAML",
    )
    parser.add_argument(
        "--allow-real-api",
        action="store_true",
        help="Opt-in flag required to perform actual network API calls",
    )
    args = parser.parse_args()

    cfg = load_pilot_config(args.config)
    result = run_real_pilot_harness(cfg, allow_real_api=args.allow_real_api)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
