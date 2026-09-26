"""Bounded one-request LLM API canary harness (offline & armed real mode)."""

import argparse
import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from decimal import Decimal

import yaml

from moh.llm.base import GenerationError, LLMClient
from moh.llm.budget import (
    ProviderAttemptBudget,
    ProviderAttemptLimits,
    ProviderUsageAccountant,
)
from moh.llm.openai_client import resolve_api_key

CANARY_FIXED_PROMPT = "Return exactly the word OK."


@dataclass(frozen=True)
class CanaryConfig:
    mode: str = "offline"
    logical_generate_limit: int = 1
    provider_attempt_limit: int = 1
    evaluate_limit: int = 0
    prompt: str = CANARY_FIXED_PROMPT
    max_output_tokens: int | None = 8
    model: str = "test-model"
    timeout_seconds: float = 5.0

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in ("offline", "real"):
            raise ValueError("mode must be either 'offline' or 'real'")

        if type(self.logical_generate_limit) is bool or not isinstance(
            self.logical_generate_limit, int
        ):
            raise TypeError("logical_generate_limit must be an integer (not bool)")
        if self.logical_generate_limit != 1:
            raise ValueError("logical_generate_limit must be exactly 1")

        if type(self.provider_attempt_limit) is bool or not isinstance(
            self.provider_attempt_limit, int
        ):
            raise TypeError("provider_attempt_limit must be an integer (not bool)")
        if self.provider_attempt_limit != 1:
            raise ValueError("provider_attempt_limit must be exactly 1")

        if type(self.evaluate_limit) is bool or not isinstance(
            self.evaluate_limit, int
        ):
            raise TypeError("evaluate_limit must be an integer (not bool)")
        if self.evaluate_limit != 0:
            raise ValueError("evaluate_limit must be exactly 0")

        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.prompt != CANARY_FIXED_PROMPT:
            raise ValueError(f"prompt must be exactly '{CANARY_FIXED_PROMPT}'")

        if self.mode == "real":
            if (
                self.max_output_tokens is None
                or type(self.max_output_tokens) is bool
                or not isinstance(self.max_output_tokens, int)
                or self.max_output_tokens <= 0
            ):
                raise ValueError(
                    "max_output_tokens must be a positive integer in real mode"
                )
        elif (
            self.max_output_tokens is not None
            and (
                type(self.max_output_tokens) is bool
                or not isinstance(self.max_output_tokens, int)
                or self.max_output_tokens <= 0
            )
        ):
            raise ValueError(
                "max_output_tokens must be a positive integer if provided"
            )

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


class CanaryLogicalGuardExceeded(GenerationError):
    """Raised when logical generation calls exceed canary limit."""

    code: str = "logical_canary_budget_exhausted"


class CanaryLogicalGuard:
    def __init__(self, max_calls: int = 1):
        if type(max_calls) is bool or not isinstance(max_calls, int) or max_calls != 1:
            raise ValueError("CanaryLogicalGuard only supports max_calls=1")
        self._max_calls = max_calls
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def admit(self) -> None:
        with self._lock:
            if self._calls >= self._max_calls:
                raise CanaryLogicalGuardExceeded("logical_canary_budget_exhausted")
            self._calls += 1


@dataclass(frozen=True)
class CanaryResult:
    status: str
    mode: str
    model: str
    text: str | None
    logical_generate_requests: int
    provider_attempts: int
    max_output_tokens_requested: int | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost_usd: Decimal | None
    error: str | None = None
    last_provider_error: str | None = None
    malformed_response_reason: str | None = None
    observed_input_tokens: int | None = None
    observed_output_tokens: int | None = None
    observed_reasoning_tokens: int | None = None
    observed_total_tokens: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["estimated_cost_usd"] is not None:
            d["estimated_cost_usd"] = str(d["estimated_cost_usd"])
        return d


def run_llm_canary(
    config: CanaryConfig,
    llm_client: LLMClient,
    logical_guard: CanaryLogicalGuard,
    attempt_budget: ProviderAttemptBudget,
    usage_accountant: ProviderUsageAccountant,
    allow_real_api: bool = False,
) -> CanaryResult:
    if config.mode == "real" and not allow_real_api:
        u = usage_accountant.usage
        return CanaryResult(
            status="failed",
            mode=config.mode,
            model=config.model,
            text=None,
            logical_generate_requests=logical_guard.calls,
            provider_attempts=attempt_budget.usage.attempts,
            max_output_tokens_requested=config.max_output_tokens,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.reasoning_tokens,
            total_tokens=u.total_tokens,
            estimated_cost_usd=usage_accountant.estimated_cost_usd,
            error="real_api_not_authorized",
        )

    if config.mode == "real":
        api_key = resolve_api_key()
        if not api_key:
            u = usage_accountant.usage
            return CanaryResult(
                status="failed",
                mode=config.mode,
                model=config.model,
                text=None,
                logical_generate_requests=logical_guard.calls,
                provider_attempts=attempt_budget.usage.attempts,
                max_output_tokens_requested=config.max_output_tokens,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                reasoning_tokens=u.reasoning_tokens,
                total_tokens=u.total_tokens,
                estimated_cost_usd=usage_accountant.estimated_cost_usd,
                error="missing_provider_credential",
            )

    if config.evaluate_limit != 0:
        raise ValueError("Canary cannot evaluate heuristics")

    try:
        logical_guard.admit()
    except CanaryLogicalGuardExceeded as exc:
        u = usage_accountant.usage
        return CanaryResult(
            status="failed",
            mode=config.mode,
            model=config.model,
            text=None,
            logical_generate_requests=logical_guard.calls,
            provider_attempts=attempt_budget.usage.attempts,
            max_output_tokens_requested=config.max_output_tokens,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.reasoning_tokens,
            total_tokens=u.total_tokens,
            estimated_cost_usd=usage_accountant.estimated_cost_usd,
            error=exc.code,
        )

    try:
        text = llm_client.generate(config.prompt)
        u = usage_accountant.usage

        if config.mode == "real" and (u.input_tokens <= 0 or u.output_tokens <= 0):
            return CanaryResult(
                status="failed",
                mode=config.mode,
                model=config.model,
                text=None,
                logical_generate_requests=logical_guard.calls,
                provider_attempts=attempt_budget.usage.attempts,
                max_output_tokens_requested=config.max_output_tokens,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                reasoning_tokens=u.reasoning_tokens,
                total_tokens=u.total_tokens,
                estimated_cost_usd=usage_accountant.estimated_cost_usd,
                error="missing_provider_usage",
            )

        return CanaryResult(
            status="success",
            mode=config.mode,
            model=config.model,
            text=text,
            logical_generate_requests=logical_guard.calls,
            provider_attempts=attempt_budget.usage.attempts,
            max_output_tokens_requested=config.max_output_tokens,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.reasoning_tokens,
            total_tokens=u.total_tokens,
            estimated_cost_usd=usage_accountant.estimated_cost_usd,
            error=None,
        )
    except GenerationError as exc:
        code = getattr(exc, "code", str(exc))
        last_prov_err = getattr(exc, "last_provider_error", None)
        malformed_reason = getattr(exc, "malformed_response_reason", None)
        obs_in = getattr(exc, "observed_input_tokens", None)
        obs_out = getattr(exc, "observed_output_tokens", None)
        obs_reas = getattr(exc, "observed_reasoning_tokens", None)
        obs_tot = getattr(exc, "observed_total_tokens", None)
        u = usage_accountant.usage
        return CanaryResult(
            status="failed",
            mode=config.mode,
            model=config.model,
            text=None,
            logical_generate_requests=logical_guard.calls,
            provider_attempts=attempt_budget.usage.attempts,
            max_output_tokens_requested=config.max_output_tokens,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.reasoning_tokens,
            total_tokens=u.total_tokens,
            estimated_cost_usd=usage_accountant.estimated_cost_usd,
            error=code,
            last_provider_error=last_prov_err,
            malformed_response_reason=malformed_reason,
            observed_input_tokens=obs_in,
            observed_output_tokens=obs_out,
            observed_reasoning_tokens=obs_reas,
            observed_total_tokens=obs_tot,
        )


def _load_yaml_config(path: str) -> CanaryConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mode_raw = raw.get("mode", "offline")
    mode = "offline" if mode_raw in ("offline", "fake") else mode_raw

    model = (
        os.environ.get("OPENAI_COMPAT_MODEL", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
        or raw.get("model", "gpt-4o-mini")
    )

    kwargs = {
        "mode": mode,
        "logical_generate_limit": raw["logical_generate_limit"],
        "provider_attempt_limit": raw["provider_attempt_limit"],
        "evaluate_limit": raw["evaluate_limit"],
        "prompt": raw["prompt"],
        "max_output_tokens": raw.get("max_output_tokens"),
        "model": model,
    }
    if "timeout_seconds" in raw:
        kwargs["timeout_seconds"] = raw["timeout_seconds"]

    return CanaryConfig(**kwargs)


def main():
    from unittest.mock import MagicMock

    from moh.llm.openai_client import OpenAILLMClient

    parser = argparse.ArgumentParser(description="Run LLM API Canary Harness")
    parser.add_argument(
        "--config", required=True, help="Path to canary YAML config file"
    )
    parser.add_argument(
        "--allow-real-api",
        action="store_true",
        help="Explicit double opt-in authorizing real API call",
    )
    args = parser.parse_args()

    config = _load_yaml_config(args.config)

    if config.mode == "real" and not args.allow_real_api:
        result = CanaryResult(
            status="failed",
            mode=config.mode,
            model=config.model,
            text=None,
            logical_generate_requests=0,
            provider_attempts=0,
            max_output_tokens_requested=config.max_output_tokens,
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
            estimated_cost_usd=None,
            error="real_api_not_authorized",
        )
        print(json.dumps(result.to_dict(), indent=2))
        return

    if config.mode == "real":
        api_key = resolve_api_key()
        if not api_key:
            result = CanaryResult(
                status="failed",
                mode=config.mode,
                model=config.model,
                text=None,
                logical_generate_requests=0,
                provider_attempts=0,
                max_output_tokens_requested=config.max_output_tokens,
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                total_tokens=0,
                estimated_cost_usd=None,
                error="missing_provider_credential",
            )
            print(json.dumps(result.to_dict(), indent=2))
            return
        transport = None
    else:
        os.environ.setdefault("OPENAI_COMPAT_API_KEY", "fake-offline-key")

        class FakeChoice:
            def __init__(self, content="OK"):
                self.message = MagicMock(content=content)

        class FakeUsage:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 1
                self.input_tokens = 10
                self.output_tokens = 1
                self.total_tokens = 11
                self.completion_tokens_details = MagicMock(reasoning_tokens=0)
                self.reasoning_tokens = 0

        class FakeResponse:
            def __init__(self):
                self.choices = [FakeChoice("OK")]
                self.usage = FakeUsage()
                self.model = config.model

        class FakeTransport:
            def __init__(self):
                self.call_count = 0
                self.chat = MagicMock()
                self.chat.completions.create.side_effect = self._create

            def _create(self, **kwargs):
                self.call_count += 1
                return FakeResponse()

        transport = FakeTransport()

    attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=config.provider_attempt_limit)
    )
    usage_accountant = ProviderUsageAccountant()
    logical_guard = CanaryLogicalGuard(max_calls=config.logical_generate_limit)

    client = OpenAILLMClient(
        model=config.model,
        timeout_seconds=config.timeout_seconds,
        observer=lambda _: None,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=usage_accountant,
        max_output_tokens=config.max_output_tokens,
        api_mode="chat_completions",
    )

    result = run_llm_canary(
        config=config,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=usage_accountant,
        allow_real_api=args.allow_real_api,
    )

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
