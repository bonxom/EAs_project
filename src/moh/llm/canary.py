"""Offline-only bounded one-request LLM API canary harness."""

import argparse
import json
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

CANARY_FIXED_PROMPT = "Return exactly the word OK."


@dataclass(frozen=True)
class CanaryConfig:
    logical_generate_limit: int
    provider_attempt_limit: int
    evaluate_limit: int
    prompt: str
    max_output_tokens: int | None
    model: str

    def __post_init__(self):
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

        if self.max_output_tokens is not None and (
            type(self.max_output_tokens) is bool
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError(
                "max_output_tokens must be a positive integer if provided"
            )

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")


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
    text: str | None
    logical_generate_requests: int
    provider_attempts: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost_usd: Decimal | None
    error: str | None = None

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
) -> CanaryResult:
    if config.evaluate_limit != 0:
        raise ValueError("Canary cannot evaluate heuristics")

    try:
        logical_guard.admit()
    except CanaryLogicalGuardExceeded as exc:
        u = usage_accountant.usage
        return CanaryResult(
            status="failed",
            text=None,
            logical_generate_requests=logical_guard.calls,
            provider_attempts=attempt_budget.usage.attempts,
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
        return CanaryResult(
            status="success",
            text=text,
            logical_generate_requests=logical_guard.calls,
            provider_attempts=attempt_budget.usage.attempts,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.reasoning_tokens,
            total_tokens=u.total_tokens,
            estimated_cost_usd=usage_accountant.estimated_cost_usd,
            error=None,
        )
    except GenerationError as exc:
        code = getattr(exc, "code", str(exc))
        u = usage_accountant.usage
        return CanaryResult(
            status="failed",
            text=None,
            logical_generate_requests=logical_guard.calls,
            provider_attempts=attempt_budget.usage.attempts,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.reasoning_tokens,
            total_tokens=u.total_tokens,
            estimated_cost_usd=usage_accountant.estimated_cost_usd,
            error=code,
        )


def _load_yaml_config(path: str) -> CanaryConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw.get("mode") != "fake":
        raise ValueError("M2E1 canary CLI only supports offline mode ('mode: fake')")

    return CanaryConfig(
        logical_generate_limit=raw["logical_generate_limit"],
        provider_attempt_limit=raw["provider_attempt_limit"],
        evaluate_limit=raw["evaluate_limit"],
        prompt=raw["prompt"],
        max_output_tokens=raw.get("max_output_tokens"),
        model=raw["model"],
    )


def main():
    import os

    parser = argparse.ArgumentParser(
        description="Run LLM API Canary Harness (Offline)"
    )
    parser.add_argument(
        "--config", required=True, help="Path to offline canary YAML config file"
    )
    args = parser.parse_args()

    config = _load_yaml_config(args.config)
    os.environ.setdefault("OPENAI_API_KEY", "fake-offline-key")

    from unittest.mock import MagicMock

    from moh.llm.openai_client import OpenAILLMClient

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

    attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=config.provider_attempt_limit)
    )
    usage_accountant = ProviderUsageAccountant()
    logical_guard = CanaryLogicalGuard(max_calls=config.logical_generate_limit)
    transport = FakeTransport()

    client = OpenAILLMClient(
        model=config.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=usage_accountant,
    )

    result = run_llm_canary(
        config=config,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=usage_accountant,
    )

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
