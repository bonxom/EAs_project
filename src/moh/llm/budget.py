"""Provider request attempt budget, token usage accounting, and pricing models."""

import threading
from dataclasses import dataclass
from decimal import Decimal

from moh.llm.base import GenerationError


@dataclass(frozen=True)
class ProviderAttemptLimits:
    max_attempts: int

    def __post_init__(self):
        if type(self.max_attempts) is bool or not isinstance(self.max_attempts, int):
            raise TypeError("max_attempts must be an integer (not bool)")
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be non-negative")


@dataclass(frozen=True)
class ProviderAttemptUsage:
    attempts: int


class ProviderAttemptBudgetExceeded(GenerationError):
    """Raised when actual provider HTTP request attempts exceed budget limit."""

    code: str = "provider_attempt_budget_exhausted"

    def __init__(
        self,
        code: str = "provider_attempt_budget_exhausted",
        last_provider_error: str | None = None,
    ):
        super().__init__(code, last_provider_error=last_provider_error)
        self.code = code


class ProviderAttemptBudget:
    def __init__(self, limits: ProviderAttemptLimits):
        if not isinstance(limits, ProviderAttemptLimits):
            raise TypeError("limits must be a ProviderAttemptLimits instance")
        self._limits = limits
        self._attempts = 0
        self._lock = threading.Lock()

    @property
    def limits(self) -> ProviderAttemptLimits:
        return self._limits

    @property
    def usage(self) -> ProviderAttemptUsage:
        with self._lock:
            return ProviderAttemptUsage(attempts=self._attempts)

    def reserve(self) -> None:
        with self._lock:
            if self._attempts >= self._limits.max_attempts:
                raise ProviderAttemptBudgetExceeded("provider_attempt_budget_exhausted")
            self._attempts += 1


@dataclass(frozen=True)
class ProviderTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __post_init__(self):
        for name, val in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("reasoning_tokens", self.reasoning_tokens),
        ):
            if type(val) is bool or not isinstance(val, int):
                raise TypeError(f"{name} must be an integer (not bool)")
            if val < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")


@dataclass(frozen=True)
class ProviderUsageLimits:
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_estimated_cost_usd: Decimal | None = None

    def __post_init__(self):
        for name, val in (
            ("max_input_tokens", self.max_input_tokens),
            ("max_output_tokens", self.max_output_tokens),
            ("max_total_tokens", self.max_total_tokens),
        ):
            if val is not None:
                if type(val) is bool or not isinstance(val, int):
                    raise TypeError(f"{name} must be an integer (not bool)")
                if val < 0:
                    raise ValueError(f"{name} must be non-negative")

        if self.max_estimated_cost_usd is not None:
            val = self.max_estimated_cost_usd
            if type(val) is bool:
                raise TypeError("max_estimated_cost_usd cannot be bool")
            if isinstance(val, (int, float, str)):
                object.__setattr__(
                    self, "max_estimated_cost_usd", Decimal(str(val))
                )
            elif not isinstance(val, Decimal):
                raise TypeError("max_estimated_cost_usd must be a Decimal")
            if self.max_estimated_cost_usd < Decimal(0):
                raise ValueError("max_estimated_cost_usd must be non-negative")


class ProviderUsageBudgetExceeded(GenerationError):
    """Raised when already-observed provider usage reaches or exceeds limit."""

    def __init__(self, code: str = "provider_usage_budget_exhausted"):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_million_tokens: Decimal
    output_usd_per_million_tokens: Decimal
    reasoning_usd_per_million_tokens: Decimal | None = None

    def __post_init__(self):
        for name, val in (
            ("input_usd_per_million_tokens", self.input_usd_per_million_tokens),
            ("output_usd_per_million_tokens", self.output_usd_per_million_tokens),
        ):
            if type(val) is bool:
                raise TypeError(f"{name} cannot be bool")
            if isinstance(val, (int, float, str)):
                object.__setattr__(self, name, Decimal(str(val)))
            elif not isinstance(val, Decimal):
                raise TypeError(f"{name} must be Decimal")

        if self.reasoning_usd_per_million_tokens is not None:
            val = self.reasoning_usd_per_million_tokens
            if type(val) is bool:
                raise TypeError("reasoning_usd_per_million_tokens cannot be bool")
            if isinstance(val, (int, float, str)):
                object.__setattr__(
                    self, "reasoning_usd_per_million_tokens", Decimal(str(val))
                )
            elif not isinstance(val, Decimal):
                raise TypeError("reasoning_usd_per_million_tokens must be Decimal")

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> Decimal:
        million = Decimal(1000000)
        if self.reasoning_usd_per_million_tokens is not None and reasoning_tokens > 0:
            visible_output_tokens = output_tokens - reasoning_tokens
            return (
                Decimal(input_tokens) * self.input_usd_per_million_tokens / million
                + Decimal(visible_output_tokens)
                * self.output_usd_per_million_tokens
                / million
                + Decimal(reasoning_tokens)
                * self.reasoning_usd_per_million_tokens
                / million
            )
        return (
            Decimal(input_tokens) * self.input_usd_per_million_tokens / million
            + Decimal(output_tokens) * self.output_usd_per_million_tokens / million
        )


class ProviderUsageAccountant:
    def __init__(self, pricing_policy: dict[str, ModelPricing] | None = None):
        self._pricing_policy = dict(pricing_policy or {})
        self._input_tokens = 0
        self._output_tokens = 0
        self._reasoning_tokens = 0
        self._estimated_cost_usd = Decimal(0)
        self._cost_available = True
        self._lock = threading.Lock()

    @property
    def usage(self) -> ProviderTokenUsage:
        with self._lock:
            return ProviderTokenUsage(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                reasoning_tokens=self._reasoning_tokens,
            )

    @property
    def estimated_cost_usd(self) -> Decimal | None:
        with self._lock:
            if not self._cost_available:
                return None
            return self._estimated_cost_usd

    def check_pre_request_guard(
        self, limits: ProviderUsageLimits | None, model: str
    ) -> None:
        if limits is None:
            return

        with self._lock:
            if (
                limits.max_input_tokens is not None
                and self._input_tokens >= limits.max_input_tokens
            ):
                raise ProviderUsageBudgetExceeded(
                    "provider_input_token_budget_exhausted"
                )

            if (
                limits.max_output_tokens is not None
                and self._output_tokens >= limits.max_output_tokens
            ):
                raise ProviderUsageBudgetExceeded(
                    "provider_output_token_budget_exhausted"
                )

            total = self._input_tokens + self._output_tokens
            if (
                limits.max_total_tokens is not None
                and total >= limits.max_total_tokens
            ):
                raise ProviderUsageBudgetExceeded(
                    "provider_total_token_budget_exhausted"
                )

            if limits.max_estimated_cost_usd is not None:
                if not self._cost_available or model not in self._pricing_policy:
                    raise ProviderUsageBudgetExceeded(
                        "provider_cost_budget_model_pricing_unavailable"
                    )
                if (
                    self._estimated_cost_usd is not None
                    and self._estimated_cost_usd >= limits.max_estimated_cost_usd
                ):
                    raise ProviderUsageBudgetExceeded("provider_cost_budget_exhausted")

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> None:
        for name, val in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("reasoning_tokens", reasoning_tokens),
        ):
            if type(val) is bool or not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer (not bool)")

        if reasoning_tokens > output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")

        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._reasoning_tokens += reasoning_tokens

            pricing = self._pricing_policy.get(model)
            if pricing is not None and self._cost_available:
                cost = pricing.calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
                self._estimated_cost_usd += cost
            else:
                self._cost_available = False
