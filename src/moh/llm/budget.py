"""Provider request attempt budget and limit definitions."""

import threading
from dataclasses import dataclass

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
