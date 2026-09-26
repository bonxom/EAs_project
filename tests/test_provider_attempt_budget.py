import threading
from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from moh.llm.base import GenerationError
from moh.llm.budget import (
    ProviderAttemptBudget,
    ProviderAttemptBudgetExceeded,
    ProviderAttemptLimits,
    ProviderAttemptUsage,
)
from moh.llm.openai_client import OpenAILLMClient


class MockTransport:
    def __init__(self, values):
        self.values = iter(values)
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


def response(text="hello"):
    return SimpleNamespace(
        output_text=text,
        model="test-model",
        status="completed",
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )


def transient():
    return openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid")
    )


def non_transient():
    return openai.AuthenticationError(
        "offline-test-key",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://example.invalid")
        ),
        body=None,
    )


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")


def test_valid_provider_attempt_limits():
    limits = ProviderAttemptLimits(max_attempts=5)
    assert limits.max_attempts == 5


@pytest.mark.parametrize("val", [-1, -10])
def test_negative_limits_rejected(val):
    with pytest.raises(ValueError):
        ProviderAttemptLimits(max_attempts=val)


@pytest.mark.parametrize("val", [True, False, 1.5, "2"])
def test_invalid_type_limits_rejected(val):
    with pytest.raises(TypeError):
        ProviderAttemptLimits(max_attempts=val)


def test_zero_budget_prevents_network_request():
    transport = MockTransport([response()])
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=0))

    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
    )

    with pytest.raises(
        ProviderAttemptBudgetExceeded, match="provider_attempt_budget_exhausted"
    ):
        client.generate("hello")

    assert len(transport.requests) == 0
    assert budget.usage == ProviderAttemptUsage(attempts=0)


def test_first_try_success():
    transport = MockTransport([response("world")])
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=3))

    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
    )

    text = client.generate("hello")

    assert text == "world"
    assert len(transport.requests) == 1
    assert budget.usage == ProviderAttemptUsage(attempts=1)


def test_retry_then_success():
    transport = MockTransport([transient(), response("success after retry")])
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=3))

    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
        sleep=lambda _: None,
    )

    text = client.generate("hello")

    assert text == "success after retry"
    assert len(transport.requests) == 2
    assert budget.usage == ProviderAttemptUsage(attempts=2)


def test_hard_limit_during_retries():
    # 3 transient errors queued, but budget max_attempts = 2
    transport = MockTransport([transient(), transient(), transient()])
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=2))

    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
        sleep=lambda _: None,
    )

    with pytest.raises(
        ProviderAttemptBudgetExceeded, match="provider_attempt_budget_exhausted"
    ):
        client.generate("hello")

    assert len(transport.requests) == 2  # Exactly 2 calls, 3rd call NEVER occurred
    assert budget.usage == ProviderAttemptUsage(attempts=2)


def test_non_retryable_failure():
    transport = MockTransport([non_transient()])
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=3))

    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
    )

    with pytest.raises(GenerationError):
        client.generate("hello")

    assert len(transport.requests) == 1
    assert budget.usage == ProviderAttemptUsage(attempts=1)


def test_shared_budget_across_multiple_generate_calls():
    # Call 1 takes 2 attempts (retry + success). Call 2 succeeds on 1st attempt. Call 3 exceeds budget.
    outcomes = [
        transient(),
        response("res-1"),
        response("res-2"),
        response("res-3"),
    ]
    transport = MockTransport(outcomes)
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=3))

    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
        sleep=lambda _: None,
    )

    r1 = client.generate("p1")
    assert r1 == "res-1"
    assert len(transport.requests) == 2
    assert budget.usage == ProviderAttemptUsage(attempts=2)

    r2 = client.generate("p2")
    assert r2 == "res-2"
    assert len(transport.requests) == 3
    assert budget.usage == ProviderAttemptUsage(attempts=3)

    # Third generate call MUST fail before network I/O
    with pytest.raises(ProviderAttemptBudgetExceeded):
        client.generate("p3")

    assert len(transport.requests) == 3  # Request count stayed at 3!
    assert budget.usage == ProviderAttemptUsage(attempts=3)


def test_thread_safe_budget_reservation():
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=100))
    exceptions = []

    def worker():
        for _ in range(10):
            try:
                budget.reserve()
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert budget.usage.attempts == 100
    assert len(exceptions) == 0

    with pytest.raises(ProviderAttemptBudgetExceeded):
        budget.reserve()
