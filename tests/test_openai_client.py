from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from moh.llm.base import GenerationError
from moh.llm.openai_client import OpenAILLMClient, validate_environment


class Transport:
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


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "offline-test-key")


@pytest.mark.parametrize("failures", [0, 1, 2, 3])
def test_bounded_retries(failures):
    transport = Transport([transient() for _ in range(failures)] + [response()])
    metadata, delays = [], []
    client = OpenAILLMClient(
        "test-model", 2.0, metadata.append, transport=transport, sleep=delays.append
    )
    if failures == 3:
        with pytest.raises(GenerationError):
            client.generate("prompt")
    else:
        assert client.generate("prompt") == "hello"
        assert metadata[-1].input_tokens == 10
    assert len(transport.requests) == min(failures + 1, 3)
    assert len(metadata) == len(transport.requests)
    assert all(x["timeout"] == 2.0 for x in transport.requests)
    assert len(delays) == min(failures, 2)


def test_permanent_error():
    error = openai.AuthenticationError(
        "offline-test-key",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://example.invalid")
        ),
        body=None,
    )
    transport = Transport([error])
    metadata = []
    client = OpenAILLMClient("test-model", 2.0, metadata.append, transport=transport)
    with pytest.raises(GenerationError) as caught:
        client.generate("prompt")
    assert len(transport.requests) == 1
    assert "offline-test-key" not in str(caught.value) + str(metadata)


@pytest.mark.parametrize("value", [response(""), response(None), object()])
def test_bad_response(value):
    transport = Transport([value])
    client = OpenAILLMClient("test-model", 2.0, lambda _: None, transport=transport)
    with pytest.raises(GenerationError):
        client.generate("prompt")
    assert len(transport.requests) == 1


def test_observer_error_not_retried():
    transport = Transport([response()])

    def observer(_):
        raise OSError("disk failed")

    client = OpenAILLMClient("test-model", 2.0, observer, transport=transport)
    with pytest.raises(OSError):
        client.generate("prompt")
    assert len(transport.requests) == 1


def test_preconditions(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    for key in ["", "  "]:
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", key)
        with pytest.raises(ValueError) as excinfo:
            validate_environment()
        assert "OPENAI_COMPAT_API_KEY must be configured" in str(excinfo.value)

    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    with pytest.raises(ValueError) as excinfo:
        validate_environment()
    assert "OPENAI_COMPAT_API_KEY must be configured" in str(excinfo.value)

    # Legacy key alone must fail validation
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-secret-only")
    with pytest.raises(ValueError) as excinfo:
        validate_environment()
    assert "OPENAI_COMPAT_API_KEY must be configured" in str(excinfo.value)

    # Legacy key does not rescue blank canonical key
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "   ")
    with pytest.raises(ValueError):
        validate_environment()

    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "offline-test-key")
    for model, timeout in [("", 2.0), ("test", 0), ("test", float("inf"))]:
        with pytest.raises(ValueError):
            OpenAILLMClient(model, timeout, lambda _: None, transport=Transport([]))


def test_last_provider_error_preserved_on_budget_exhaustion():
    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptBudgetExceeded,
        ProviderAttemptLimits,
    )

    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    transport = Transport([transient(), response()])
    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderAttemptBudgetExceeded) as caught:
        client.generate("prompt")

    assert caught.value.code == "provider_attempt_budget_exhausted"
    assert caught.value.last_provider_error == "provider_connection_error"
    assert not hasattr(client, "last_provider_error")
    assert len(transport.requests) == 1
    assert budget.usage.attempts == 1


def test_retry_blocked_on_timeout():
    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptBudgetExceeded,
        ProviderAttemptLimits,
    )

    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    err = openai.APITimeoutError(
        request=httpx.Request("POST", "https://example.invalid")
    )
    transport = Transport([err, response()])
    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderAttemptBudgetExceeded) as caught:
        client.generate("prompt")

    assert caught.value.code == "provider_attempt_budget_exhausted"
    assert caught.value.last_provider_error == "provider_timeout"
    assert not hasattr(client, "last_provider_error")
    assert len(transport.requests) == 1
    assert budget.usage.attempts == 1


def test_transient_failure_then_success():
    from moh.llm.budget import ProviderAttemptBudget, ProviderAttemptLimits

    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=3))
    transport = Transport([transient(), response("success text")])
    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
        sleep=lambda _: None,
    )
    res = client.generate("prompt")
    assert res == "success text"
    assert len(transport.requests) == 2
    assert budget.usage.attempts == 2
    assert not hasattr(client, "last_provider_error")


def test_same_client_failure_then_success():
    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptBudgetExceeded,
        ProviderAttemptLimits,
    )

    b1 = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    t1 = Transport([transient(), response("r1")])
    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=t1,
        attempt_budget=b1,
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderAttemptBudgetExceeded):
        client.generate("prompt1")

    # Call 2: fresh transport & budget
    b2 = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    t2 = Transport([response("r2")])
    client.attempt_budget = b2
    client.transport = t2
    res2 = client.generate("prompt2")
    assert res2 == "r2"
    assert not hasattr(client, "last_provider_error")


def test_first_attempt_success():
    transport = Transport([response("first try")])
    client = OpenAILLMClient("test-model", 2.0, lambda _: None, transport=transport)
    res = client.generate("prompt")
    assert res == "first try"
    assert len(transport.requests) == 1
    assert not hasattr(client, "last_provider_error")


def test_secret_redaction():
    from moh.llm.budget import ProviderAttemptBudget, ProviderAttemptLimits

    secret = "TEST_SECRET_MUST_NOT_LEAK"
    err = openai.APIConnectionError(
        message=secret, request=httpx.Request("POST", "https://example.invalid")
    )
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    transport = Transport([err])
    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=transport,
        attempt_budget=budget,
    )
    with pytest.raises(GenerationError) as caught:
        client.generate("prompt")

    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert caught.value.code == "provider_attempt_budget_exhausted"
    assert caught.value.last_provider_error == "provider_connection_error"


def test_generation_error_subclass_codes():
    from moh.llm.budget import (
        ProviderAttemptBudgetExceeded,
        ProviderUsageBudgetExceeded,
    )
    from moh.llm.canary import CanaryLogicalGuardExceeded

    assert ProviderAttemptBudgetExceeded().code == "provider_attempt_budget_exhausted"
    assert CanaryLogicalGuardExceeded().code == "logical_canary_budget_exhausted"
    assert ProviderUsageBudgetExceeded().code == "provider_usage_budget_exhausted"
    assert GenerationError("custom").code == "custom"


def test_both_keys_present_precedence(monkeypatch):
    captured_kwargs = {}

    class FakeTransport:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeTransport)
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "canonical-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-secret-value")

    _ = OpenAILLMClient("test-model", 2.0, lambda _: None, transport=None)
    assert captured_kwargs["api_key"] == "canonical-secret-value"
    assert captured_kwargs["api_key"] != "legacy-secret-value"


def test_sdk_constructor_receives_sentinel_timeout(monkeypatch):
    captured_sdk_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured_sdk_kwargs.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    _ = OpenAILLMClient(
        "test-model",
        12.5,
        lambda _: None,
        transport=None,
    )
    assert captured_sdk_kwargs.get("timeout") == 12.5
    assert captured_sdk_kwargs.get("max_retries") == 0


def test_per_call_receives_sentinel_timeout():
    transport = Transport([response("sentinel test")])
    client = OpenAILLMClient(
        "test-model",
        12.5,
        lambda _: None,
        transport=transport,
    )
    res = client.generate("test prompt")
    assert res == "sentinel test"
    assert len(transport.requests) == 1
    assert transport.requests[0]["timeout"] == 12.5


# ---------------------------------------------------------
# M2E2B-MO: Malformed Response & GenerationError Tests
# ---------------------------------------------------------

def test_generation_error_malformed_response_reason_defaults():
    err1 = GenerationError("foo")
    assert err1.code == "foo"
    assert err1.last_provider_error is None
    assert err1.malformed_response_reason is None

    err2 = GenerationError("malformed_response", malformed_response_reason="empty_text")
    assert err2.code == "malformed_response"
    assert err2.last_provider_error is None
    assert err2.malformed_response_reason == "empty_text"

    err3 = GenerationError("foo", last_provider_error="provider_timeout")
    assert err3.code == "foo"
    assert err3.last_provider_error == "provider_timeout"
    assert err3.malformed_response_reason is None


def make_mock_response(
    content="hello",
    model="test-model",
    input_tokens=10,
    output_tokens=2,
    reasoning_tokens=0,
    total_tokens=12,
    status="completed",
):
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        input_tokens=input_tokens,
        completion_tokens=output_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens)
        if reasoning_tokens is not None
        else None,
    )
    return SimpleNamespace(
        output_text=content,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))] if isinstance(content, str) else None,
        model=model,
        usage=usage,
        status=status,
    )


@pytest.mark.parametrize(
    "resp_kwargs, expected_reason",
    [
        ({"content": None}, "missing_text"),
        ({"content": ""}, "empty_text"),
        ({"content": "   "}, "empty_text"),
        ({"content": 123}, "invalid_text_type"),
        ({"model": None}, "invalid_model"),
        ({"status": "failed"}, "incomplete_response"),
        ({"input_tokens": True, "total_tokens": 12}, "invalid_input_tokens_type"),
        ({"input_tokens": -1, "total_tokens": 1}, "negative_input_tokens"),
        ({"output_tokens": True, "total_tokens": 12}, "invalid_output_tokens_type"),
        ({"output_tokens": -1, "total_tokens": 9}, "negative_output_tokens"),
        ({"reasoning_tokens": True, "total_tokens": 12}, "invalid_reasoning_tokens_type"),
        ({"reasoning_tokens": -1, "total_tokens": 12}, "negative_reasoning_tokens"),
        ({"total_tokens": True}, "invalid_total_tokens_type"),
        ({"total_tokens": -1}, "negative_total_tokens"),
        ({"reasoning_tokens": 10, "output_tokens": 5, "total_tokens": 15}, "reasoning_exceeds_output"),
        (
            {"input_tokens": 5, "output_tokens": 5, "total_tokens": 20},
            "inconsistent_total_tokens",
        ),
    ],
)
def test_openai_client_malformed_response_reasons(resp_kwargs, expected_reason):
    resp = make_mock_response(**resp_kwargs)
    client = OpenAILLMClient("test-model", 2.0, lambda _: None, transport=Transport([resp]))
    with pytest.raises(GenerationError) as exc_info:
        client.generate("prompt")

    assert exc_info.value.code == "malformed_response"
    assert exc_info.value.malformed_response_reason == expected_reason
    assert exc_info.value.last_provider_error is None
    assert not hasattr(client, "malformed_response_reason")


def test_openai_client_usage_accountant_rejected():
    from unittest.mock import MagicMock

    from moh.llm.budget import ProviderUsageAccountant

    accountant = MagicMock(spec=ProviderUsageAccountant)
    accountant.record.side_effect = ValueError("Accountant cap exceeded")

    resp = make_mock_response(content="hello")
    client = OpenAILLMClient(
        "test-model",
        2.0,
        lambda _: None,
        transport=Transport([resp]),
        usage_accountant=accountant,
    )
    with pytest.raises(GenerationError) as exc_info:
        client.generate("prompt")

    assert exc_info.value.code == "malformed_response"
    assert exc_info.value.malformed_response_reason == "usage_accountant_rejected"
    assert exc_info.value.last_provider_error is None
    assert not hasattr(client, "malformed_response_reason")


def test_openai_client_none_token_semantics_accepted():
    resp = make_mock_response(
        content="hello",
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        total_tokens=None,
    )
    client = OpenAILLMClient("test-model", 2.0, lambda _: None, transport=Transport([resp]))
    result = client.generate("prompt")
    assert result == "hello"



# ---------------------------------------------------------
# M2E2B-UN: Token Usage Semantics Normalization Tests
# ---------------------------------------------------------

from moh.llm.openai_client import _normalize_token_usage_semantics


def test_normalize_native_openai_noop():
    # 21, 8, 6, 29 -> native OpenAI (total == 21+8, 6 <= 8)
    inp, out, reas, tot = _normalize_token_usage_semantics(21, 8, 6, 29)
    assert (inp, out, reas, tot) == (21, 8, 6, 29)

def test_normalize_r3_9router_separate_thinking():
    # 21, 2, 6, 29 -> 9router separate thinking (total == 21+2+6)
    inp, out, reas, tot = _normalize_token_usage_semantics(21, 2, 6, 29)
    assert (inp, out, reas, tot) == (21, 8, 6, 29)

def test_normalize_separate_thinking_reasoning_le_visible():
    # 10, 7, 3, 20 -> separate thinking with reasoning <= output (total == 10+7+3)
    inp, out, reas, tot = _normalize_token_usage_semantics(10, 7, 3, 20)
    assert (inp, out, reas, tot) == (10, 10, 3, 20)

def test_normalize_zero_reasoning_unchanged():
    inp, out, reas, tot = _normalize_token_usage_semantics(10, 2, 0, 12)
    assert (inp, out, reas, tot) == (10, 2, 0, 12)

def test_normalize_inconsistent_counts_not_normalized():
    # 10, 2, 6, 17 -> neither 10+2 nor 10+2+6 matches 17
    inp, out, reas, tot = _normalize_token_usage_semantics(10, 2, 6, 17)
    assert (inp, out, reas, tot) == (10, 2, 6, 17)

def test_normalize_missing_total_not_normalized():
    # 10, 2, 6, None -> total is None
    inp, out, reas, tot = _normalize_token_usage_semantics(10, 2, 6, None)
    assert (inp, out, reas, tot) == (10, 2, 6, None)

def test_openai_client_generate_r3_9router_usage_succeeds():
    resp = make_mock_response(
        content="OK",
        input_tokens=21,
        output_tokens=2,
        reasoning_tokens=6,
        total_tokens=29,
    )
    client = OpenAILLMClient("test-model", 2.0, lambda _: None, transport=Transport([resp]))
    res = client.generate("prompt")
    assert res == "OK"

def test_openai_client_inconsistent_usage_stays_malformed():
    resp = make_mock_response(
        content="OK",
        input_tokens=10,
        output_tokens=2,
        reasoning_tokens=6,
        total_tokens=17,
    )
    client = OpenAILLMClient("test-model", 2.0, lambda _: None, transport=Transport([resp]))
    with pytest.raises(GenerationError) as exc_info:
        client.generate("prompt")
    assert exc_info.value.code == "malformed_response"
    assert exc_info.value.malformed_response_reason == "reasoning_exceeds_output"
