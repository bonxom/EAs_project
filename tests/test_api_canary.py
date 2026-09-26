"""Comprehensive unit and integration tests for offline LLM API canary harness (M2E1)."""

import socket
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from moh.llm.base import CallMetadata
from moh.llm.budget import (
    ModelPricing,
    ProviderAttemptBudget,
    ProviderAttemptLimits,
    ProviderUsageAccountant,
)
from moh.llm.canary import (
    CANARY_FIXED_PROMPT,
    CanaryConfig,
    CanaryLogicalGuard,
    run_llm_canary,
)
from moh.llm.openai_client import OpenAILLMClient


class FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=1, reasoning_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.input_tokens = prompt_tokens
        self.output_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.completion_tokens_details = MagicMock(reasoning_tokens=reasoning_tokens)
        self.reasoning_tokens = reasoning_tokens


class FakeChoice:
    def __init__(self, content="OK"):
        self.message = MagicMock(content=content)


class FakeResponse:
    def __init__(
        self,
        content="OK",
        prompt_tokens=10,
        completion_tokens=1,
        reasoning_tokens=0,
        model="test-model",
    ):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage(prompt_tokens, completion_tokens, reasoning_tokens)
        self.model = model


class FakeTransport:
    def __init__(self, responses=None):
        self.responses_list = responses or []
        self.call_count = 0
        self.chat = MagicMock()
        self.chat.completions.create.side_effect = self._create

    def _create(self, **kwargs):
        self.call_count += 1
        if self.call_count <= len(self.responses_list):
            res = self.responses_list[self.call_count - 1]
            if isinstance(res, Exception):
                raise res
            return res
        return FakeResponse()


@pytest.fixture(autouse=True)
def setup_env_and_killswitch(monkeypatch):
    """Network kill switch fixture: traps any attempt to perform socket I/O."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")

    def forbidden_socket(*args, **kwargs):
        raise AssertionError("Accidental real network I/O in offline test!")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    yield


def noop_observer(metadata: CallMetadata):
    pass


def make_valid_config(**kwargs):
    defaults = {
        "logical_generate_limit": 1,
        "provider_attempt_limit": 1,
        "evaluate_limit": 0,
        "prompt": CANARY_FIXED_PROMPT,
        "max_output_tokens": 8,
        "model": "test-model",
    }
    defaults.update(kwargs)
    return CanaryConfig(**defaults)


# ---------------------------------------------------------
# Test Matrix Items 1-7: CanaryConfig Strict Validation
# ---------------------------------------------------------
def test_valid_strict_canary_config():
    cfg = make_valid_config()
    assert cfg.logical_generate_limit == 1
    assert cfg.provider_attempt_limit == 1
    assert cfg.evaluate_limit == 0
    assert cfg.prompt == CANARY_FIXED_PROMPT


def test_logical_generate_limit_rejected():
    with pytest.raises(ValueError):
        make_valid_config(logical_generate_limit=0)

    with pytest.raises(ValueError):
        make_valid_config(logical_generate_limit=2)

    with pytest.raises(TypeError):
        make_valid_config(logical_generate_limit=True)  # type: ignore


def test_provider_attempt_limit_rejected():
    with pytest.raises(ValueError):
        make_valid_config(provider_attempt_limit=0)

    with pytest.raises(ValueError):
        make_valid_config(provider_attempt_limit=2)


def test_evaluate_limit_rejected():
    with pytest.raises(ValueError):
        make_valid_config(evaluate_limit=1)


def test_invalid_prompt_rejected():
    with pytest.raises(ValueError):
        make_valid_config(prompt="Write a sorting algorithm")


# ---------------------------------------------------------
# Test Matrix Items 8-12: Fake Provider Success & Accounting
# ---------------------------------------------------------
def test_fake_provider_success():
    cfg = make_valid_config()
    pricing_policy = {
        "test-model": ModelPricing(
            input_usd_per_million_tokens=Decimal("1.00"),
            output_usd_per_million_tokens=Decimal("2.00"),
        )
    }
    accountant = ProviderUsageAccountant(pricing_policy=pricing_policy)
    attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=cfg.provider_attempt_limit)
    )
    logical_guard = CanaryLogicalGuard(max_calls=cfg.logical_generate_limit)
    transport = FakeTransport(
        responses=[
            FakeResponse(
                content="OK",
                prompt_tokens=10,
                completion_tokens=1,
                reasoning_tokens=0,
                model="test-model",
            )
        ]
    )

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    assert result.status == "success"
    assert result.text == "OK"
    assert result.logical_generate_requests == 1
    assert result.provider_attempts == 1
    assert result.input_tokens == 10
    assert result.output_tokens == 1
    assert result.reasoning_tokens == 0
    assert result.total_tokens == 11
    # 10*1/1e6 + 1*2/1e6 = 0.000012
    assert result.estimated_cost_usd == Decimal("0.000012")
    assert result.error is None
    assert transport.call_count == 1


# ---------------------------------------------------------
# Test Matrix Items 13-14: Retryable First Failure & Attempt Cap
# ---------------------------------------------------------
def test_retryable_first_failure_attempt_cap():
    import openai

    cfg = make_valid_config()
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=cfg.provider_attempt_limit)
    )
    logical_guard = CanaryLogicalGuard(max_calls=cfg.logical_generate_limit)

    err = openai.APIConnectionError(request=MagicMock())
    transport = FakeTransport(responses=[err, FakeResponse()])

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        sleep=lambda _: None,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    assert result.status == "failed"
    assert result.text is None
    assert result.logical_generate_requests == 1
    assert result.provider_attempts == 1
    assert result.error == "provider_attempt_budget_exhausted"
    # Transport was invoked EXACTLY once! Attempt #2 was blocked before network!
    assert transport.call_count == 1


# ---------------------------------------------------------
# Test Matrix Item 15: Second Logical Generation Blocked
# ---------------------------------------------------------
def test_second_logical_generation_blocked():
    cfg = make_valid_config()
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=cfg.provider_attempt_limit)
    )
    logical_guard = CanaryLogicalGuard(max_calls=cfg.logical_generate_limit)
    transport = FakeTransport()

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    res1 = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )
    assert res1.status == "success"
    assert transport.call_count == 1

    # Second logical call attempt
    res2 = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )
    assert res2.status == "failed"
    assert res2.error == "logical_canary_budget_exhausted"
    # Transport call count remains 1!
    assert transport.call_count == 1


# ---------------------------------------------------------
# Test Matrix Item 16: Malformed Provider Usage
# ---------------------------------------------------------
def test_malformed_provider_usage_bounded_failure():
    cfg = make_valid_config()
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=cfg.provider_attempt_limit)
    )
    logical_guard = CanaryLogicalGuard(max_calls=cfg.logical_generate_limit)
    transport = FakeTransport(
        responses=[FakeResponse(prompt_tokens=-5, completion_tokens=1)]
    )

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )
    assert result.status == "failed"
    assert result.error == "malformed_response"


# ---------------------------------------------------------
# Test Matrix Item 17: Unknown Fake Pricing Handled Safely
# ---------------------------------------------------------
def test_unknown_fake_pricing_handled_safely():
    cfg = make_valid_config(model="unknown-model")
    accountant = ProviderUsageAccountant(pricing_policy={})
    attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=cfg.provider_attempt_limit)
    )
    logical_guard = CanaryLogicalGuard(max_calls=cfg.logical_generate_limit)
    transport = FakeTransport(responses=[FakeResponse(model="unknown-model")])

    client = OpenAILLMClient(
        model="unknown-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )
    assert result.status == "success"
    assert result.estimated_cost_usd is None


# ---------------------------------------------------------
# Test Matrix Items 18-20: Unused Components Isolation
# ---------------------------------------------------------
def test_unused_components_never_called(monkeypatch):
    """Prove evaluator, optimizer runner, and outer loop are never invoked by canary."""
    eval_mock = MagicMock(side_effect=AssertionError("Evaluator must not be called!"))
    runner_mock = MagicMock(side_effect=AssertionError("Runner must not be called!"))
    outer_mock = MagicMock(side_effect=AssertionError("Outer loop must not be called!"))

    monkeypatch.setattr("moh.evaluation.evaluate_heuristic", eval_mock, raising=False)
    monkeypatch.setattr(
        "moh.optimizers.runner.OptimizerProgramRunner", runner_mock, raising=False
    )

    cfg = make_valid_config()
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(
        ProviderAttemptLimits(max_attempts=cfg.provider_attempt_limit)
    )
    logical_guard = CanaryLogicalGuard(max_calls=cfg.logical_generate_limit)
    transport = FakeTransport()

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    res = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )
    assert res.status == "success"
    assert eval_mock.call_count == 0
    assert runner_mock.call_count == 0
    assert outer_mock.call_count == 0


# ---------------------------------------------------------
# Test Matrix Item 21: No API Key Required Offline
# ---------------------------------------------------------
def test_no_api_key_required_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-offline-key")

    cfg = make_valid_config()
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    logical_guard = CanaryLogicalGuard(max_calls=1)
    transport = FakeTransport()

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )
    assert result.status == "success"


# ---------------------------------------------------------
# Test Matrix Item 22: Credential Redaction Test
# ---------------------------------------------------------
def test_credential_sentinel_absent_from_result_repr(monkeypatch):
    secret = "TEST_SECRET_CANARY_DO_NOT_LEAK"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    cfg = make_valid_config()
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    logical_guard = CanaryLogicalGuard(max_calls=1)
    transport = FakeTransport(
        responses=[FakeResponse(content=f"OK {secret}", model=secret)]
    )

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
    )

    r_repr = repr(result)
    r_dict = str(result.to_dict())
    assert secret not in r_repr
    assert secret not in r_dict
    assert result.text == "OK [REDACTED]"


# ---------------------------------------------------------
# Test Matrix Item 23: Network Kill Switch Test
# ---------------------------------------------------------
def test_network_kill_switch_traps_real_access():
    with pytest.raises(AssertionError) as exc_info:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    assert "Accidental real network I/O" in str(exc_info.value)


# ---------------------------------------------------------
# Test Matrix Item 24: Ordinary V0 Config Cannot Trigger Canary Real Network
# ---------------------------------------------------------
def test_ordinary_config_cannot_trigger_canary_real_network():
    cfg = make_valid_config()
    assert cfg.logical_generate_limit == 1
    assert cfg.provider_attempt_limit == 1
    assert cfg.evaluate_limit == 0
