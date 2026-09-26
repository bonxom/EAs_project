"""Tests for provider token/cost accounting and future-request guards (M2D2)."""

import os
import threading
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from moh.llm.base import CallMetadata, GenerationError
from moh.llm.budget import (
    ModelPricing,
    ProviderAttemptBudget,
    ProviderAttemptLimits,
    ProviderAttemptUsage,
    ProviderTokenUsage,
    ProviderUsageAccountant,
    ProviderUsageBudgetExceeded,
    ProviderUsageLimits,
)
from moh.llm.openai_client import OpenAILLMClient


class FakeUsage:
    def __init__(self, prompt_tokens=None, completion_tokens=None, reasoning_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.input_tokens = prompt_tokens
        self.output_tokens = completion_tokens
        if reasoning_tokens is not None:
            self.completion_tokens_details = MagicMock(reasoning_tokens=reasoning_tokens)
            self.reasoning_tokens = reasoning_tokens


class FakeChoice:
    def __init__(self, content="generated response"):
        self.message = MagicMock(content=content)


class FakeResponse:
    def __init__(
        self,
        content="generated response",
        prompt_tokens=100,
        completion_tokens=25,
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
def setup_env():
    os.environ["OPENAI_API_KEY"] = "sk-test-key-12345"
    yield
    os.environ.pop("OPENAI_API_KEY", None)


def noop_observer(metadata: CallMetadata):
    pass


# ---------------------------------------------------------
# Step 2: Token Usage Model Unit Tests
# ---------------------------------------------------------
def test_provider_token_usage_validation():
    usage = ProviderTokenUsage(input_tokens=100, output_tokens=50, reasoning_tokens=10)
    assert usage.total_tokens == 160

    with pytest.raises(TypeError):
        ProviderTokenUsage(input_tokens=True)  # type: ignore

    with pytest.raises(TypeError):
        ProviderTokenUsage(output_tokens="10")  # type: ignore

    with pytest.raises(ValueError):
        ProviderTokenUsage(input_tokens=-1)


# ---------------------------------------------------------
# Step 4: Usage Limits Model Unit Tests
# ---------------------------------------------------------
def test_provider_usage_limits_validation():
    limits = ProviderUsageLimits(
        max_input_tokens=100,
        max_output_tokens=50,
        max_total_tokens=150,
        max_estimated_cost_usd=Decimal("0.05"),
    )
    assert limits.max_estimated_cost_usd == Decimal("0.05")

    with pytest.raises(TypeError):
        ProviderUsageLimits(max_input_tokens=True)  # type: ignore

    with pytest.raises(ValueError):
        ProviderUsageLimits(max_total_tokens=-10)

    with pytest.raises(TypeError):
        ProviderUsageLimits(max_estimated_cost_usd=True)  # type: ignore

    with pytest.raises(ValueError):
        ProviderUsageLimits(max_estimated_cost_usd=Decimal("-1.0"))


# ---------------------------------------------------------
# Step 7 & 8: Pricing & Cost Calculation
# ---------------------------------------------------------
def test_model_pricing_cost_calculation():
    pricing = ModelPricing(
        input_usd_per_million_tokens=Decimal("1.50"),
        output_usd_per_million_tokens=Decimal("6.00"),
        reasoning_usd_per_million_tokens=Decimal("10.00"),
    )
    # 1,000,000 input = $1.50
    # 1,000,000 output = $6.00
    # 100,000 reasoning = $1.00
    cost = pricing.calculate_cost(
        input_tokens=1_000_000, output_tokens=1_000_000, reasoning_tokens=100_000
    )
    assert cost == Decimal("8.50")


# ---------------------------------------------------------
# Step 12: Single Response Test
# ---------------------------------------------------------
def test_single_response_accounting():
    pricing_policy = {
        "test-model": ModelPricing(
            input_usd_per_million_tokens=Decimal("2.00"),
            output_usd_per_million_tokens=Decimal("4.00"),
        )
    }
    accountant = ProviderUsageAccountant(pricing_policy=pricing_policy)
    transport = FakeTransport(
        responses=[
            FakeResponse(
                content="hello",
                prompt_tokens=100,
                completion_tokens=25,
                reasoning_tokens=0,
                model="test-model",
            )
        ]
    )
    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
    )

    res = client.generate("test prompt")
    assert res == "hello"

    u = accountant.usage
    assert u.input_tokens == 100
    assert u.output_tokens == 25
    assert u.reasoning_tokens == 0
    assert u.total_tokens == 125

    # 100 * 2.00 / 1e6 = 0.0002
    # 25 * 4.00 / 1e6 = 0.0001
    # total cost = 0.0003
    assert accountant.estimated_cost_usd == Decimal("0.0003")


# ---------------------------------------------------------
# Step 13: Multi-Call Aggregation
# ---------------------------------------------------------
def test_multi_call_aggregation():
    pricing_policy = {
        "test-model": ModelPricing(
            input_usd_per_million_tokens=Decimal("1.00"),
            output_usd_per_million_tokens=Decimal("2.00"),
        )
    }
    accountant = ProviderUsageAccountant(pricing_policy=pricing_policy)
    transport = FakeTransport(
        responses=[
            FakeResponse(prompt_tokens=100, completion_tokens=20, model="test-model"),
            FakeResponse(prompt_tokens=80, completion_tokens=30, model="test-model"),
        ]
    )
    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
    )

    client.generate("prompt 1")
    client.generate("prompt 2")

    u = accountant.usage
    assert u.input_tokens == 180
    assert u.output_tokens == 50
    assert u.total_tokens == 230

    # Call 1: 100*1 + 20*2 = 140 / 1e6 = 0.00014
    # Call 2: 80*1 + 30*2 = 140 / 1e6 = 0.00014
    # Total cost = 0.00028
    assert accountant.estimated_cost_usd == Decimal("0.00028")


# ---------------------------------------------------------
# Step 14: Exact Limit Block
# ---------------------------------------------------------
def test_exact_limit_block():
    accountant = ProviderUsageAccountant()
    limits = ProviderUsageLimits(max_total_tokens=200)

    # Manually record 200 tokens
    accountant.record(
        model="test-model", input_tokens=150, output_tokens=50, reasoning_tokens=0
    )
    assert accountant.usage.total_tokens == 200

    transport = FakeTransport()
    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        usage_limits=limits,
    )

    with pytest.raises(ProviderUsageBudgetExceeded) as exc_info:
        client.generate("prompt")

    assert exc_info.value.code == "provider_total_token_budget_exhausted"
    assert transport.call_count == 0


# ---------------------------------------------------------
# Step 15: Overshoot Then Block
# ---------------------------------------------------------
def test_overshoot_then_block():
    accountant = ProviderUsageAccountant()
    limits = ProviderUsageLimits(max_total_tokens=200)

    # Pre-record 180 tokens
    accountant.record(model="test-model", input_tokens=150, output_tokens=30)
    assert accountant.usage.total_tokens == 180

    transport = FakeTransport(
        responses=[
            FakeResponse(prompt_tokens=30, completion_tokens=20, model="test-model")
        ]
    )
    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        usage_limits=limits,
    )

    # Admitted! (180 < 200)
    res = client.generate("prompt 1")
    assert res == "generated response"
    assert transport.call_count == 1
    assert accountant.usage.total_tokens == 230  # Overshot 200 to 230!

    # Next request must be blocked before network I/O
    with pytest.raises(ProviderUsageBudgetExceeded) as exc_info:
        client.generate("prompt 2")

    assert exc_info.value.code == "provider_total_token_budget_exhausted"
    assert transport.call_count == 1  # Network transport invocation count unchanged!


# ---------------------------------------------------------
# Step 16: Zero Limit
# ---------------------------------------------------------
def test_zero_limit():
    accountant = ProviderUsageAccountant()
    limits = ProviderUsageLimits(max_total_tokens=0)
    transport = FakeTransport()
    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        usage_limits=limits,
    )

    with pytest.raises(ProviderUsageBudgetExceeded) as exc_info:
        client.generate("prompt")

    assert exc_info.value.code == "provider_total_token_budget_exhausted"
    assert transport.call_count == 0


def test_zero_input_and_output_limits():
    accountant = ProviderUsageAccountant()
    limits = ProviderUsageLimits(max_input_tokens=0, max_output_tokens=0)
    transport = FakeTransport()
    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        usage_limits=limits,
    )

    with pytest.raises(ProviderUsageBudgetExceeded) as exc_info:
        client.generate("prompt")

    assert exc_info.value.code == "provider_input_token_budget_exhausted"
    assert transport.call_count == 0


# ---------------------------------------------------------
# Step 17: Cost Limit
# ---------------------------------------------------------
def test_cost_limit():
    pricing_policy = {
        "test-model": ModelPricing(
            input_usd_per_million_tokens=Decimal("1.00"),
            output_usd_per_million_tokens=Decimal("2.00"),
        )
    }
    accountant = ProviderUsageAccountant(pricing_policy=pricing_policy)
    # $0.0002 limit
    limits = ProviderUsageLimits(max_estimated_cost_usd=Decimal("0.0002"))

    transport = FakeTransport(
        responses=[
            # Call 1: 100 in, 50 out -> cost = 100*1e-6 + 50*2*1e-6 = 0.0002
            FakeResponse(prompt_tokens=100, completion_tokens=50, model="test-model"),
            # Call 2 should be blocked
            FakeResponse(prompt_tokens=10, completion_tokens=10, model="test-model"),
        ]
    )
    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        usage_limits=limits,
    )

    client.generate("prompt 1")
    assert accountant.estimated_cost_usd == Decimal("0.0002")

    with pytest.raises(ProviderUsageBudgetExceeded) as exc_info:
        client.generate("prompt 2")

    assert exc_info.value.code == "provider_cost_budget_exhausted"
    assert transport.call_count == 1


# ---------------------------------------------------------
# Step 18: Unknown Pricing (Case A & Case B)
# ---------------------------------------------------------
def test_unknown_pricing_case_a():
    """Case A: pricing unknown, no cost limit -> generation succeeds, cost unavailable."""
    accountant = ProviderUsageAccountant(pricing_policy={})
    transport = FakeTransport(
        responses=[
            FakeResponse(
                prompt_tokens=100, completion_tokens=25, model="unknown-model"
            )
        ]
    )
    client = OpenAILLMClient(
        model="unknown-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        usage_limits=ProviderUsageLimits(),  # No cost limit
    )

    res = client.generate("prompt")
    assert res == "generated response"
    assert accountant.usage.total_tokens == 125
    assert accountant.estimated_cost_usd is None


def test_unknown_pricing_case_b():
    """Case B: pricing unknown, cost limit configured -> fails closed."""
    accountant = ProviderUsageAccountant(pricing_policy={})
    limits = ProviderUsageLimits(max_estimated_cost_usd=Decimal("0.10"))
    transport = FakeTransport()
    client = OpenAILLMClient(
        model="unknown-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        usage_limits=limits,
    )

    with pytest.raises(ProviderUsageBudgetExceeded) as exc_info:
        client.generate("prompt")

    assert (
        exc_info.value.code == "provider_cost_budget_model_pricing_unavailable"
    )
    assert transport.call_count == 0


# ---------------------------------------------------------
# Step 19: Malformed Provider Usage
# ---------------------------------------------------------
def test_malformed_provider_usage():
    accountant = ProviderUsageAccountant()

    # Case 1: negative token count
    t1 = FakeTransport(
        responses=[FakeResponse(prompt_tokens=-10, completion_tokens=20)]
    )
    c1 = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=t1,
        usage_accountant=accountant,
    )
    with pytest.raises(GenerationError) as exc_info:
        c1.generate("prompt")
    assert str(exc_info.value) == "malformed_response"

    # Case 2: bool token count
    t2 = FakeTransport(
        responses=[FakeResponse(prompt_tokens=True, completion_tokens=20)]
    )
    c2 = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=t2,
        usage_accountant=accountant,
    )
    with pytest.raises(GenerationError) as exc_info:
        c2.generate("prompt")
    assert str(exc_info.value) == "malformed_response"

    # Case 3: string token count
    t3 = FakeTransport(
        responses=[FakeResponse(prompt_tokens="100", completion_tokens=20)]
    )
    c3 = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=t3,
        usage_accountant=accountant,
    )
    with pytest.raises(GenerationError) as exc_info:
        c3.generate("prompt")
    assert str(exc_info.value) == "malformed_response"


# ---------------------------------------------------------
# Step 20: Thread Safety
# ---------------------------------------------------------
def test_accountant_thread_safety():
    pricing_policy = {
        "test-model": ModelPricing(
            input_usd_per_million_tokens=Decimal("1.00"),
            output_usd_per_million_tokens=Decimal("2.00"),
        )
    }
    accountant = ProviderUsageAccountant(pricing_policy=pricing_policy)

    def worker():
        for _ in range(50):
            accountant.record(
                model="test-model", input_tokens=10, output_tokens=5
            )

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 10 threads * 50 iterations * 10 input = 5,000 input
    # 10 threads * 50 iterations * 5 output = 2,500 output
    u = accountant.usage
    assert u.input_tokens == 5000
    assert u.output_tokens == 2500
    assert u.total_tokens == 7500


# ---------------------------------------------------------
# Step 21: Counter Distinction Check
# ---------------------------------------------------------
def test_counter_distinction_semantics():
    """Verify that ProviderAttemptUsage, ProviderTokenUsage, and ProviderUsageLimits remain distinct abstractions."""
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=5))
    accountant = ProviderUsageAccountant()

    attempt_budget.reserve()
    accountant.record("test-model", input_tokens=100, output_tokens=50)

    attempt_usage = attempt_budget.usage
    token_usage = accountant.usage

    assert isinstance(attempt_usage, ProviderAttemptUsage)
    assert isinstance(token_usage, ProviderTokenUsage)
    assert attempt_usage.attempts == 1
    assert token_usage.input_tokens == 100
    assert token_usage.output_tokens == 50


# ---------------------------------------------------------
# Step 22: Cross-Level Offline Example
# ---------------------------------------------------------
def test_cross_level_offline_example():
    """Demonstrate: 1 logical request, 2 provider attempts, 125 provider-reported tokens."""
    import openai

    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=3))
    accountant = ProviderUsageAccountant()

    # Attempt 1: transient connection error
    # Attempt 2: success with 100 input + 25 output = 125 tokens
    err = openai.APIConnectionError(request=MagicMock())
    success_resp = FakeResponse(
        prompt_tokens=100, completion_tokens=25, model="test-model"
    )

    transport = FakeTransport(responses=[err, success_resp])

    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        sleep=lambda _: None,  # Fast retry
    )

    # 1 logical generate call
    res = client.generate("test prompt")
    assert res == "generated response"

    # 2 provider attempts admitted
    assert attempt_budget.usage.attempts == 2

    # 125 tokens recorded (only from successful attempt 2)
    assert accountant.usage.total_tokens == 125
