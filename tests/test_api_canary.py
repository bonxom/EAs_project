"""Comprehensive unit and integration tests for bounded LLM API canary harness (M2E2A)."""

import json
import socket
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from moh.llm.base import CallMetadata
from moh.llm.budget import (
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
        self.output_text = content
        self.status = "completed"


class FakeTransport:
    def __init__(self, responses=None):
        self.responses_list = responses or []
        self.call_count = 0
        self.last_chat_kwargs = None
        self.last_responses_kwargs = None
        self.chat = MagicMock()
        self.chat.completions.create.side_effect = self._create_chat
        self.responses = MagicMock()
        self.responses.create.side_effect = self._create_responses

    def _create_chat(self, **kwargs):
        self.call_count += 1
        self.last_chat_kwargs = kwargs
        if self.call_count <= len(self.responses_list):
            res = self.responses_list[self.call_count - 1]
            if isinstance(res, Exception):
                raise res
            return res
        return FakeResponse()

    def _create_responses(self, **kwargs):
        self.call_count += 1
        self.last_responses_kwargs = kwargs
        if self.call_count <= len(self.responses_list):
            res = self.responses_list[self.call_count - 1]
            if isinstance(res, Exception):
                raise res
            return res
        res = FakeResponse()
        res.output_text = "OK"
        res.status = "completed"
        return res


@pytest.fixture(autouse=True)
def setup_env_and_killswitch(monkeypatch):
    """Network kill switch fixture: traps any attempt to perform real socket I/O."""
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-test-key-12345")

    def forbidden_socket(*args, **kwargs):
        raise AssertionError("Accidental real network I/O in offline test!")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    yield


def noop_observer(metadata: CallMetadata):
    pass


def make_valid_config(**kwargs):
    defaults = {
        "mode": "offline",
        "logical_generate_limit": 1,
        "provider_attempt_limit": 1,
        "evaluate_limit": 0,
        "prompt": CANARY_FIXED_PROMPT,
        "max_output_tokens": 8,
        "model": "test-model",
        "timeout_seconds": 5.0,
    }
    defaults.update(kwargs)
    return CanaryConfig(**defaults)


# ---------------------------------------------------------
# Test Matrix Items 1-7: CanaryConfig Strict Validation
# ---------------------------------------------------------
def test_valid_strict_canary_config_offline():
    cfg = make_valid_config(mode="offline")
    assert cfg.mode == "offline"
    assert cfg.logical_generate_limit == 1
    assert cfg.provider_attempt_limit == 1
    assert cfg.evaluate_limit == 0
    assert cfg.prompt == CANARY_FIXED_PROMPT
    assert cfg.max_output_tokens == 8


def test_valid_strict_canary_config_real():
    cfg = make_valid_config(mode="real", max_output_tokens=8)
    assert cfg.mode == "real"
    assert cfg.max_output_tokens == 8


def test_canary_config_invalid_mode_rejected():
    with pytest.raises(ValueError):
        make_valid_config(mode="invalid_mode")


def test_max_output_tokens_validation_real_mode():
    with pytest.raises(ValueError):
        make_valid_config(mode="real", max_output_tokens=None)

    with pytest.raises(ValueError):
        make_valid_config(mode="real", max_output_tokens=0)

    with pytest.raises(ValueError):
        make_valid_config(mode="real", max_output_tokens=-5)

    with pytest.raises(ValueError):
        make_valid_config(mode="real", max_output_tokens=True)  # type: ignore


def test_max_output_tokens_validation_offline_mode():
    # None is allowed in offline mode
    cfg = make_valid_config(mode="offline", max_output_tokens=None)
    assert cfg.max_output_tokens is None

    with pytest.raises(ValueError):
        make_valid_config(mode="offline", max_output_tokens=0)

    with pytest.raises(ValueError):
        make_valid_config(mode="offline", max_output_tokens=True)  # type: ignore


# ---------------------------------------------------------
# SDK Boundary Kwargs Enforcement (Steps 4 & 5 / Matrix Items 8 & 9)
# ---------------------------------------------------------
def test_responses_sdk_receives_output_cap_8():
    accountant = ProviderUsageAccountant()
    transport = FakeTransport()
    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        max_output_tokens=8,
    )

    res = client.generate("prompt")
    assert res == "OK"
    assert transport.call_count == 1
    assert transport.last_responses_kwargs is not None
    assert transport.last_responses_kwargs["max_output_tokens"] == 8


def test_chat_completions_sdk_receives_output_cap_8():
    accountant = ProviderUsageAccountant()
    # Transport without responses attribute forces chat completions path
    transport = FakeTransport()
    del transport.responses

    client = OpenAILLMClient(
        model="test-model",
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        usage_accountant=accountant,
        max_output_tokens=8,
    )

    res = client.generate("prompt")
    assert res == "OK"
    assert transport.call_count == 1
    assert transport.last_chat_kwargs is not None
    assert transport.last_chat_kwargs["max_completion_tokens"] == 8


# ---------------------------------------------------------
# Double Opt-In & Real Mode Arming Tests (Matrix Items 10-13)
# ---------------------------------------------------------
def test_mode_real_without_allow_flag_blocked():
    cfg = make_valid_config(mode="real", max_output_tokens=8)
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
        max_output_tokens=8,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        allow_real_api=False,  # Flag absent!
    )

    assert result.status == "failed"
    assert result.error == "real_api_not_authorized"
    assert transport.call_count == 0


def test_offline_config_with_allow_flag_remains_offline():
    cfg = make_valid_config(mode="offline")
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
        allow_real_api=True,
    )

    assert result.status == "success"
    assert result.mode == "offline"
    assert transport.call_count == 1


def test_real_mode_allow_flag_missing_credential_blocked(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "   ")  # Blank credential!

    cfg = make_valid_config(mode="real", max_output_tokens=8)
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    logical_guard = CanaryLogicalGuard(max_calls=1)
    transport = FakeTransport()

    # Fail closed BEFORE transport or SDK creation
    client = MagicMock()

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        allow_real_api=True,
    )

    assert result.status == "failed"
    assert result.error == "missing_provider_credential"
    assert transport.call_count == 0


def test_real_mode_allow_flag_dummy_credential_fake_transport_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-armed-key")

    cfg = make_valid_config(mode="real", max_output_tokens=8)
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
        max_output_tokens=8,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        allow_real_api=True,
    )

    assert result.status == "success"
    assert result.mode == "real"
    assert result.max_output_tokens_requested == 8
    assert transport.call_count == 1


# ---------------------------------------------------------
# Hard Bounds & Accounting Tests (Matrix Items 14-16 & 20-22)
# ---------------------------------------------------------
def test_retry_blocked_before_sdk():
    import openai

    cfg = make_valid_config(mode="offline")
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    logical_guard = CanaryLogicalGuard(max_calls=1)

    err = openai.APIConnectionError(request=MagicMock())
    transport = FakeTransport(responses=[err, FakeResponse()])

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        max_output_tokens=8,
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
    assert result.error == "provider_attempt_budget_exhausted"
    assert result.last_provider_error == "provider_connection_error"
    assert transport.call_count == 1
    assert attempt_budget.usage.attempts == 1


def test_retry_blocked_on_timeout():
    import openai

    cfg = make_valid_config(mode="offline")
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    logical_guard = CanaryLogicalGuard(max_calls=1)

    err = openai.APITimeoutError(request=MagicMock())
    transport = FakeTransport(responses=[err, FakeResponse()])

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        max_output_tokens=8,
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
    assert result.error == "provider_attempt_budget_exhausted"
    assert result.last_provider_error == "provider_timeout"
    assert transport.call_count == 1
    assert attempt_budget.usage.attempts == 1


def test_secret_redaction_in_provider_error():
    import openai

    secret = "TEST_SECRET_MUST_NOT_LEAK"
    cfg = make_valid_config(mode="offline")
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    logical_guard = CanaryLogicalGuard(max_calls=1)

    err = openai.APIConnectionError(message=secret, request=MagicMock())
    transport = FakeTransport(responses=[err, FakeResponse()])

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        max_output_tokens=8,
        sleep=lambda _: None,
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
    r_json = json.dumps(result.to_dict())

    assert secret not in r_repr
    assert secret not in r_dict
    assert secret not in r_json
    assert result.error == "provider_attempt_budget_exhausted"
    assert result.last_provider_error == "provider_connection_error"


def test_malformed_usage_fails_real_canary(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-armed-key")

    cfg = make_valid_config(mode="real", max_output_tokens=8)
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    logical_guard = CanaryLogicalGuard(max_calls=1)

    resp = FakeResponse(prompt_tokens=-5, completion_tokens=1)
    transport = FakeTransport(responses=[resp])

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        max_output_tokens=8,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        allow_real_api=True,
    )

    assert result.status == "failed"
    assert result.error == "malformed_response"
    assert transport.call_count == 1


def test_missing_usage_fails_real_canary(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-armed-key")

    cfg = make_valid_config(mode="real", max_output_tokens=8)
    accountant = ProviderUsageAccountant()
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    logical_guard = CanaryLogicalGuard(max_calls=1)

    resp = FakeResponse(prompt_tokens=0, completion_tokens=0)
    transport = FakeTransport(responses=[resp])

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=noop_observer,
        transport=transport,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        max_output_tokens=8,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        allow_real_api=True,
    )

    assert result.status == "failed"
    assert result.error == "missing_provider_usage"
    assert transport.call_count == 1


# ---------------------------------------------------------
# Credential Redaction & Safety (Matrix Item 23)
# ---------------------------------------------------------
def test_credential_sentinel_absent_from_result_repr(monkeypatch):
    secret = "TEST_SECRET_M2E2A_DO_NOT_LEAK"
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", secret)

    cfg = make_valid_config(mode="real", max_output_tokens=8)
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
        max_output_tokens=8,
    )

    result = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=accountant,
        allow_real_api=True,
    )

    r_repr = repr(result)
    r_dict = str(result.to_dict())
    assert secret not in r_repr
    assert secret not in r_dict
    assert result.text == "OK [REDACTED]"


# ---------------------------------------------------------
# CLI Double Opt-In & Runtime Warning Checks (Matrix Items 25-26)
# ---------------------------------------------------------
def test_cli_real_mode_without_allow_flag_outputs_failure_json():
    cmd = [
        sys.executable,
        "-m",
        "moh.llm.canary",
        "--config",
        "configs/api_canary_real.yaml",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "RuntimeWarning" not in proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "failed"
    assert data["error"] == "real_api_not_authorized"
    assert data["max_output_tokens_requested"] == 8



# ---------------------------------------------------------
# M2E2A Real Proxy Wiring & Transport Separation Tests
# ---------------------------------------------------------
@pytest.fixture(autouse=True)
def block_sockets(monkeypatch):
    import socket

    def guard(*args, **kwargs):
        raise RuntimeError("Network socket creation blocked in test suite")

    monkeypatch.setattr(socket, "socket", guard)


def test_base_url_resolution_precedence(monkeypatch):
    from moh.llm.openai_client import resolve_base_url

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://other.invalid/v1")
    assert resolve_base_url() == "http://localhost:20128/v1"

    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL")
    assert resolve_base_url() == "http://other.invalid/v1"

    monkeypatch.delenv("OPENAI_BASE_URL")
    assert resolve_base_url() is None


def test_model_resolution_precedence(monkeypatch, tmp_path):
    from moh.llm.canary import _load_yaml_config

    cfg_file = tmp_path / "test_cfg.yaml"
    cfg_file.write_text(
        "mode: real\n"
        "logical_generate_limit: 1\n"
        "provider_attempt_limit: 1\n"
        "evaluate_limit: 0\n"
        "prompt: 'Return exactly the word OK.'\n"
        "max_output_tokens: 8\n"
        "model: 'ag/yaml-model'\n"
    )

    # Test A: COMPAT > MODEL > YAML
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "ag/compat-model")
    monkeypatch.setenv("OPENAI_MODEL", "ag/openai-model")
    assert _load_yaml_config(str(cfg_file)).model == "ag/compat-model"

    # Test B: MODEL > YAML (when COMPAT absent)
    monkeypatch.delenv("OPENAI_COMPAT_MODEL")
    assert _load_yaml_config(str(cfg_file)).model == "ag/openai-model"

    # Test C: YAML (when both absent)
    monkeypatch.delenv("OPENAI_MODEL")
    assert _load_yaml_config(str(cfg_file)).model == "ag/yaml-model"


def test_real_mode_construction_with_fake_openai_client(monkeypatch):
    from unittest.mock import MagicMock

    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary
    from moh.llm.openai_client import OpenAILLMClient

    captured_sdk_kwargs = {}
    captured_create_kwargs = {}

    class FakeSDKChoice:
        def __init__(self, content="OK"):
            self.message = MagicMock(content=content)

    class FakeSDKUsage:
        def __init__(self):
            self.prompt_tokens = 10
            self.completion_tokens = 1
            self.input_tokens = 10
            self.output_tokens = 1
            self.total_tokens = 11
            self.completion_tokens_details = MagicMock(reasoning_tokens=0)

    class FakeSDKResponse:
        def __init__(self, model):
            self.choices = [FakeSDKChoice("OK")]
            self.usage = FakeSDKUsage()
            self.model = model

    class FakeSDKTransport:
        def __init__(self, **kwargs):
            captured_sdk_kwargs.update(kwargs)
            self.chat = MagicMock()
            self.chat.completions.create.side_effect = self._create

        def _create(self, **kwargs):
            captured_create_kwargs.update(kwargs)
            return FakeSDKResponse(kwargs.get("model", "unknown"))

    monkeypatch.setattr("openai.OpenAI", FakeSDKTransport)
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-local-proxy-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "ag/gemini-3.6-flash-low")

    cfg = _load_yaml_config("configs/api_canary_real.yaml")
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=cfg.provider_attempt_limit))
    usage_accountant = ProviderUsageAccountant()
    logical_guard = CanaryLogicalGuard(max_calls=cfg.logical_generate_limit)

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=None,
        attempt_budget=attempt_budget,
        usage_accountant=usage_accountant,
        max_output_tokens=cfg.max_output_tokens,
        api_mode="chat_completions",
    )

    res = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=usage_accountant,
        allow_real_api=True,
    )

    assert res.status == "success"
    assert captured_sdk_kwargs["api_key"] == "fake-local-proxy-key"
    assert captured_sdk_kwargs["base_url"] == "http://localhost:20128/v1"
    assert captured_sdk_kwargs["max_retries"] == 0
    assert captured_create_kwargs["model"] == "ag/gemini-3.6-flash-low"
    assert captured_create_kwargs["max_completion_tokens"] == 8
    assert res.logical_generate_requests == 1
    assert res.provider_attempts == 1


def test_offline_mode_does_not_instantiate_openai(monkeypatch):
    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary
    from moh.llm.openai_client import OpenAILLMClient

    def error_factory(*args, **kwargs):
        raise RuntimeError("openai.OpenAI should not be called in offline mode")

    monkeypatch.setattr("openai.OpenAI", error_factory)

    cfg = _load_yaml_config("configs/api_canary_offline.yaml")
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    usage_accountant = ProviderUsageAccountant()
    logical_guard = CanaryLogicalGuard(max_calls=1)

    class FakeOfflineTransport:
        def __init__(self):
            from unittest.mock import MagicMock
            self.chat = MagicMock()
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="OK"))]
            resp.usage = MagicMock(prompt_tokens=5, completion_tokens=1, input_tokens=5, output_tokens=1, total_tokens=6, completion_tokens_details=MagicMock(reasoning_tokens=0))
            resp.model = cfg.model
            self.chat.completions.create.return_value = resp

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=FakeOfflineTransport(),
        attempt_budget=attempt_budget,
        usage_accountant=usage_accountant,
        max_output_tokens=cfg.max_output_tokens,
    )

    res = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=logical_guard,
        attempt_budget=attempt_budget,
        usage_accountant=usage_accountant,
        allow_real_api=False,
    )
    assert res.status == "success"


def test_double_opt_in_regression_cases(monkeypatch):
    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary

    called = []
    def fake_openai(*args, **kwargs):
        called.append(kwargs)
        raise RuntimeError("Fake SDK instantiated")

    monkeypatch.setattr("openai.OpenAI", fake_openai)
    cfg = _load_yaml_config("configs/api_canary_real.yaml")

    # Case A: Real mode without --allow-real-api
    res_a = run_llm_canary(
        config=cfg,
        llm_client=None,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1)),
        usage_accountant=ProviderUsageAccountant(),
        allow_real_api=False,
    )
    assert res_a.status == "failed"
    assert res_a.error == "real_api_not_authorized"
    assert len(called) == 0

    # Case B: Real mode with allow flag but missing credential
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    res_b = run_llm_canary(
        config=cfg,
        llm_client=None,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1)),
        usage_accountant=ProviderUsageAccountant(),
        allow_real_api=True,
    )
    assert res_b.status == "failed"
    assert res_b.error == "missing_provider_credential"
    assert len(called) == 0



def test_real_canary_legacy_key_only_fails_closed(monkeypatch):
    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary

    called = []

    def fake_openai(*args, **kwargs):
        called.append(kwargs)
        raise RuntimeError("Fake SDK instantiated")

    monkeypatch.setattr("openai.OpenAI", fake_openai)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-only-secret")

    cfg = _load_yaml_config("configs/api_canary_real.yaml")
    res = run_llm_canary(
        config=cfg,
        llm_client=None,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1)),
        usage_accountant=ProviderUsageAccountant(),
        allow_real_api=True,
    )

    assert res.status == "failed"
    assert res.error == "missing_provider_credential"
    assert res.logical_generate_requests == 0
    assert res.provider_attempts == 0
    assert len(called) == 0


def test_offline_mode_does_not_set_legacy_key(monkeypatch):
    import os

    from moh.llm.canary import main

    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["canary", "--config", "configs/api_canary_offline.yaml"])

    main()

    assert "OPENAI_COMPAT_API_KEY" in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_secret_redaction_with_compat_key(monkeypatch):
    from unittest.mock import MagicMock

    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary
    from moh.llm.openai_client import OpenAILLMClient, redact_credentials

    compat_secret = "TEST_COMPAT_SECRET_MUST_NOT_LEAK"
    legacy_secret = "LEGACY_SECRET_MUST_NOT_APPEAR"
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", compat_secret)
    monkeypatch.setenv("OPENAI_API_KEY", legacy_secret)

    assert redact_credentials(f"Error with {compat_secret}") == "Error with [REDACTED]"

    cfg = _load_yaml_config("configs/api_canary_real.yaml")

    class FakeOfflineTransport:
        def __init__(self):
            self.chat = MagicMock()
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content=f"OK {compat_secret}"))]
            resp.usage = MagicMock(prompt_tokens=5, completion_tokens=1, input_tokens=5, output_tokens=1, total_tokens=6, completion_tokens_details=MagicMock(reasoning_tokens=0))
            resp.model = cfg.model
            self.chat.completions.create.return_value = resp

    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=FakeOfflineTransport(),
        attempt_budget=ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1)),
        usage_accountant=ProviderUsageAccountant(),
        max_output_tokens=cfg.max_output_tokens,
    )

    res = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1)),
        usage_accountant=ProviderUsageAccountant(),
        allow_real_api=True,
    )

    r_repr = repr(res)
    r_dict = str(res.to_dict())
    assert compat_secret not in r_repr
    assert compat_secret not in r_dict
    assert legacy_secret not in r_repr
    assert legacy_secret not in r_dict


# ---------------------------------------------------------
# M2E2B-TC: Configurable Timeout Tests
# ---------------------------------------------------------
@pytest.mark.parametrize("valid_timeout", [0.1, 1, 5.0, 30, 30.0])
def test_canary_config_valid_timeout_seconds(valid_timeout):
    cfg = make_valid_config(timeout_seconds=valid_timeout)
    assert cfg.timeout_seconds == float(valid_timeout)
    assert isinstance(cfg.timeout_seconds, float)


@pytest.mark.parametrize(
    "invalid_timeout",
    [
        0,
        -1,
        -0.1,
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "30",
    ],
)
def test_canary_config_invalid_timeout_seconds_rejected(invalid_timeout):
    with pytest.raises(ValueError, match="timeout_seconds must be positive and finite"):
        make_valid_config(timeout_seconds=invalid_timeout)


def test_load_yaml_config_real_yaml():
    from moh.llm.canary import _load_yaml_config

    cfg = _load_yaml_config("configs/api_canary_real.yaml")
    assert cfg.timeout_seconds == 30.0
    assert cfg.mode == "real"
    assert cfg.logical_generate_limit == 1
    assert cfg.provider_attempt_limit == 1
    assert cfg.evaluate_limit == 0
    assert cfg.max_output_tokens == 8
    assert cfg.model == "ag/gemini-3.6-flash-low"


def test_load_yaml_config_offline_yaml():
    from moh.llm.canary import _load_yaml_config

    cfg = _load_yaml_config("configs/api_canary_offline.yaml")
    assert cfg.timeout_seconds == 5.0
    assert cfg.mode == "offline"
    assert cfg.logical_generate_limit == 1
    assert cfg.provider_attempt_limit == 1
    assert cfg.evaluate_limit == 0
    assert cfg.max_output_tokens == 8
    assert cfg.model == "test-model"


def test_main_uses_configured_timeout(monkeypatch):
    captured_kwargs = {}
    orig_init = OpenAILLMClient.__init__

    def dummy_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        if len(args) >= 2:
            captured_kwargs["timeout_seconds"] = args[1]
        elif "timeout_seconds" in kwargs:
            captured_kwargs["timeout_seconds"] = kwargs["timeout_seconds"]
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(OpenAILLMClient, "__init__", dummy_init)
    monkeypatch.setattr(
        "sys.argv",
        ["canary", "--config", "configs/api_canary_offline.yaml"],
    )

    from moh.llm.canary import main

    main()
    assert captured_kwargs.get("timeout_seconds") == 5.0


# ---------------------------------------------------------
# M2E2B-MO: Canary Propagation Tests
# ---------------------------------------------------------

def test_canary_propagation_of_malformed_response_reason():
    from types import SimpleNamespace

    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary
    from moh.llm.openai_client import OpenAILLMClient

    class Transport:
        def __init__(self, resp):
            self.resp = resp
            self.responses = self
        def create(self, **kwargs):
            return self.resp

    cfg = _load_yaml_config("configs/api_canary_offline.yaml")
    resp = SimpleNamespace(
        output_text="   ",  # empty_text
        choices=[SimpleNamespace(message=SimpleNamespace(content="   "))],
        model=cfg.model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12, reasoning_tokens=0),
        status="completed",
    )
    attempt_budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=Transport(resp),
        attempt_budget=attempt_budget,
    )

    res = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=attempt_budget,
        usage_accountant=ProviderUsageAccountant(),
        allow_real_api=False,
    )

    assert res.status == "failed"
    assert res.error == "malformed_response"
    assert res.malformed_response_reason == "empty_text"
    assert res.last_provider_error is None
    assert res.logical_generate_requests == 1
    assert res.provider_attempts == 1


def test_canary_success_and_provider_failure_reasons_are_none():
    from types import SimpleNamespace

    import httpx2 as httpx
    import openai

    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary
    from moh.llm.openai_client import OpenAILLMClient

    class Transport:
        def __init__(self, value):
            self.value = value
            self.responses = self
        def create(self, **kwargs):
            if isinstance(self.value, Exception):
                raise self.value
            return self.value

    cfg = _load_yaml_config("configs/api_canary_offline.yaml")
    
    # Success case
    resp_success = SimpleNamespace(
        output_text="OK",
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        model=cfg.model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12, reasoning_tokens=0),
        status="completed",
    )
    b_ok = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    client_ok = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=Transport(resp_success),
        attempt_budget=b_ok,
    )
    res_ok = run_llm_canary(
        config=cfg,
        llm_client=client_ok,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=b_ok,
        usage_accountant=ProviderUsageAccountant(),
        allow_real_api=False,
    )
    assert res_ok.status == "success"
    assert res_ok.error is None
    assert res_ok.last_provider_error is None
    assert res_ok.malformed_response_reason is None

    # Timeout case
    err_timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    b_tout = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    client_timeout = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=Transport(err_timeout),
        attempt_budget=b_tout,
    )
    res_timeout = run_llm_canary(
        config=cfg,
        llm_client=client_timeout,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=b_tout,
        usage_accountant=ProviderUsageAccountant(),
        allow_real_api=False,
    )
    assert res_timeout.status == "failed"
    assert res_timeout.error == "provider_attempt_budget_exhausted"
    assert res_timeout.last_provider_error == "provider_timeout"
    assert res_timeout.malformed_response_reason is None



# ---------------------------------------------------------
# M2E2B-UN: Canary Token Normalization Integration Tests
# ---------------------------------------------------------

def test_canary_r3_9router_separate_thinking_succeeds():
    from types import SimpleNamespace

    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary
    from moh.llm.openai_client import OpenAILLMClient

    class Transport:
        def __init__(self, resp):
            self.resp = resp
            self.responses = self
        def create(self, **kwargs):
            return self.resp

    cfg = _load_yaml_config("configs/api_canary_offline.yaml")
    resp = SimpleNamespace(
        output_text="OK",
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        model=cfg.model,
        usage=SimpleNamespace(prompt_tokens=21, completion_tokens=2, total_tokens=29, reasoning_tokens=6),
        status="completed",
    )
    b = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    acc = ProviderUsageAccountant()
    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=Transport(resp),
        attempt_budget=b,
        usage_accountant=acc,
    )

    res = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=b,
        usage_accountant=acc,
        allow_real_api=False,
    )

    assert res.status == "success"
    assert res.text == "OK"
    assert res.input_tokens == 21
    assert res.output_tokens == 8
    assert res.reasoning_tokens == 6
    assert res.total_tokens == 29
    assert res.error is None
    assert res.last_provider_error is None
    assert res.malformed_response_reason is None


def test_canary_native_openai_usage_remains_canonical():
    from types import SimpleNamespace

    from moh.llm.budget import (
        ProviderAttemptBudget,
        ProviderAttemptLimits,
        ProviderUsageAccountant,
    )
    from moh.llm.canary import CanaryLogicalGuard, _load_yaml_config, run_llm_canary
    from moh.llm.openai_client import OpenAILLMClient

    class Transport:
        def __init__(self, resp):
            self.resp = resp
            self.responses = self
        def create(self, **kwargs):
            return self.resp

    cfg = _load_yaml_config("configs/api_canary_offline.yaml")
    resp = SimpleNamespace(
        output_text="OK",
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        model=cfg.model,
        usage=SimpleNamespace(prompt_tokens=21, completion_tokens=8, total_tokens=29, reasoning_tokens=6),
        status="completed",
    )
    b = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=1))
    acc = ProviderUsageAccountant()
    client = OpenAILLMClient(
        model=cfg.model,
        timeout_seconds=5.0,
        observer=lambda _: None,
        transport=Transport(resp),
        attempt_budget=b,
        usage_accountant=acc,
    )

    res = run_llm_canary(
        config=cfg,
        llm_client=client,
        logical_guard=CanaryLogicalGuard(max_calls=1),
        attempt_budget=b,
        usage_accountant=acc,
        allow_real_api=False,
    )

    assert res.status == "success"
    assert res.input_tokens == 21
    assert res.output_tokens == 8
    assert res.reasoning_tokens == 6
    assert res.total_tokens == 29
