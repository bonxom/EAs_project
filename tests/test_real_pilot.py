"""Comprehensive unit and integration tests for Full-MoH real pilot harness (M6A)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx2 as httpx
import openai
import pytest

from moh.real_pilot import (
    EXPECTED_PILOT_MODEL,
    InnerSearchSettings,
    OuterLLMSettings,
    RealPilotConfig,
    RuntimeSettings,
    load_pilot_config,
    make_tsp_evaluator,
    run_real_pilot_harness,
    validate_real_pilot_environment,
)


class FakeUsage:
    def __init__(self, prompt_tokens=100, completion_tokens=20, reasoning_tokens=5):
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
        prompt_tokens=100,
        completion_tokens=20,
        reasoning_tokens=5,
        model=EXPECTED_PILOT_MODEL,
    ):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage(prompt_tokens, completion_tokens, reasoning_tokens)
        self.model = model
        self.output_text = content
        self.status = "completed"


class FakeTransport:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise RuntimeError("No more fake responses configured in FakeTransport")
        val = self.responses.pop(0)
        if isinstance(val, Exception):
            raise val
        return val


SAMPLE_HEURISTIC_SOURCE_1 = """def select_next_node(current_node, unvisited, coordinates):
    return min(unvisited, key=lambda j: (
        (coordinates[current_node][0] - coordinates[j][0]) ** 2
        + (coordinates[current_node][1] - coordinates[j][1]) ** 2,
        j,
    ))
"""

SAMPLE_HEURISTIC_SOURCE_2 = """def select_next_node(current_node, unvisited, coordinates):
    return sorted(unvisited)[0]
"""

SAMPLE_OFFSPRING_OPTIMIZER_SOURCE = """```python
def improve_algorithm(api):
    c = api.generate('KIND: mutate\\nGenerate offspring heuristic.')
    return api.evaluate(c)
```
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_cli_without_allow_real_api_performs_zero_network():
    cfg = RealPilotConfig()
    res = run_real_pilot_harness(cfg, allow_real_api=False)
    assert res.status == "dry_run_refused"
    assert res.reason is not None
    assert "--allow-real-api" in res.reason
    assert res.outer_provider_attempts == 0
    assert res.inner_provider_attempts == 0
    assert res.total_provider_attempts == 0
    assert res.total_tokens == 0


def test_config_validation():
    cfg = RealPilotConfig()
    assert cfg.population_size == 1
    assert cfg.generations == 1
    assert cfg.outer_llm.max_output_tokens == 2048
    assert cfg.inner_llm.max_output_tokens == 2048

    with pytest.raises(ValueError, match="population_size"):
        RealPilotConfig(population_size=2)

    with pytest.raises(ValueError, match="generations"):
        RealPilotConfig(generations=2)

    with pytest.raises(ValueError, match="max_generate_requests"):
        InnerSearchSettings(max_generate_requests=2)

    with pytest.raises(ValueError, match="max_evaluate_requests"):
        InnerSearchSettings(max_evaluate_requests=2)

    with pytest.raises(ValueError, match="max_output_tokens"):
        OuterLLMSettings(max_output_tokens=8)

    with pytest.raises(ValueError, match="max_output_tokens"):
        OuterLLMSettings(max_output_tokens=4096)

    with pytest.raises(ValueError, match="max_output_tokens"):
        OuterLLMSettings(max_output_tokens=True)  # type: ignore

    with pytest.raises(ValueError, match="provider_attempt_limit"):
        OuterLLMSettings(provider_attempt_limit=2)

    with pytest.raises(ValueError, match="timeout_seconds"):
        RuntimeSettings(timeout_seconds=60.0)

    with pytest.raises(ValueError, match="sdk_max_retries"):
        RuntimeSettings(sdk_max_retries=1)


def test_environment_validation(monkeypatch):
    cfg = RealPilotConfig()

    # Case 1: Missing API key
    with pytest.raises(ValueError, match="OPENAI_COMPAT_API_KEY"):
        validate_real_pilot_environment(cfg)

    # Case 2: Legacy OPENAI_API_KEY set
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    with pytest.raises(ValueError, match="OPENAI_API_KEY must be inactive"):
        validate_real_pilot_environment(cfg)

    # Case 3: Missing base URL
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_COMPAT_BASE_URL"):
        validate_real_pilot_environment(cfg)

    # Case 4: Valid env
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    validate_real_pilot_environment(cfg)

    # Case 5: Model mismatch
    bad_cfg = RealPilotConfig(outer_llm=OuterLLMSettings(requested_model="gpt-4"))
    with pytest.raises(ValueError, match="outer requested_model"):
        validate_real_pilot_environment(bad_cfg)


def test_load_pilot_config(tmp_path):
    config_yaml = """
population_size: 1
generations: 1
inner:
  max_generate_requests: 1
  max_evaluate_requests: 1
outer_llm:
  requested_model: "ag/gemini-3.6-flash-low"
  max_output_tokens: 2048
  provider_attempt_limit: 1
inner_llm:
  requested_model: "ag/gemini-3.6-flash-low"
  max_output_tokens: 2048
  provider_attempt_limit_per_optimizer: 1
runtime:
  timeout_seconds: 30.0
  sdk_max_retries: 0
  route_provider: "antigravity"
  proxy: "9router"
  proxy_version: "0.5.81"
  api_mode: "chat_completions"
task:
  name: "tsp10"
  size: 10
  count: 1
  seed: 42
"""
    p = tmp_path / "pilot_test.yaml"
    p.write_text(config_yaml, encoding="utf-8")

    cfg = load_pilot_config(p)
    assert cfg.population_size == 1
    assert cfg.generations == 1
    assert cfg.outer_llm.max_output_tokens == 2048
    assert cfg.inner_llm.max_output_tokens == 2048


def test_valid_e2e_pilot_with_fake_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    responses = [
        FakeResponse(SAMPLE_HEURISTIC_SOURCE_1, prompt_tokens=100, completion_tokens=50),
        FakeResponse(SAMPLE_OFFSPRING_OPTIMIZER_SOURCE, prompt_tokens=200, completion_tokens=80),
        FakeResponse(SAMPLE_HEURISTIC_SOURCE_2, prompt_tokens=120, completion_tokens=40),
    ]
    transport = FakeTransport(responses)

    cfg = RealPilotConfig()
    result = run_real_pilot_harness(
        cfg, allow_real_api=True, transport=transport
    )

    assert result.status == "success"
    assert result.outer_programs_evaluated == 2
    assert result.outer_offspring_generated == 1
    assert result.inner_generate_requests == 2
    assert result.inner_evaluate_requests == 2

    assert result.outer_provider_attempts == 1
    assert result.inner_provider_attempts == 2
    assert result.total_provider_attempts == 3

    assert result.outer_input_tokens == 200
    assert result.outer_output_tokens == 80
    assert result.inner_input_tokens == 100 + 120
    assert result.inner_output_tokens == 50 + 40

    assert result.total_input_tokens == result.outer_input_tokens + result.inner_input_tokens
    assert result.total_output_tokens == result.outer_output_tokens + result.inner_output_tokens
    assert result.total_tokens == result.outer_total_tokens + result.inner_total_tokens

    assert result.best_utility is not None
    assert result.best_program_id is not None


def test_attempt_ceiling_and_token_separation(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    responses = [
        FakeResponse(SAMPLE_HEURISTIC_SOURCE_1, prompt_tokens=50, completion_tokens=10),
        FakeResponse(SAMPLE_OFFSPRING_OPTIMIZER_SOURCE, prompt_tokens=60, completion_tokens=15),
        FakeResponse(SAMPLE_HEURISTIC_SOURCE_2, prompt_tokens=70, completion_tokens=20),
    ]
    transport = FakeTransport(responses)

    cfg = RealPilotConfig()
    result = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)

    assert result.total_provider_attempts <= 3
    assert result.outer_provider_attempts <= 1
    assert result.inner_provider_attempts <= 2


def test_transient_failure_no_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    transient_exc = openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid")
    )
    responses = [
        transient_exc,
    ]
    transport = FakeTransport(responses)

    cfg = RealPilotConfig()
    result = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)

    assert result.outer_provider_attempts <= 1
    assert result.total_provider_attempts <= 3


def test_invalid_offspring_handling(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    responses = [
        FakeResponse(SAMPLE_HEURISTIC_SOURCE_1, prompt_tokens=100, completion_tokens=50),
        FakeResponse("def invalid_code(:", prompt_tokens=200, completion_tokens=30),
    ]
    transport = FakeTransport(responses)

    cfg = RealPilotConfig()
    result = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)

    assert result.outer_offspring_generated == 0
    assert result.outer_programs_evaluated == 1
    assert result.inner_generate_requests == 1

    assert result.outer_provider_attempts == 1
    assert result.inner_provider_attempts == 1
    assert result.total_provider_attempts == 2


def test_make_tsp_evaluator():
    evaluator = make_tsp_evaluator(size=10, count=1, seed=42)
    score1 = evaluator(SAMPLE_HEURISTIC_SOURCE_1)
    score2 = evaluator(SAMPLE_HEURISTIC_SOURCE_2)

    assert isinstance(score1, float)
    assert isinstance(score2, float)

    with pytest.raises(ValueError, match="Heuristic execution failed"):
        evaluator("def bad_syntax(:")


def test_prompt_injection_and_separation(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    responses = [
        FakeResponse(SAMPLE_HEURISTIC_SOURCE_1),
        FakeResponse(SAMPLE_OFFSPRING_OPTIMIZER_SOURCE),
        FakeResponse(SAMPLE_HEURISTIC_SOURCE_2),
    ]
    transport = FakeTransport(responses)
    cfg = RealPilotConfig()
    result = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)
    assert result.status == "success"

    assert len(transport.requests) == 3
    req_inner_1 = transport.requests[0]["messages"][-1]["content"]
    req_outer = transport.requests[1]["messages"][-1]["content"]
    req_inner_2 = transport.requests[2]["messages"][-1]["content"]

    assert "select_next_node" in req_inner_1
    assert "[TASK CONTRACT]" in req_inner_1
    assert "KIND: mutate" in req_inner_1

    assert "select_next_node" in req_inner_2
    assert "[TASK CONTRACT]" in req_inner_2

    assert "improve_algorithm" in req_outer
    assert "select_next_node" not in req_outer
    assert "[TASK CONTRACT]" not in req_outer


def test_fenced_python_candidate_accepted(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    fenced_source = f"```python\n{SAMPLE_HEURISTIC_SOURCE_1}\n```"
    responses = [
        FakeResponse(fenced_source),
        FakeResponse(SAMPLE_OFFSPRING_OPTIMIZER_SOURCE),
        FakeResponse(fenced_source),
    ]
    transport = FakeTransport(responses)
    cfg = RealPilotConfig()
    res = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)
    assert res.status == "success"
    assert res.inner_generate_requests == 2
    assert res.inner_evaluate_requests == 2


def test_prose_plus_code_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    prose_source = f"Here is the requested code:\n```python\n{SAMPLE_HEURISTIC_SOURCE_1}\n```"
    responses = [
        FakeResponse(prose_source),
        FakeResponse(SAMPLE_OFFSPRING_OPTIMIZER_SOURCE),
        FakeResponse(prose_source),
    ]
    transport = FakeTransport(responses)
    cfg = RealPilotConfig()
    res = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)
    assert res.status == "failed"
    assert res.inner_generate_requests == 2
    assert res.inner_evaluate_requests == 2
    assert res.total_provider_attempts == 3


def test_wrong_function_and_signature_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    wrong_fn = "def choose_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]\n"
    wrong_sig = "def select_next_node(current_node):\n    return current_node\n"
    responses = [
        FakeResponse(wrong_fn),
        FakeResponse(SAMPLE_OFFSPRING_OPTIMIZER_SOURCE),
        FakeResponse(wrong_sig),
    ]
    transport = FakeTransport(responses)
    cfg = RealPilotConfig()
    res = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)
    assert res.status == "failed"
    assert res.inner_generate_requests == 2
    assert res.inner_evaluate_requests == 2
    assert res.total_provider_attempts == 3


def test_syntax_error_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    bad_syntax = "def select_next_node(:"
    responses = [
        FakeResponse(bad_syntax),
        FakeResponse(SAMPLE_OFFSPRING_OPTIMIZER_SOURCE),
        FakeResponse(bad_syntax),
    ]
    transport = FakeTransport(responses)
    cfg = RealPilotConfig()
    res = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)
    assert res.status == "failed"
    assert res.inner_generate_requests == 2
    assert res.inner_evaluate_requests == 2
    assert res.total_provider_attempts == 3


def test_m6b_topology_accounting_regression(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "fake-key")
    bad_cand_1 = "def select_next_node(current_node, unvisited, coordinates):\n    return 'invalid_type'\n"
    bad_cand_2 = "def select_next_node(current_node, unvisited, coordinates):\n    raise RuntimeError()\n"
    responses = [
        FakeResponse(bad_cand_1),
        FakeResponse(SAMPLE_OFFSPRING_OPTIMIZER_SOURCE),
        FakeResponse(bad_cand_2),
    ]
    transport = FakeTransport(responses)
    cfg = RealPilotConfig()
    res = run_real_pilot_harness(cfg, allow_real_api=True, transport=transport)

    assert res.status == "failed"
    assert res.best_utility is None
    assert res.outer_provider_attempts == 1
    assert res.inner_provider_attempts == 2
    assert res.total_provider_attempts == 3

    assert res.inner_generate_requests == 2
    assert res.inner_evaluate_requests == 2

