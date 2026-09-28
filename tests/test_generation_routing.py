import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from moh.experiments.runner import load_experiment_config, run_experiment
from moh.llm.base import GenerationError
from moh.llm.budget import ProviderUsageAccountant
from moh.llm.openai_client import OpenAILLMClient
from moh.optimizers.capabilities import CapabilityLimits, OptimizerCapabilityController
from moh.optimizers.evaluator import FakeEvaluator, InnerSearchTracker
from moh.optimizers.programs import parse_optimizer_program
from moh.optimizers.runner import OptimizerProgramRunner
from moh.real_pilot import (
    TSP_CANDIDATE_CONTRACT_PROMPT,
    TaskCandidateLLMAdapter,
    parse_generation_kind,
)


class FakeMockTransport:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.call_history = []
        self.chat = MagicMock()
        self.chat.completions.create = self.create

    def create(self, **kwargs):
        self.call_history.append(kwargs)
        if not self.responses:
            res_text = "```python\ndef select_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]\n```"
        else:
            res_text = self.responses.pop(0)

        if isinstance(res_text, Exception):
            raise res_text

        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = res_text
        mock_resp.choices = [mock_choice]
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.reasoning_tokens = 0
        mock_usage.total_tokens = 150
        mock_usage.completion_tokens_details = MagicMock()
        mock_usage.completion_tokens_details.reasoning_tokens = 0
        mock_resp.usage = mock_usage
        mock_resp.model = "ag/gemini-3.6-flash-low"
        return mock_resp


def test_parse_generation_kind():
    assert parse_generation_kind("KIND: reflection\nAnalyze performance.") == "reflection"
    assert parse_generation_kind("KIND: mutate\nImprove solution.") == "mutate"
    assert parse_generation_kind("KIND: crossover\nCombine parents.") == "crossover"

    with pytest.raises(GenerationError, match="missing_prompt_kind"):
        parse_generation_kind("no kind header")

    with pytest.raises(GenerationError, match="unsupported_generation_kind"):
        parse_generation_kind("KIND: unsupported_kind\nTest")


def test_reflection_plus_candidate_plus_evaluate(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    transport = FakeMockTransport([
        "The current solution is greedy. Try distant nodes first.", # Reflection prose
        "def select_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]\n" # Mutate candidate
    ])

    client = OpenAILLMClient(
        model="ag/gemini-3.6-flash-low",
        timeout_seconds=30.0,
        observer=lambda meta: None,
        transport=transport,
        api_mode="chat_completions",
        max_attempts_per_request=1,
    )
    adapter = TaskCandidateLLMAdapter(client, TSP_CANDIDATE_CONTRACT_PROMPT)

    prog = parse_optimizer_program(
        "def improve_algorithm(api):\n"
        "    idea = api.generate('KIND: reflection\\nAnalyze performance.')\n"
        "    code = api.generate('KIND: mutate\\nImprove solution.')\n"
        "    return api.evaluate(code)\n",
        "o000001"
    )

    evaluator = FakeEvaluator(default_score=42.0)
    tracker = InnerSearchTracker(evaluator)
    limits = CapabilityLimits(max_generate_requests=5, max_evaluate_requests=5)
    controller = OptimizerCapabilityController(adapter, tracker.evaluate_candidate, limits)
    runner = OptimizerProgramRunner()

    res = runner.run(prog, controller.handle)
    assert res["status"] == "success"
    assert res["result"] == 42.0
    assert controller.usage.generate_requests == 2
    assert controller.usage.evaluate_requests == 1
    assert len(transport.call_history) == 2


def test_reflection_must_not_be_candidate_validated(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    transport = FakeMockTransport(["The previous strategy is too greedy..."])
    client = OpenAILLMClient(
        model="ag/gemini-3.6-flash-low",
        timeout_seconds=30.0,
        observer=lambda meta: None,
        transport=transport,
        api_mode="chat_completions",
    )
    adapter = TaskCandidateLLMAdapter(client, TSP_CANDIDATE_CONTRACT_PROMPT)

    # Calling reflection returns text prose without syntax error
    result_text = adapter.generate("KIND: reflection\nAnalyze performance.")
    assert result_text == "The previous strategy is too greedy..."
    assert TSP_CANDIDATE_CONTRACT_PROMPT not in transport.call_history[0]["messages"][0]["content"]


def test_mutate_uses_tsp_contract(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    transport = FakeMockTransport(["def select_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]\n"])
    client = OpenAILLMClient(
        model="ag/gemini-3.6-flash-low",
        timeout_seconds=30.0,
        observer=lambda meta: None,
        transport=transport,
        api_mode="chat_completions",
    )
    adapter = TaskCandidateLLMAdapter(client, TSP_CANDIDATE_CONTRACT_PROMPT)

    adapter.generate("KIND: mutate\nOption 1.")
    sent_content = transport.call_history[0]["messages"][0]["content"]
    assert TSP_CANDIDATE_CONTRACT_PROMPT in sent_content
    assert "[OPTIMIZER INSTRUCTION]\nKIND: mutate\nOption 1." in sent_content


def test_unknown_kind_rejected_locally(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    transport = FakeMockTransport()
    client = OpenAILLMClient(
        model="ag/gemini-3.6-flash-low",
        timeout_seconds=30.0,
        observer=lambda meta: None,
        transport=transport,
        api_mode="chat_completions",
    )
    adapter = TaskCandidateLLMAdapter(client, TSP_CANDIDATE_CONTRACT_PROMPT)

    with pytest.raises(GenerationError, match="unsupported_generation_kind"):
        adapter.generate("KIND: unknown_kind\nTest")

    # 0 provider attempts made
    assert len(transport.call_history) == 0


def test_token_aggregation(monkeypatch):
    accountant = ProviderUsageAccountant()
    accountant.record(model="test", input_tokens=100, output_tokens=50, reasoning_tokens=10)
    accountant.record(model="test", input_tokens=200, output_tokens=80, reasoning_tokens=20)

    u = accountant.usage
    assert u.input_tokens == 300
    assert u.output_tokens == 130
    assert u.reasoning_tokens == 30
    assert u.total_tokens == 430
    assert u.reasoning_tokens <= u.output_tokens
    assert u.total_tokens == u.input_tokens + u.output_tokens


def test_single_replicate_offline_control(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    raw_cfg = load_experiment_config("configs/full_moh_sanity.yaml")
    cfg = replace(raw_cfg, replicate_index=0, output_dir=tmp_path / "single_replicate_test")

    # Transport returning reflection text for 1st call, and candidate code for 2nd call
    def fake_transport_factory():
        return FakeMockTransport([
            "Reflection analysis text.",
            "def select_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]\n",
            "Option 2 reflection.",
            "def select_next_node(current_node, unvisited, coordinates):\n    return min(unvisited)\n",
        ])

    manifest = run_experiment(
        cfg,
        allow_real_api=True,
        transport_factory=fake_transport_factory,
    )

    assert manifest["status"] == "COMPLETED_VALID"
    assert manifest["best_utility"] is not None
    assert manifest["work_counts"]["inner_generate_requests"] > 0
    assert manifest["work_counts"]["inner_evaluate_requests"] > 0
    manifest_data = json.loads(Path(manifest["manifest_path"]).read_text())
    assert manifest_data["token_counts"]["combined_total_tokens"] > 0
