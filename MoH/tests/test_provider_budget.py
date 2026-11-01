"""Offline tests for shared ProviderBudget across heuristic/meta roles."""

from types import SimpleNamespace

from utils.llm_client.base import BaseClient, ProviderBudget


class FlakyLLM(BaseClient):
    def __init__(self, *args, fail_times=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_times = fail_times
        self.api_calls = 0

    def _chat_completion_api(self, messages, temperature, n=1):
        self.api_calls += 1
        if self.api_calls <= self.fail_times:
            raise RuntimeError("transient")
        content = messages[-1]["content"]
        return [SimpleNamespace(message=SimpleNamespace(content=content))]


def test_shared_budget_limits_requests_across_clients(monkeypatch):
    monkeypatch.setattr("utils.llm_client.base.time.sleep", lambda _: None)
    budget = ProviderBudget(max_requests=2, max_attempts=100)
    a = FlakyLLM("fake", budget=budget, max_retries=1)
    b = FlakyLLM("fake", budget=budget, max_retries=1)
    assert a.prompt("e", "one") == "one"
    assert b.prompt("e", "two") == "two"
    assert a.prompt("e", "three") == "Error: provider request budget exhausted"
    assert budget.snapshot()["requests"] == 2
    assert budget.exhausted_reason == "max_requests=2"


def test_shared_budget_limits_attempts_during_retries(monkeypatch):
    monkeypatch.setattr("utils.llm_client.base.time.sleep", lambda _: None)
    budget = ProviderBudget(max_requests=10, max_attempts=2)
    llm = FlakyLLM("fake", budget=budget, max_retries=5, fail_times=5)
    result = llm.prompt("e", "hello")
    assert result == "Error: provider attempt budget exhausted"
    assert llm.api_calls == 2
    assert budget.snapshot()["attempts"] == 2
    assert budget.exhausted_reason == "max_attempts=2"


def test_default_unlimited_budget_is_none():
    llm = FlakyLLM("fake")
    assert llm.budget is None
    assert llm.max_retries == 10


def test_budget_snapshot_is_shared_view(monkeypatch):
    monkeypatch.setattr("utils.llm_client.base.time.sleep", lambda _: None)
    budget = ProviderBudget(max_requests=5, max_attempts=10)
    a = FlakyLLM("fake", budget=budget)
    b = FlakyLLM("fake", budget=budget)
    a.prompt("e", "x")
    b.prompt("e", "y")
    snap = budget.snapshot()
    assert snap["requests"] == 2
    assert snap["attempts"] == 2
    assert snap["max_requests"] == 5
