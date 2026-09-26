import pytest

from moh.optimizers.capabilities import (
    CapabilityLimits,
    CapabilityUsage,
    OptimizerCapabilityController,
)


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or ["default text"])
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if isinstance(self.responses[0], Exception):
            raise self.responses.pop(0)
        return self.responses.pop(0)


class FakeEvaluator:
    def __init__(self, scores=None):
        self.scores = list(scores or [10.0])
        self.calls = 0

    def __call__(self, source_code: str) -> float:
        self.calls += 1
        val = self.scores.pop(0)
        if isinstance(val, Exception):
            raise val
        return val


def test_valid_capability_limits():
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=5)
    assert limits.max_generate_requests == 2
    assert limits.max_evaluate_requests == 5


@pytest.mark.parametrize("val", [-1, -10])
def test_negative_limits_rejected(val):
    with pytest.raises(ValueError):
        CapabilityLimits(max_generate_requests=val, max_evaluate_requests=1)
    with pytest.raises(ValueError):
        CapabilityLimits(max_generate_requests=1, max_evaluate_requests=val)


@pytest.mark.parametrize("val", [True, False, 1.5, "2"])
def test_invalid_type_limits_rejected(val):
    with pytest.raises(TypeError):
        CapabilityLimits(max_generate_requests=val, max_evaluate_requests=1)
    with pytest.raises(TypeError):
        CapabilityLimits(max_generate_requests=1, max_evaluate_requests=val)


def test_valid_generation():
    llm = FakeLLM(["gen1"])
    evaluator = FakeEvaluator()
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    req = {"type": "generate", "request_id": "req-001", "prompt": "make heuristic"}
    resp = controller.handle(req)

    assert resp == {"type": "generate_result", "request_id": "req-001", "text": "gen1"}
    assert llm.calls == 1
    assert controller.usage == CapabilityUsage(generate_requests=1, evaluate_requests=0)


def test_valid_evaluation():
    llm = FakeLLM()
    evaluator = FakeEvaluator([42.5])
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    req = {"type": "evaluate", "request_id": "req-002", "source_code": "def solve(): pass"}
    resp = controller.handle(req)

    assert resp == {"type": "evaluate_result", "request_id": "req-002", "score": 42.5}
    assert evaluator.calls == 1
    assert controller.usage == CapabilityUsage(generate_requests=0, evaluate_requests=1)


def test_zero_generation_budget():
    llm = FakeLLM(["gen1"])
    evaluator = FakeEvaluator()
    limits = CapabilityLimits(max_generate_requests=0, max_evaluate_requests=2)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    req = {"type": "generate", "request_id": "req-001", "prompt": "make heuristic"}
    resp = controller.handle(req)

    assert resp["type"] == "error"
    assert resp["code"] == "generation_budget_exhausted"
    assert llm.calls == 0
    assert controller.usage.generate_requests == 0


def test_zero_evaluation_budget():
    llm = FakeLLM()
    evaluator = FakeEvaluator([10.0])
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=0)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    req = {"type": "evaluate", "request_id": "req-001", "source_code": "code"}
    resp = controller.handle(req)

    assert resp["type"] == "error"
    assert resp["code"] == "evaluation_budget_exhausted"
    assert evaluator.calls == 0
    assert controller.usage.evaluate_requests == 0


def test_exact_generation_limit():
    llm = FakeLLM(["g1", "g2", "g3"])
    evaluator = FakeEvaluator()
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    resp1 = controller.handle({"type": "generate", "request_id": "req-1", "prompt": "p1"})
    resp2 = controller.handle({"type": "generate", "request_id": "req-2", "prompt": "p2"})
    resp3 = controller.handle({"type": "generate", "request_id": "req-3", "prompt": "p3"})

    assert resp1["type"] == "generate_result"
    assert resp2["type"] == "generate_result"
    assert resp3["type"] == "error"
    assert resp3["code"] == "generation_budget_exhausted"
    assert llm.calls == 2
    assert controller.usage.generate_requests == 2


def test_exact_evaluation_limit():
    llm = FakeLLM()
    evaluator = FakeEvaluator([1.0, 2.0, 3.0])
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    resp1 = controller.handle({"type": "evaluate", "request_id": "req-1", "source_code": "c1"})
    resp2 = controller.handle({"type": "evaluate", "request_id": "req-2", "source_code": "c2"})
    resp3 = controller.handle({"type": "evaluate", "request_id": "req-3", "source_code": "c3"})

    assert resp1["type"] == "evaluate_result"
    assert resp2["type"] == "evaluate_result"
    assert resp3["type"] == "error"
    assert resp3["code"] == "evaluation_budget_exhausted"
    assert evaluator.calls == 2
    assert controller.usage.evaluate_requests == 2


def test_generation_failure_consumes_budget():
    llm = FakeLLM([RuntimeError("llm error"), "g2"])
    evaluator = FakeEvaluator()
    limits = CapabilityLimits(max_generate_requests=1, max_evaluate_requests=2)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    resp1 = controller.handle({"type": "generate", "request_id": "req-1", "prompt": "p1"})
    resp2 = controller.handle({"type": "generate", "request_id": "req-2", "prompt": "p2"})

    assert resp1["type"] == "error"
    assert resp1["code"] == "generation_failed"
    assert resp2["type"] == "error"
    assert resp2["code"] == "generation_budget_exhausted"
    assert llm.calls == 1
    assert controller.usage.generate_requests == 1


def test_evaluation_failure_consumes_budget():
    llm = FakeLLM()
    evaluator = FakeEvaluator([ValueError("eval error"), 10.0])
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=1)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    resp1 = controller.handle({"type": "evaluate", "request_id": "req-1", "source_code": "c1"})
    resp2 = controller.handle({"type": "evaluate", "request_id": "req-2", "source_code": "c2"})

    assert resp1["type"] == "error"
    assert resp1["code"] == "evaluation_failed"
    assert resp2["type"] == "error"
    assert resp2["code"] == "evaluation_budget_exhausted"
    assert evaluator.calls == 1
    assert controller.usage.evaluate_requests == 1


@pytest.mark.parametrize("invalid_val", [None, 123, 45.6, object(), True])
def test_invalid_llm_returns(invalid_val):
    llm = FakeLLM([invalid_val])
    evaluator = FakeEvaluator()
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    resp = controller.handle({"type": "generate", "request_id": "req-1", "prompt": "p1"})
    assert resp["type"] == "error"
    assert resp["code"] == "generation_failed"


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), float("-inf"), True, False, "10.0", None, object()],
)
def test_invalid_evaluator_returns(invalid_score):
    llm = FakeLLM()
    evaluator = FakeEvaluator([invalid_score])
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    resp = controller.handle({"type": "evaluate", "request_id": "req-1", "source_code": "c1"})
    assert resp["type"] == "error"
    assert resp["code"] == "evaluation_failed"


def test_duplicate_request_id_rejected():
    llm = FakeLLM(["g1", "g2"])
    evaluator = FakeEvaluator()
    limits = CapabilityLimits(max_generate_requests=5, max_evaluate_requests=5)
    controller = OptimizerCapabilityController(llm, evaluator, limits)

    resp1 = controller.handle({"type": "generate", "request_id": "req-same", "prompt": "p1"})
    resp2 = controller.handle({"type": "generate", "request_id": "req-same", "prompt": "p2"})

    assert resp1["type"] == "generate_result"
    assert resp2["type"] == "error"
    assert resp2["code"] == "duplicate_request"
    assert llm.calls == 1  # Second request did NOT call LLM again


def test_independent_controller_counters():
    llm = FakeLLM(["g1", "g2"])
    evaluator = FakeEvaluator([1.0, 2.0])
    limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)

    c1 = OptimizerCapabilityController(llm, evaluator, limits)
    c2 = OptimizerCapabilityController(llm, evaluator, limits)

    c1.handle({"type": "generate", "request_id": "req-1", "prompt": "p1"})
    assert c1.usage == CapabilityUsage(generate_requests=1, evaluate_requests=0)
    assert c2.usage == CapabilityUsage(generate_requests=0, evaluate_requests=0)
