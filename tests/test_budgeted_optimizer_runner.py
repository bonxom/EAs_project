from moh.core.models import OptimizerProgram
from moh.optimizers.capabilities import (
    CapabilityLimits,
    CapabilityUsage,
    run_optimizer_with_capabilities,
)


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or ["default text"])
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        item = self.responses.pop(0) if self.responses else "generated-item"
        if isinstance(item, Exception):
            raise item
        return item


class FakeEvaluator:
    def __init__(self, scores=None):
        self.scores = list(scores or [10.0])
        self.calls = 0

    def __call__(self, source_code: str) -> float:
        self.calls += 1
        val = self.scores.pop(0) if self.scores else 1.0
        if isinstance(val, Exception):
            raise val
        return val


def test_e2e_worker_controller_success():
    code = """
def improve_algorithm(api):
    a = api.generate("one")
    sa = api.evaluate(a)

    b = api.generate("two")
    sb = api.evaluate(b)

    return {
        "a": a,
        "sa": sa,
        "b": b,
        "sb": sb,
    }
"""
    prog = OptimizerProgram(id="opt-e2e-1", source_code=code)
    llm = FakeLLM(["heuristic-a", "heuristic-b"])
    evaluator = FakeEvaluator([12.5, 9.0])
    cap_limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)

    res, usage = run_optimizer_with_capabilities(prog, llm, evaluator, cap_limits)

    assert res["status"] == "success"
    assert res["result"] == {
        "a": "heuristic-a",
        "sa": 12.5,
        "b": "heuristic-b",
        "sb": 9.0,
    }
    assert usage == CapabilityUsage(generate_requests=2, evaluate_requests=2)
    assert llm.calls == 2
    assert evaluator.calls == 2


def test_e2e_generation_exhaustion():
    code = """
def improve_algorithm(api):
    a = api.generate("one")
    b = api.generate("two")
    c = api.generate("three")
    return [a, b, c]
"""
    prog = OptimizerProgram(id="opt-e2e-2", source_code=code)
    llm = FakeLLM(["gen-1", "gen-2", "gen-3"])
    evaluator = FakeEvaluator()
    cap_limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=5)

    res, usage = run_optimizer_with_capabilities(prog, llm, evaluator, cap_limits)

    assert res["status"] == "failed"
    assert res["code"] == "capability_error"
    assert "generation_budget_exhausted" in res["message"] or "parent error" in res["message"]
    assert usage == CapabilityUsage(generate_requests=2, evaluate_requests=0)
    assert llm.calls == 2


def test_e2e_evaluation_exhaustion():
    code = """
def improve_algorithm(api):
    e1 = api.evaluate("c1")
    e2 = api.evaluate("c2")
    e3 = api.evaluate("c3")
    return [e1, e2, e3]
"""
    prog = OptimizerProgram(id="opt-e2e-3", source_code=code)
    llm = FakeLLM()
    evaluator = FakeEvaluator([1.0, 2.0, 3.0])
    cap_limits = CapabilityLimits(max_generate_requests=5, max_evaluate_requests=2)

    res, usage = run_optimizer_with_capabilities(prog, llm, evaluator, cap_limits)

    assert res["status"] == "failed"
    assert res["code"] == "capability_error"
    assert "evaluation_budget_exhausted" in res["message"] or "parent error" in res["message"]
    assert usage == CapabilityUsage(generate_requests=0, evaluate_requests=2)
    assert evaluator.calls == 2


def test_e2e_zero_limit_generation():
    code = """
def improve_algorithm(api):
    return api.generate("one")
"""
    prog = OptimizerProgram(id="opt-e2e-4", source_code=code)
    llm = FakeLLM(["gen-1"])
    evaluator = FakeEvaluator()
    cap_limits = CapabilityLimits(max_generate_requests=0, max_evaluate_requests=2)

    res, usage = run_optimizer_with_capabilities(prog, llm, evaluator, cap_limits)

    assert res["status"] == "failed"
    assert res["code"] == "capability_error"
    assert usage == CapabilityUsage(generate_requests=0, evaluate_requests=0)
    assert llm.calls == 0


def test_e2e_zero_limit_evaluation():
    code = """
def improve_algorithm(api):
    return api.evaluate("c1")
"""
    prog = OptimizerProgram(id="opt-e2e-5", source_code=code)
    llm = FakeLLM()
    evaluator = FakeEvaluator([10.0])
    cap_limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=0)

    res, usage = run_optimizer_with_capabilities(prog, llm, evaluator, cap_limits)

    assert res["status"] == "failed"
    assert res["code"] == "capability_error"
    assert usage == CapabilityUsage(generate_requests=0, evaluate_requests=0)
    assert evaluator.calls == 0


def test_e2e_trusted_parent_exception_containment():
    code = """
def improve_algorithm(api):
    return api.generate("one")
"""
    prog = OptimizerProgram(id="opt-e2e-6", source_code=code)
    llm = FakeLLM([RuntimeError("LLM server crash")])
    evaluator = FakeEvaluator()
    cap_limits = CapabilityLimits(max_generate_requests=2, max_evaluate_requests=2)

    res, usage = run_optimizer_with_capabilities(prog, llm, evaluator, cap_limits)

    assert res["status"] == "failed"
    assert res["code"] == "capability_error"
    assert "generation_failed" in res["message"] or "parent error" in res["message"]
    assert usage == CapabilityUsage(generate_requests=1, evaluate_requests=0)
    assert llm.calls == 1
