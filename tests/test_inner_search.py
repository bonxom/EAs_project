"""End-to-end integration tests for M3 inner search execution semantics."""

from moh.core.models import OptimizerProgram
from moh.optimizers.capabilities import CapabilityLimits
from moh.optimizers.evaluator import FakeEvaluator, run_optimizer_inner_search


class MockLLM:
    """Simple deterministic LLM mock for capability controller tests."""

    def __init__(self, responses=None):
        self.responses = list(responses or ["default_response"])

    def generate(self, prompt: str) -> str:
        if self.responses:
            val = self.responses.pop(0)
            if isinstance(val, Exception):
                raise val
            return val
        return "default_response"


OPTIMIZER_TWO_CANDIDATES = OptimizerProgram(
    id="opt-two-candidates",
    source_code="""
def improve_algorithm(api):
    c1 = api.generate("prompt 1")
    s1 = api.evaluate("candidate_code_1")
    c2 = api.generate("prompt 2")
    s2 = api.evaluate("candidate_code_2")
    return f"scores: {s1}, {s2}"
""",
)

OPTIMIZER_THREE_EVALUATIONS = OptimizerProgram(
    id="opt-three-evaluations",
    source_code="""
def improve_algorithm(api):
    s1 = api.evaluate("c1")
    s2 = api.evaluate("c2")
    s3 = api.evaluate("c3")
    return "done"
""",
)

OPTIMIZER_FAILING_EVALUATION = OptimizerProgram(
    id="opt-failing-eval",
    source_code="""
def improve_algorithm(api):
    try:
        api.evaluate("bad_candidate")
    except Exception:
        pass
    s2 = api.evaluate("good_candidate")
    return f"score: {s2}"
""",
)

OPTIMIZER_NO_EVALUATIONS = OptimizerProgram(
    id="opt-no-eval",
    source_code="""
def improve_algorithm(api):
    return "no eval performed"
""",
)


def test_e2e_inner_search_success():
    llm = MockLLM(["candidate_code_1", "candidate_code_2"])
    evaluator = FakeEvaluator(
        {"candidate_code_1": 0.25, "candidate_code_2": 0.80}
    )
    cap_limits = CapabilityLimits(
        max_generate_requests=5, max_evaluate_requests=5
    )

    worker_result, usage, inner_result = run_optimizer_inner_search(
        program=OPTIMIZER_TWO_CANDIDATES,
        llm=llm,
        evaluator=evaluator,
        capability_limits=cap_limits,
    )

    assert worker_result["status"] == "success"
    assert worker_result["result"] == "scores: 0.25, 0.8"
    assert usage.generate_requests == 2
    assert usage.evaluate_requests == 2

    # req-000001: generate, req-000002: evaluate c1 (0.25), req-000003: generate, req-000004: evaluate c2 (0.80)
    assert inner_result.best_candidate_id == "req-000004"
    assert inner_result.best_score == 0.80
    assert inner_result.utility == 0.80
    assert inner_result.generated_count == 2
    assert inner_result.evaluated_count == 2
    assert inner_result.valid_evaluation_count == 2
    assert inner_result.invalid_evaluation_count == 0
    assert len(inner_result.evaluations) == 2


def test_e2e_inner_search_evaluator_failure_handled():
    llm = MockLLM()
    evaluator = FakeEvaluator(
        {"bad_candidate": ValueError("eval error"), "good_candidate": 0.65}
    )
    cap_limits = CapabilityLimits(
        max_generate_requests=5, max_evaluate_requests=5
    )

    worker_result, usage, inner_result = run_optimizer_inner_search(
        program=OPTIMIZER_FAILING_EVALUATION,
        llm=llm,
        evaluator=evaluator,
        capability_limits=cap_limits,
    )

    assert worker_result["status"] == "success"
    assert worker_result["result"] == "score: 0.65"
    assert usage.evaluate_requests == 2

    assert inner_result.best_candidate_id == "req-000002"
    assert inner_result.best_score == 0.65
    assert inner_result.utility == 0.65
    assert inner_result.valid_evaluation_count == 1
    assert inner_result.invalid_evaluation_count == 1
    assert len(inner_result.evaluations) == 2


def test_e2e_inner_search_zero_valid():
    llm = MockLLM()
    evaluator = FakeEvaluator(
        {"bad_candidate": ValueError("crash")}
    )
    program = OptimizerProgram(
        id="opt-zero-valid",
        source_code="""
def improve_algorithm(api):
    try:
        api.evaluate("bad_candidate")
    except Exception:
        pass
    return "done"
""",
    )
    cap_limits = CapabilityLimits(
        max_generate_requests=5, max_evaluate_requests=5
    )

    worker_result, usage, inner_result = run_optimizer_inner_search(
        program=program,
        llm=llm,
        evaluator=evaluator,
        capability_limits=cap_limits,
    )

    assert worker_result["status"] == "success"
    assert usage.evaluate_requests == 1

    assert inner_result.best_candidate_id is None
    assert inner_result.best_score is None
    assert inner_result.utility is None
    assert inner_result.valid_evaluation_count == 0
    assert inner_result.invalid_evaluation_count == 1


def test_e2e_inner_search_zero_evaluations():
    llm = MockLLM()
    evaluator = FakeEvaluator()
    cap_limits = CapabilityLimits(
        max_generate_requests=5, max_evaluate_requests=5
    )

    worker_result, usage, inner_result = run_optimizer_inner_search(
        program=OPTIMIZER_NO_EVALUATIONS,
        llm=llm,
        evaluator=evaluator,
        capability_limits=cap_limits,
    )

    assert worker_result["status"] == "success"
    assert worker_result["result"] == "no eval performed"
    assert usage.evaluate_requests == 0
    assert inner_result.best_candidate_id is None
    assert inner_result.best_score is None
    assert inner_result.utility is None
    assert inner_result.generated_count == 0
    assert inner_result.evaluated_count == 0
    assert inner_result.valid_evaluation_count == 0
    assert inner_result.invalid_evaluation_count == 0


def test_e2e_inner_search_evaluation_budget_exhaustion():
    llm = MockLLM()
    evaluator = FakeEvaluator({"c1": 1.0, "c2": 2.0, "c3": 3.0})
    cap_limits = CapabilityLimits(
        max_generate_requests=5, max_evaluate_requests=2
    )

    worker_result, usage, inner_result = run_optimizer_inner_search(
        program=OPTIMIZER_THREE_EVALUATIONS,
        llm=llm,
        evaluator=evaluator,
        capability_limits=cap_limits,
    )

    assert worker_result["status"] == "failed"
    assert "evaluation_budget_exhausted" in worker_result["message"]
    assert usage.evaluate_requests == 2

    assert inner_result.best_candidate_id == "req-000002"
    assert inner_result.best_score == 2.0
    assert inner_result.utility == 2.0
    assert inner_result.evaluated_count == 2
    assert inner_result.valid_evaluation_count == 2
    assert inner_result.invalid_evaluation_count == 0
