"""Unit and integration tests for M4 Outer Evolution over OptimizerProgram."""

import pytest

from moh.core.models import OptimizerProgram
from moh.optimizers.evaluator import (
    CandidateEvaluationResult,
    FakeEvaluator,
    InnerSearchResult,
)
from moh.optimizers.evolution import (
    EvaluatedOptimizer,
    initial_optimizer_programs,
    outer_rank_key,
    run_outer_evolution,
)


class MockLLM:
    """Simple deterministic LLM mock for outer evolution tests."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])

    def generate(self, prompt: str) -> str:
        if prompt.startswith("KIND: optimizer_program"):
            if self.responses:
                val = self.responses.pop(0)
                if isinstance(val, Exception):
                    raise val
                return val
            return (
                "def improve_algorithm(api):\n"
                "    code = api.generate('KIND: mutate\\nDefault option.')\n"
                "    return api.evaluate(code)\n"
            )
        return "default_code"


def make_inner_result(utility: float | None, best_id: str | None = "req-001") -> InnerSearchResult:
    evals = ()
    if utility is not None:
        evals = (
            CandidateEvaluationResult(
                candidate_id=best_id or "req-001",
                source_code="def solve(): pass",
                score=utility,
                valid=True,
            ),
        )
        return InnerSearchResult(
            best_candidate_id=best_id,
            best_score=utility,
            utility=utility,
            evaluations=evals,
            generated_count=1,
            evaluated_count=1,
            valid_evaluation_count=1,
            invalid_evaluation_count=0,
        )
    return InnerSearchResult(
        best_candidate_id=None,
        best_score=None,
        utility=None,
        evaluations=(),
        generated_count=0,
        evaluated_count=0,
        valid_evaluation_count=0,
        invalid_evaluation_count=0,
    )


def test_evaluated_optimizer_dataclass():
    prog = OptimizerProgram(id="p1", source_code="def improve_algorithm(api): pass")
    inner = make_inner_result(0.85, "req-001")
    eval_opt = EvaluatedOptimizer(program=prog, inner_result=inner, utility=0.85)

    assert eval_opt.program == prog
    assert eval_opt.inner_result == inner
    assert eval_opt.utility == 0.85

    with pytest.raises(ValueError, match="utility must match inner_result.utility"):
        EvaluatedOptimizer(program=prog, inner_result=inner, utility=0.50)


def test_outer_rank_key():
    prog1 = OptimizerProgram(id="p1", source_code="def improve_algorithm(api): pass")
    prog2 = OptimizerProgram(id="p2", source_code="def improve_algorithm(api): pass")
    prog3 = OptimizerProgram(id="p3", source_code="def improve_algorithm(api): pass")
    prog4 = OptimizerProgram(id="p4", source_code="def improve_algorithm(api): pass")

    eo1 = EvaluatedOptimizer(prog1, make_inner_result(0.8), 0.8)
    eo2 = EvaluatedOptimizer(prog2, make_inner_result(0.4), 0.4)
    eo3 = EvaluatedOptimizer(prog3, make_inner_result(-3.0), -3.0)
    eo4 = EvaluatedOptimizer(prog4, make_inner_result(None), None)

    # 0.8 ranks above 0.4
    assert outer_rank_key(eo1) < outer_rank_key(eo2)
    # 0.4 ranks above -3.0
    assert outer_rank_key(eo2) < outer_rank_key(eo3)
    # -3.0 ranks above None
    assert outer_rank_key(eo3) < outer_rank_key(eo4)


def test_outer_rank_key_negative_scores():
    prog1 = OptimizerProgram(id="p1", source_code="pass")
    prog2 = OptimizerProgram(id="p2", source_code="pass")
    prog3 = OptimizerProgram(id="p3", source_code="pass")

    eo1 = EvaluatedOptimizer(prog1, make_inner_result(-10.0), -10.0)
    eo2 = EvaluatedOptimizer(prog2, make_inner_result(-3.0), -3.0)
    eo3 = EvaluatedOptimizer(prog3, make_inner_result(-7.0), -7.0)

    ranked = sorted([eo1, eo2, eo3], key=outer_rank_key)
    # Best is -3.0 (eo2), then -7.0 (eo3), then -10.0 (eo1)
    assert ranked[0] == eo2
    assert ranked[1] == eo3
    assert ranked[2] == eo1


def test_outer_rank_key_ties_first_wins():
    prog1 = OptimizerProgram(id="p1", source_code="pass")
    prog2 = OptimizerProgram(id="p2", source_code="pass")

    eo1 = EvaluatedOptimizer(prog1, make_inner_result(0.8, "req-1"), 0.8)
    eo2 = EvaluatedOptimizer(prog2, make_inner_result(0.8, "req-2"), 0.8)

    ranked = sorted([eo1, eo2], key=outer_rank_key)
    # "p1" < "p2" deterministically
    assert ranked[0] == eo1
    assert ranked[1] == eo2


def test_initial_optimizer_programs():
    seeds = initial_optimizer_programs(3)
    assert len(seeds) == 3
    assert seeds[0].id == "o000001"
    assert seeds[1].id == "o000002"
    assert seeds[2].id == "o000003"
    assert "def improve_algorithm(api):" in seeds[0].source_code


def test_e2e_outer_evolution_initial_population():
    seeds = initial_optimizer_programs(2)
    # Candidate codes mapped to scores in evaluator
    evaluator = FakeEvaluator(default_score=0.5)
    llm = MockLLM()

    result = run_outer_evolution(
        initial_programs=seeds,
        llm=llm,
        evaluator=evaluator,
        iterations=0,
        outer_capacity=2,
    )

    assert result.status == "success"
    assert result.winner is not None
    assert result.best_utility == 0.5
    assert result.programs_evaluated == 2
    assert result.offspring_generated == 0
    assert len(result.generations) == 1


def test_e2e_outer_evolution_one_generation():
    p1 = OptimizerProgram(
        id="p1",
        source_code="""
def improve_algorithm(api):
    return api.evaluate("c1")
""",
    )
    p2 = OptimizerProgram(
        id="p2",
        source_code="""
def improve_algorithm(api):
    return api.evaluate("c2")
""",
    )

    # Evaluator scores: c1 -> 0.2, c2 -> 0.8, c3 (offspring) -> 0.95
    evaluator = FakeEvaluator({"c1": 0.2, "c2": 0.8, "c3": 0.95})

    offspring_code = """
def improve_algorithm(api):
    return api.evaluate("c3")
"""
    llm = MockLLM([offspring_code])

    result = run_outer_evolution(
        initial_programs=[p1, p2],
        llm=llm,
        evaluator=evaluator,
        iterations=1,
        outer_capacity=2,
    )

    assert result.status == "success"
    assert result.winner is not None
    assert result.winner.program.id == "o000003"
    assert result.best_utility == 0.95
    assert result.programs_evaluated == 3
    assert result.offspring_generated == 1
    assert len(result.generations) == 2


def test_e2e_outer_best_parent_survives_when_offspring_worse():
    p1 = OptimizerProgram(
        id="p1",
        source_code="""
def improve_algorithm(api):
    return api.evaluate("good_code")
""",
    )
    p2 = OptimizerProgram(
        id="p2",
        source_code="""
def improve_algorithm(api):
    return api.evaluate("med_code")
""",
    )

    evaluator = FakeEvaluator({"good_code": 0.90, "med_code": 0.70, "worse_code": 0.30})
    offspring_code = """
def improve_algorithm(api):
    return api.evaluate("worse_code")
"""
    llm = MockLLM([offspring_code])

    result = run_outer_evolution(
        initial_programs=[p1, p2],
        llm=llm,
        evaluator=evaluator,
        iterations=1,
        outer_capacity=2,
    )

    assert result.status == "success"
    assert result.winner is not None
    assert result.winner.program.id == "p1"
    assert result.best_utility == 0.90
    assert result.programs_evaluated == 3
    assert result.offspring_generated == 1


def test_e2e_invalid_offspring_handled():
    p1 = OptimizerProgram(
        id="p1",
        source_code="""
def improve_algorithm(api):
    return api.evaluate("c1")
""",
    )
    evaluator = FakeEvaluator({"c1": 0.5})

    # LLM returns malformed code missing def improve_algorithm(api):
    llm = MockLLM(["invalid code block"])

    events = []
    result = run_outer_evolution(
        initial_programs=[p1],
        llm=llm,
        evaluator=evaluator,
        iterations=1,
        outer_capacity=1,
        emit=lambda e, p: events.append((e, p)),
    )

    assert result.status == "success"
    assert result.winner.program.id == "p1"
    assert result.best_utility == 0.5
    assert any(e == "outer_generation_failed" for e, _ in events)


def test_e2e_all_outer_utilities_none():
    p1 = OptimizerProgram(
        id="p1",
        source_code="""
def improve_algorithm(api):
    try:
        api.evaluate("bad_candidate")
    except Exception:
        pass
    return "done"
""",
    )

    def failing_evaluator(src: str):
        raise ValueError("all fail")

    llm = MockLLM()

    result = run_outer_evolution(
        initial_programs=[p1],
        llm=llm,
        evaluator=failing_evaluator,
        iterations=1,
        outer_capacity=1,
    )

    assert result.status == "failed"
    assert result.winner is None
    assert result.best_program is None
    assert result.best_utility is None
