"""Unit and integration tests for Full MoH Offline Orchestration (M5)."""

import pytest

from moh.core.models import OptimizerProgram
from moh.full_moh import (
    FullMoHConfig,
    FullMoHResult,
    FullMoHWorkCounts,
    run_full_moh,
)
from moh.llm.base import GenerationError
from moh.optimizers.evaluator import FakeEvaluator
from moh.optimizers.evolution import OuterEvolutionResult
from moh.optimizers.programs import parse_optimizer_program


class ScriptedLLM:
    """Helper LLM that returns scripted text responses sequentially."""

    def __init__(self, responses: list[str | Exception] | None = None):
        self._responses = list(responses or [])
        self._index = 0

    def generate(self, prompt: str) -> str:
        if self._index >= len(self._responses):
            raise GenerationError("Scripted responses exhausted")
        resp = self._responses[self._index]
        self._index += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


def create_simple_seed_program(prog_id: str) -> OptimizerProgram:
    return parse_optimizer_program(
        "def improve_algorithm(api):\n"
        f"    c = api.generate('KIND: mutate\\nOption {prog_id}.')\n"
        "    return api.evaluate(c)\n",
        prog_id,
        idea=f"Seed program {prog_id}",
    )


def test_full_moh_config_validation():
    cfg = FullMoHConfig(population_size=2, generations=1)
    assert cfg.population_size == 2
    assert cfg.generations == 1

    with pytest.raises(ValueError, match="population_size"):
        FullMoHConfig(population_size=0)

    with pytest.raises(ValueError, match="population_size"):
        FullMoHConfig(population_size=-1)

    with pytest.raises(ValueError, match="population_size"):
        FullMoHConfig(population_size=True)

    with pytest.raises(ValueError, match="generations"):
        FullMoHConfig(generations=-1)

    with pytest.raises(ValueError, match="generations"):
        FullMoHConfig(generations=True)

    with pytest.raises(TypeError, match="capability_limits"):
        FullMoHConfig(capability_limits="invalid")

    with pytest.raises(TypeError, match="program_limits"):
        FullMoHConfig(program_limits=123)


def test_full_moh_work_counts_validation():
    counts = FullMoHWorkCounts(
        outer_programs_evaluated=2,
        outer_offspring_generated=1,
        inner_generate_requests=4,
        inner_evaluate_requests=4,
    )
    assert counts.outer_programs_evaluated == 2
    assert counts.outer_offspring_generated == 1
    assert counts.inner_generate_requests == 4
    assert counts.inner_evaluate_requests == 4

    with pytest.raises(ValueError, match="outer_programs_evaluated"):
        FullMoHWorkCounts(
            outer_programs_evaluated=-1,
            outer_offspring_generated=0,
            inner_generate_requests=0,
            inner_evaluate_requests=0,
        )

    with pytest.raises(ValueError, match="outer_offspring_generated"):
        FullMoHWorkCounts(
            outer_programs_evaluated=0,
            outer_offspring_generated=True,
            inner_generate_requests=0,
            inner_evaluate_requests=0,
        )


def test_full_moh_result_properties_and_mirroring():
    evaluator = FakeEvaluator(
        {"def solve(): return 'sol'": 0.8},
        default_score=None,
    )
    inner_llm = ScriptedLLM(["def solve(): return 'sol'"] * 10)
    meta_llm = ScriptedLLM(
        [
            (
                "def improve_algorithm(api):\n"
                "    c = api.generate('KIND: mutate\\nOption 1.')\n"
                "    return api.evaluate(c)\n"
            )
        ]
        * 10
    )

    seeds = [create_simple_seed_program("o000001"), create_simple_seed_program("o000002")]
    config = FullMoHConfig(population_size=2, generations=1)
    res = run_full_moh(
        config=config,
        seed_programs=seeds,
        meta_llm=meta_llm,
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert isinstance(res, FullMoHResult)
    assert isinstance(res.outer_result, OuterEvolutionResult)
    assert isinstance(res.work_counts, FullMoHWorkCounts)
    assert res.status == res.outer_result.status
    assert res.winner == res.outer_result.winner
    assert res.best_program == res.outer_result.best_program
    assert res.best_inner_result == res.outer_result.best_inner_result
    assert res.best_utility == res.outer_result.best_utility
    assert res.best_utility == 0.8


def test_run_full_moh_zero_generations():
    evaluator = FakeEvaluator(
        {"def solve(): return 'sol'": 0.75},
        default_score=None,
    )
    inner_llm = ScriptedLLM(["def solve(): return 'sol'"] * 5)
    meta_llm = ScriptedLLM([])

    seeds = [create_simple_seed_program("o000001"), create_simple_seed_program("o000002")]
    config = FullMoHConfig(population_size=2, generations=0)
    res = run_full_moh(
        config=config,
        seed_programs=seeds,
        meta_llm=meta_llm,
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.status == "success"
    assert res.best_utility == 0.75
    assert res.work_counts.outer_programs_evaluated == 2
    assert res.work_counts.outer_offspring_generated == 0
    assert len(res.outer_result.generations) == 1
    assert res.outer_result.generations[0].generation == -1


def test_run_full_moh_one_generation_e2e():
    evaluator = FakeEvaluator(
        {
            "def solve(): return 'code_p1'": 0.30,
            "def solve(): return 'code_p2'": 0.70,
            "def solve(): return 'code_c1'": 0.95,
        },
        default_score=None,
    )

    inner_llm = ScriptedLLM(
        [
            "def solve(): return 'code_p1'",
            "def solve(): return 'code_p2'",
            "def solve(): return 'code_c1'",
        ]
    )

    meta_llm = ScriptedLLM(
        [
            (
                "def improve_algorithm(api):\n"
                "    c = api.generate('KIND: mutate\\nMutate solution.')\n"
                "    return api.evaluate(c)\n"
            )
        ]
    )

    seeds = [create_simple_seed_program("o000001"), create_simple_seed_program("o000002")]
    config = FullMoHConfig(population_size=2, generations=1)
    res = run_full_moh(
        config=config,
        seed_programs=seeds,
        meta_llm=meta_llm,
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.status == "success"
    assert res.best_utility == 0.95
    assert res.best_program is not None
    assert res.best_program.id == "o000003"
    assert res.work_counts.outer_programs_evaluated == 3
    assert res.work_counts.outer_offspring_generated == 1
    assert res.work_counts.inner_generate_requests == 3
    assert res.work_counts.inner_evaluate_requests == 3

    # Assert 2-level separation: outer program ID != inner candidate
    assert res.best_program.id.startswith("o")
    assert res.best_inner_result is not None
    assert res.best_inner_result.best_candidate_id is not None
    best_eval = next(
        e for e in res.best_inner_result.evaluations
        if e.candidate_id == res.best_inner_result.best_candidate_id
    )
    assert best_eval.source_code == "def solve(): return 'code_c1'"


def test_run_full_moh_two_generations_e2e():
    evaluator = FakeEvaluator(
        {
            "def solve(): return 'code_p1'": 0.20,
            "def solve(): return 'code_p2'": 0.50,
            "def solve(): return 'code_gen1'": 0.70,
            "def solve(): return 'code_gen2'": 0.90,
        },
        default_score=None,
    )

    inner_llm = ScriptedLLM(
        [
            "def solve(): return 'code_p1'",
            "def solve(): return 'code_p2'",
            "def solve(): return 'code_gen1'",
            "def solve(): return 'code_gen2'",
        ]
    )

    meta_llm = ScriptedLLM(
        [
            (
                "def improve_algorithm(api):\n"
                "    c = api.generate('KIND: mutate\\nGen 1.')\n"
                "    return api.evaluate(c)\n"
            ),
            (
                "def improve_algorithm(api):\n"
                "    c = api.generate('KIND: mutate\\nGen 2.')\n"
                "    return api.evaluate(c)\n"
            ),
        ]
    )

    seeds = [create_simple_seed_program("o000001"), create_simple_seed_program("o000002")]
    config = FullMoHConfig(population_size=2, generations=2)
    res = run_full_moh(
        config=config,
        seed_programs=seeds,
        meta_llm=meta_llm,
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.status == "success"
    assert res.best_utility == 0.90
    assert res.best_program.id == "o000004"
    assert res.work_counts.outer_programs_evaluated == 4
    assert res.work_counts.outer_offspring_generated == 2
    assert len(res.outer_result.generations) == 3


def test_run_full_moh_multi_step_inner_search():
    evaluator = FakeEvaluator(
        {
            "def solve(): return 'cand1'": 0.20,
            "def solve(): return 'cand2'": 0.85,
        },
        default_score=None,
    )

    inner_llm = ScriptedLLM(
        [
            "def solve(): return 'cand1'",
            "def solve(): return 'cand2'",
        ]
    )

    multi_step_program = parse_optimizer_program(
        "def improve_algorithm(api):\n"
        "    c1 = api.generate('KIND: mutate\\nFirst try.')\n"
        "    s1 = api.evaluate(c1)\n"
        "    c2 = api.generate('KIND: mutate\\nSecond try.')\n"
        "    s2 = api.evaluate(c2)\n"
        "    return max(s1, s2)\n",
        "o_multi",
        idea="Multi step candidate comparison",
    )

    config = FullMoHConfig(population_size=1, generations=0)
    res = run_full_moh(
        config=config,
        seed_programs=[multi_step_program],
        meta_llm=ScriptedLLM([]),
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.status == "success"
    assert res.best_utility == 0.85
    assert res.work_counts.inner_generate_requests == 2
    assert res.work_counts.inner_evaluate_requests == 2


def test_run_full_moh_all_none_utilities():
    evaluator = FakeEvaluator(default_score=None)
    inner_llm = ScriptedLLM(["def invalid(): syntax error !!!"] * 10)
    meta_llm = ScriptedLLM([])

    seeds = [create_simple_seed_program("o000001"), create_simple_seed_program("o000002")]
    config = FullMoHConfig(population_size=2, generations=0)
    res = run_full_moh(
        config=config,
        seed_programs=seeds,
        meta_llm=meta_llm,
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.status == "failed"
    assert res.winner is None
    assert res.best_program is None
    assert res.best_utility is None


def test_run_full_moh_negative_utilities():
    evaluator = FakeEvaluator(
        {
            "def solve(): return 'c1'": -10.0,
            "def solve(): return 'c2'": -3.0,
            "def solve(): return 'c3'": -7.0,
        },
        default_score=None,
    )

    inner_llm = ScriptedLLM(
        [
            "def solve(): return 'c1'",
            "def solve(): return 'c2'",
            "def solve(): return 'c3'",
        ]
    )

    p1 = parse_optimizer_program(
        "def improve_algorithm(api):\n"
        "    c = api.generate('KIND: mutate\\nOption 1.')\n"
        "    return api.evaluate(c)\n",
        "o000001",
    )
    p2 = parse_optimizer_program(
        "def improve_algorithm(api):\n"
        "    c = api.generate('KIND: mutate\\nOption 2.')\n"
        "    return api.evaluate(c)\n",
        "o000002",
    )
    p3 = parse_optimizer_program(
        "def improve_algorithm(api):\n"
        "    c = api.generate('KIND: mutate\\nOption 3.')\n"
        "    return api.evaluate(c)\n",
        "o000003",
    )

    config = FullMoHConfig(population_size=3, generations=0)
    res = run_full_moh(
        config=config,
        seed_programs=[p1, p2, p3],
        meta_llm=ScriptedLLM([]),
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.status == "success"
    assert res.best_utility == -3.0
    assert res.best_program.id == "o000002"


def test_run_full_moh_invalid_offspring_handled_gracefully():
    evaluator = FakeEvaluator(
        {"def solve(): return 'sol'": 0.50},
        default_score=None,
    )
    inner_llm = ScriptedLLM(["def solve(): return 'sol'"] * 5)
    meta_llm = ScriptedLLM(["not valid python code !!!"])

    seeds = [create_simple_seed_program("o000001"), create_simple_seed_program("o000002")]
    config = FullMoHConfig(population_size=2, generations=1)
    res = run_full_moh(
        config=config,
        seed_programs=seeds,
        meta_llm=meta_llm,
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.status == "success"
    assert res.best_utility == 0.50
    assert res.work_counts.outer_offspring_generated == 0
    assert res.work_counts.outer_programs_evaluated == 2


def test_run_full_moh_repeatability():
    def execute_run():
        evaluator = FakeEvaluator(
            {"def solve(): return 'sol'": 0.88},
            default_score=None,
        )
        inner_llm = ScriptedLLM(["def solve(): return 'sol'"] * 5)
        meta_llm = ScriptedLLM(
            [
                (
                    "def improve_algorithm(api):\n"
                    "    c = api.generate('KIND: mutate\\nTry.')\n"
                    "    return api.evaluate(c)\n"
                )
            ]
        )
        seeds = [create_simple_seed_program("o000001"), create_simple_seed_program("o000002")]
        return run_full_moh(
            config=FullMoHConfig(population_size=2, generations=1),
            seed_programs=seeds,
            meta_llm=meta_llm,
            inner_llm=inner_llm,
            evaluator=evaluator,
        )

    res1 = execute_run()
    res2 = execute_run()

    assert res1.status == res2.status
    assert res1.best_utility == res2.best_utility
    assert res1.best_program.id == res2.best_program.id
    assert res1.work_counts == res2.work_counts


def test_work_counts_generation_failure_after_admission():
    """Verify generate request admitted before LLM failure is counted in logical inner_generate_requests."""
    evaluator = FakeEvaluator(default_score=None)
    inner_llm = ScriptedLLM([GenerationError("LLM generation failed")])

    prog = parse_optimizer_program(
        "def improve_algorithm(api):\n"
        "    c = api.generate('KIND: mutate\\nTry.')\n"
        "    return api.evaluate(c)\n",
        "o000001",
    )

    config = FullMoHConfig(population_size=1, generations=0)
    res = run_full_moh(
        config=config,
        seed_programs=[prog],
        meta_llm=ScriptedLLM([]),
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.work_counts.inner_generate_requests == 1
    assert res.work_counts.inner_evaluate_requests == 0
    inner_res = res.outer_result.all_evaluated[0].inner_result
    assert inner_res.generated_count == 1
    assert len(inner_res.evaluations) == 0


def test_work_counts_evaluator_failure_after_admission():
    """Verify evaluate request admitted before evaluator exception is counted in logical inner_evaluate_requests."""

    def crashing_evaluator(_src: str):
        raise RuntimeError("Evaluator crashed")

    evaluator = FakeEvaluator(crashing_evaluator, default_score=None)
    inner_llm = ScriptedLLM(["def solve(): return 'c1'"])

    prog = parse_optimizer_program(
        "def improve_algorithm(api):\n"
        "    c = api.generate('KIND: mutate\\nTry.')\n"
        "    return api.evaluate(c)\n",
        "o000001",
    )

    config = FullMoHConfig(population_size=1, generations=0)
    res = run_full_moh(
        config=config,
        seed_programs=[prog],
        meta_llm=ScriptedLLM([]),
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    assert res.work_counts.inner_generate_requests == 1
    assert res.work_counts.inner_evaluate_requests == 1
    inner_res = res.outer_result.all_evaluated[0].inner_result
    assert inner_res.evaluated_count == 1
    assert inner_res.valid_evaluation_count == 0
    assert inner_res.invalid_evaluation_count == 1


def test_work_counts_no_double_counting_survivors_across_generations():
    """Verify surviving elite optimizers are counted exactly once across multiple generation snapshots."""
    evaluator = FakeEvaluator(
        {
            "def solve(): return 'code_p1'": 0.80,
            "def solve(): return 'code_p2'": 0.10,
            "def solve(): return 'code_gen1'": 0.05,
            "def solve(): return 'code_gen2'": 0.05,
        },
        default_score=None,
    )

    inner_llm = ScriptedLLM(
        [
            "def solve(): return 'code_p1'",
            "def solve(): return 'code_p2'",
            "def solve(): return 'code_gen1'",
            "def solve(): return 'code_gen2'",
        ]
    )

    meta_llm = ScriptedLLM(
        [
            (
                "def improve_algorithm(api):\n"
                "    c = api.generate('KIND: mutate\\nGen 1.')\n"
                "    return api.evaluate(c)\n"
            ),
            (
                "def improve_algorithm(api):\n"
                "    c = api.generate('KIND: mutate\\nGen 2.')\n"
                "    return api.evaluate(c)\n"
            ),
        ]
    )

    seeds = [create_simple_seed_program("o000001"), create_simple_seed_program("o000002")]
    config = FullMoHConfig(population_size=2, generations=2)
    res = run_full_moh(
        config=config,
        seed_programs=seeds,
        meta_llm=meta_llm,
        inner_llm=inner_llm,
        evaluator=evaluator,
    )

    # 2 seed programs + 2 offspring programs evaluated = 4 total outer evaluations
    assert res.work_counts.outer_programs_evaluated == 4
    assert res.work_counts.outer_offspring_generated == 2
    # 4 unique executions * 1 generate request each = 4 (NOT 2+2+2=6 snapshot appearances)
    assert res.work_counts.inner_generate_requests == 4
    assert res.work_counts.inner_evaluate_requests == 4


def test_hifo_absent_in_m5_data_models():
    result_fields = set(FullMoHResult.__dataclass_fields__.keys())
    config_fields = set(FullMoHConfig.__dataclass_fields__.keys())
    work_fields = set(FullMoHWorkCounts.__dataclass_fields__.keys())

    forbidden = {
        "foresight_state",
        "hindsight_memory",
        "semantic_diversity",
        "explore_mode",
        "exploit_mode",
        "verbal_gradient",
    }

    assert result_fields.isdisjoint(forbidden)
    assert config_fields.isdisjoint(forbidden)
    assert work_fields.isdisjoint(forbidden)
