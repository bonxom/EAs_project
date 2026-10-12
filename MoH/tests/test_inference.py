import hashlib

import pytest
from deployment_fakes import HEURISTIC, OPTIMIZER, FakeEvaluator, FakeLLM
from utils.checkpoint import Elite, TrainingCheckpoint

BETTER = HEURISTIC.replace("return edge_distance", "return edge_distance.copy()")
THIRD = HEURISTIC.replace(
    "return edge_distance", "updated = edge_distance.copy()\n    return updated"
)


def _checkpoint():
    return TrainingCheckpoint(
        OPTIMIZER,
        (Elite(20, "reference", HEURISTIC, "fixture-hash"),),
        {},
    )


def test_inference_passes_heuristic_population_and_never_changes_driver(tmp_path):
    from inference import InferenceRunner

    observed = []

    class FakeProgram:
        def __init__(self, code, timeout, seed):
            assert code == OPTIMIZER

        def __call__(self, population, utility, language_model, function_format, task):
            observed.append(task)
            utility(HEURISTIC, "candidate", task)
            return "candidate", HEURISTIC, -999

    checkpoint = _checkpoint()
    runner = InferenceRunner(
        checkpoint,
        FakeLLM(),
        FakeEvaluator({HEURISTIC: 1.0}),
        tmp_path,
        HEURISTIC,
        iterations=10,
        seed_count=1,
        program_factory=FakeProgram,
    )
    selected = runner.run_target(500)
    assert observed == ["tsp_gls-500"] * 10
    assert selected.validation.mean_gap_pct == 1.0
    assert checkpoint.code == OPTIMIZER


def test_unreturned_evaluated_candidate_can_win(tmp_path):
    from inference import InferenceRunner

    class FakeProgram:
        def __init__(self, code, timeout, seed):
            pass

        def __call__(self, population, utility, language_model, function_format, task):
            utility(BETTER, "unreturned winner", task)
            best = population.get_best_solution(task)
            return best["idea"], best["best_sol"], 0.0

    runner = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        FakeEvaluator({HEURISTIC: 5.0, BETTER: 1.0}),
        tmp_path,
        HEURISTIC,
        iterations=1,
        seed_count=1,
        program_factory=FakeProgram,
    )
    selected = runner.run_target(500)
    assert selected.code == BETTER
    assert selected.idea == "unreturned winner"


def test_returned_unscored_code_is_evaluated(tmp_path):
    from inference import InferenceRunner

    class FakeProgram:
        def __init__(self, code, timeout, seed):
            pass

        def __call__(self, *args):
            return "returned", BETTER, -1000

    evaluator = FakeEvaluator({HEURISTIC: 5.0, BETTER: 2.0})
    selected = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        evaluator,
        tmp_path,
        HEURISTIC,
        iterations=1,
        seed_count=1,
        program_factory=FakeProgram,
    ).run_target(500)
    assert (BETTER, 500, "val") in evaluator.calls
    assert selected.validation.mean_gap_pct == 2.0


def test_child_population_mutations_are_not_trusted(tmp_path):
    from inference import InferenceRunner

    poison = HEURISTIC.replace("return edge_distance", "return edge_distance * 0")

    class FakeProgram:
        def __init__(self, code, timeout, seed):
            pass

        def __call__(self, population, utility, language_model, function_format, task):
            population.save_solution(task, "poison", poison, -1000)

    evaluator = FakeEvaluator({HEURISTIC: 3.0})
    selected = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        evaluator,
        tmp_path,
        HEURISTIC,
        iterations=1,
        seed_count=1,
        program_factory=FakeProgram,
    ).run_target(500)
    assert selected.code == HEURISTIC
    assert all(call[0] != poison for call in evaluator.calls)


def test_timeout_keeps_population_and_continues(tmp_path):
    from inference import InferenceRunner

    calls = []

    class FakeProgram:
        def __init__(self, code, timeout, seed):
            self.seed = seed

        def __call__(self, *args):
            calls.append(self.seed)
            raise TimeoutError("generated timeout")

    selected = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        FakeEvaluator({HEURISTIC: 1.0}),
        tmp_path,
        HEURISTIC,
        iterations=3,
        seed_count=1,
        program_factory=FakeProgram,
    ).run_target(500)
    assert len(calls) == 3
    assert selected.code == HEURISTIC


def test_program_construction_failure_keeps_population(tmp_path):
    from inference import InferenceRunner

    class BrokenProgram:
        def __init__(self, code, timeout, seed):
            raise RuntimeError("load failed")

    selected = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        FakeEvaluator({HEURISTIC: 1.0}),
        tmp_path,
        HEURISTIC,
        iterations=2,
        seed_count=1,
        program_factory=BrokenProgram,
    ).run_target(500)
    assert selected.code == HEURISTIC
    assert selected.metadata["iterations_completed"] == 2


def test_malformed_return_keeps_population(tmp_path):
    from inference import InferenceRunner

    class FakeProgram:
        def __init__(self, code, timeout, seed):
            pass

        def __call__(self, *args):
            return "wrong shape"

    selected = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        FakeEvaluator({HEURISTIC: 1.0}),
        tmp_path,
        HEURISTIC,
        iterations=2,
        seed_count=1,
        program_factory=FakeProgram,
    ).run_target(500)
    assert selected.code == HEURISTIC


def test_nonfinite_or_failed_evaluations_are_not_selected(tmp_path):
    from inference import InferenceRunner

    class FakeProgram:
        def __init__(self, code, timeout, seed):
            pass

        def __call__(self, population, utility, language_model, function_format, task):
            utility(BETTER, "nan", task)
            utility(THIRD, "failed", task)
            return "failed", THIRD, -1000

    evaluator = FakeEvaluator(
        {HEURISTIC: 4.0, BETTER: float("nan"), THIRD: 0.0},
        failed_codes=(THIRD,),
    )
    selected = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        evaluator,
        tmp_path,
        HEURISTIC,
        iterations=1,
        seed_count=1,
        program_factory=FakeProgram,
    ).run_target(500)
    assert selected.code == HEURISTIC
    assert selected.validation.mean_gap_pct == 4.0


def test_validation_cache_is_target_specific(tmp_path):
    from inference import InferenceRunner

    class NoopProgram:
        def __init__(self, code, timeout, seed):
            pass

        def __call__(self, population, utility, language_model, function_format, task):
            return population.get_best_solution(task).values()

    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    runner = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        evaluator,
        tmp_path,
        HEURISTIC,
        iterations=0,
        seed_count=1,
        program_factory=NoopProgram,
    )
    runner.run_target(500)
    runner.run_target(1000)
    assert (HEURISTIC, 500, "val") in evaluator.calls
    assert (HEURISTIC, 1000, "val") in evaluator.calls


def test_budget_includes_initialization_and_never_overshoots(tmp_path):
    from inference import InferenceRunner

    class FakeProgram:
        calls = 0

        def __init__(self, code, timeout, seed):
            pass

        def __call__(self, population, utility, language_model, function_format, task):
            type(self).calls += 1
            utility(BETTER if self.calls == 1 else THIRD, "new", task)
            return "new", BETTER, -1

    evaluator = FakeEvaluator({HEURISTIC: 5.0, BETTER: 2.0, THIRD: 1.0})
    selected = InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        evaluator,
        tmp_path,
        HEURISTIC,
        iterations=10,
        seed_count=1,
        max_evaluations=2,
        program_factory=FakeProgram,
    ).run_target(500)
    assert len(evaluator.calls) == 2
    assert FakeProgram.calls == 1
    assert selected.metadata["early_stop_reason"] == "evaluation_budget_exhausted"


def test_iteration_seeds_are_stable_and_distinct(tmp_path):
    from inference import InferenceRunner

    observed = []

    class SeedProgram:
        def __init__(self, code, timeout, seed):
            observed.append(seed)

        def __call__(self, population, utility, language_model, function_format, task):
            best = population.get_best_solution(task)
            return best["idea"], best["best_sol"], best["utility"]

    InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        FakeEvaluator({HEURISTIC: 1.0}),
        tmp_path,
        HEURISTIC,
        iterations=3,
        seed_count=1,
        seed=7,
        program_factory=SeedProgram,
    ).run_target(500)
    expected = [
        int.from_bytes(
            hashlib.sha256(f"7:500:{iteration}".encode()).digest()[:4], "big"
        )
        for iteration in range(3)
    ]
    assert observed == expected
    assert len(set(observed)) == 3


def test_real_generated_improver_runs_optimizer_in_child(tmp_path):
    from inference import InferenceRunner

    selected = InferenceRunner(
        _checkpoint(),
        FakeLLM([HEURISTIC, HEURISTIC]),
        FakeEvaluator({HEURISTIC: 1.0}),
        tmp_path,
        HEURISTIC,
        iterations=1,
        seed_count=1,
        optimizer_timeout=10,
    ).run_target(500)
    assert selected.validation.mean_gap_pct == 1.0
    assert selected.metadata["iterations_completed"] == 1


def test_cross_target_utility_request_is_rejected(tmp_path):
    from inference import InferenceRunner

    class CrossTargetProgram:
        def __init__(self, code, timeout, seed):
            pass

        def __call__(self, population, utility, language_model, function_format, task):
            with pytest.raises(ValueError, match="target"):
                utility(BETTER, "wrong", "tsp_gls-1000")
            best = population.get_best_solution(task)
            return best["idea"], best["best_sol"], best["utility"]

    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    InferenceRunner(
        _checkpoint(),
        FakeLLM(),
        evaluator,
        tmp_path,
        HEURISTIC,
        iterations=1,
        seed_count=1,
        program_factory=CrossTargetProgram,
    ).run_target(500)
    assert all(call[1] == 500 for call in evaluator.calls)
