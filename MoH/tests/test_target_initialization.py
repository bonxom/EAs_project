import hashlib

import pytest
from deployment_fakes import HEURISTIC, FakeEvaluator, FakeLLM
from utils.checkpoint import Elite


def _elite(size, idea, code):
    return Elite(size, idea, code, hashlib.sha256(code.encode()).hexdigest())


def test_initialization_uses_elites_but_scores_on_target_validation():
    from inference import initialize_target_population

    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    llm = FakeLLM()
    elites = (_elite(20, "reuse this elite idea", HEURISTIC),)
    pop = initialize_target_population(
        500,
        elites,
        llm,
        lambda code, idea: evaluator.evaluate(code, 500, "val"),
        HEURISTIC,
        seed_count=1,
        max_attempts=2,
    )
    assert pop.get_best_solution("tsp_gls-500")["utility"] == 1.0
    assert evaluator.calls == [(HEURISTIC, 500, "val")]
    assert "reuse this elite idea" in llm.messages[0]


def test_reference_selection_is_deterministic():
    from utils.elite_reference import EliteReferenceLLM

    elites = tuple(
        _elite(
            size,
            f"idea-{size}",
            HEURISTIC.replace("return edge_distance", f"return edge_distance  # {size}"),
        )
        for size in (20, 50, 100, 200)
    )
    first_client = FakeLLM()
    second_client = FakeLLM()
    first = EliteReferenceLLM(first_client, elites, 500, seed=17, reference_count=2)
    second = EliteReferenceLLM(second_client, reversed(elites), 500, seed=17, reference_count=2)
    first.prompt("expert", "generate")
    second.prompt("expert", "generate")
    assert first_client.messages == second_client.messages
    assert first.reference_code_hashes == second.reference_code_hashes
    assert len(first.reference_code_hashes) == 2


def test_no_training_numeric_scores_in_prompt():
    from utils.elite_reference import EliteReferenceLLM

    client = FakeLLM()
    wrapper = EliteReferenceLLM(
        client,
        (_elite(20, "elite idea", HEURISTIC),),
        500,
        reference_count=1,
    )
    wrapper.prompt_batch("expert", ["first", "second"])
    assert len(client.messages) == 2
    assert all("training_utility" not in message for message in client.messages)
    assert all("0.01" not in message for message in client.messages)


def test_duplicate_seeds_do_not_satisfy_requested_count():
    from inference import initialize_target_population

    evaluator = FakeEvaluator({HEURISTIC: 2.0})
    llm = FakeLLM([HEURISTIC, HEURISTIC, HEURISTIC])
    pop = initialize_target_population(
        500,
        (_elite(20, "reference", HEURISTIC),),
        llm,
        lambda code, idea: evaluator.evaluate(code, 500, "val"),
        HEURISTIC,
        seed_count=2,
        max_attempts=3,
    )
    assert pop.get_subtask_size("tsp_gls-500") == 1
    assert len(evaluator.calls) == 1
    assert len(llm.messages) == 3
    assert pop.initialization["partial"] is True


def test_distinct_seeds_with_equal_scores_are_kept():
    from inference import initialize_target_population

    other = HEURISTIC.replace("return edge_distance", "return edge_distance.copy()")
    evaluator = FakeEvaluator({HEURISTIC: 1.0, other: 1.0})
    pop = initialize_target_population(
        500,
        (_elite(20, "reference", HEURISTIC),),
        FakeLLM([HEURISTIC, other]),
        lambda code, idea: evaluator.evaluate(code, 500, "val"),
        HEURISTIC,
        seed_count=2,
        max_attempts=2,
    )
    assert pop.get_subtask_size("tsp_gls-500") == 2


def test_generation_failure_respects_attempt_limit():
    from inference import TargetInitializationError, initialize_target_population

    llm = FakeLLM(["not code", "still not code", "no code"])
    evaluator = FakeEvaluator()
    with pytest.raises(TargetInitializationError, match="500"):
        initialize_target_population(
            500,
            (_elite(20, "reference", HEURISTIC),),
            llm,
            lambda code, idea: evaluator.evaluate(code, 500, "val"),
            HEURISTIC,
            seed_count=1,
            max_attempts=3,
        )
    assert len(llm.messages) == 3
    assert evaluator.calls == []


def test_partial_population_can_proceed():
    from inference import initialize_target_population

    other = HEURISTIC.replace("return edge_distance", "return edge_distance.copy()")
    evaluator = FakeEvaluator({HEURISTIC: 1.0}, failed_codes=(other,))
    pop = initialize_target_population(
        1000,
        (_elite(20, "reference", HEURISTIC),),
        FakeLLM([other, HEURISTIC]),
        lambda code, idea: evaluator.evaluate(code, 1000, "val"),
        HEURISTIC,
        seed_count=2,
        max_attempts=2,
    )
    assert pop.get_subtask_size("tsp_gls-1000") == 1
    assert pop.initialization == {
        "requested": 2,
        "accepted": 1,
        "attempts": 2,
        "partial": True,
    }


def test_empty_population_fails_target():
    from inference import TargetInitializationError, initialize_target_population

    evaluator = FakeEvaluator(failed_codes=(HEURISTIC,))
    with pytest.raises(TargetInitializationError, match="no valid seed"):
        initialize_target_population(
            500,
            (_elite(20, "reference", HEURISTIC),),
            FakeLLM(),
            lambda code, idea: evaluator.evaluate(code, 500, "val"),
            HEURISTIC,
            seed_count=1,
            max_attempts=1,
        )


def test_shared_solution_files_are_never_read(monkeypatch):
    from inference import initialize_target_population
    from utils.population import Pop

    def forbidden(*_args, **_kwargs):
        raise AssertionError("shared solution file was read")

    monkeypatch.setattr(Pop, "load_subtask_from_file", forbidden)
    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    pop = initialize_target_population(
        500,
        (_elite(20, "reference", HEURISTIC),),
        FakeLLM(),
        lambda code, idea: evaluator.evaluate(code, 500, "val"),
        HEURISTIC,
        seed_count=1,
        max_attempts=1,
    )
    assert pop.get_subtask_size("tsp_gls-500") == 1
