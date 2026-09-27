"""Offline Unit Tests for M7A Experiment Protocol & Fair Budget Contract."""

import pytest

from moh.experiments.protocol import (
    BudgetMaxima,
    ExperimentManifest,
    ExperimentProtocolConfig,
    FullMoHAlgorithmConfig,
    LLMRuntimeConfig,
    TaskConfig,
    calculate_budget_maxima,
    compute_prompt_fingerprints,
    hash_config,
    validate_manifest_safety,
)


def test_config_validation_valid():
    config = ExperimentProtocolConfig()
    assert config.protocol_version == "1.0.0"
    assert config.algorithm.method == "full_moh"
    assert config.llm.sdk_retries == 0
    assert config.llm.provider_attempt_limit_per_request == 1


@pytest.mark.parametrize(
    "pop,gen,exc_type",
    [
        (0, 1, ValueError),
        (-1, 1, ValueError),
        (True, 1, ValueError),
        (2, -1, ValueError),
        (2, True, ValueError),
    ],
)
def test_invalid_algorithm_config(pop, gen, exc_type):
    with pytest.raises(exc_type):
        FullMoHAlgorithmConfig(population_size=pop, generations=gen)


def test_method_id_baseline():
    cfg = FullMoHAlgorithmConfig(method="full_moh")
    assert cfg.method == "full_moh"
    with pytest.raises(ValueError, match="method must be 'full_moh'"):
        FullMoHAlgorithmConfig(method="invalid_method")


@pytest.mark.parametrize(
    "retries,attempts,exc_type",
    [
        (1, 1, ValueError),
        (-1, 1, ValueError),
        (True, 1, ValueError),
        (0, 0, ValueError),
        (0, -1, ValueError),
        (0, True, ValueError),
    ],
)
def test_invalid_llm_runtime_config(retries, attempts, exc_type):
    with pytest.raises(exc_type):
        LLMRuntimeConfig(sdk_retries=retries, provider_attempt_limit_per_request=attempts)


def test_invalid_task_config_sizes():
    with pytest.raises(ValueError, match="supported TSP sizes"):
        TaskConfig(sizes=(15,))


def test_deterministic_config_hashing():
    cfg1 = ExperimentProtocolConfig(run_seed=42)
    cfg2 = ExperimentProtocolConfig(run_seed=42)
    cfg3 = ExperimentProtocolConfig(run_seed=43)

    hash1 = hash_config(cfg1)
    hash2 = hash_config(cfg2)
    hash3 = hash_config(cfg3)

    assert isinstance(hash1, str)
    assert len(hash1) == 64
    assert hash1 == hash2
    assert hash1 != hash3


def test_pilot_budget_reproduction():
    """Verify pilot topology reproduces M6D 2/1/2/2/3 structure."""
    cfg = ExperimentProtocolConfig(
        algorithm=FullMoHAlgorithmConfig(
            population_size=1,
            generations=1,
            max_inner_generate_requests=1,
            max_inner_evaluate_requests=1,
        ),
        llm=LLMRuntimeConfig(provider_attempt_limit_per_request=1),
    )
    maxima = calculate_budget_maxima(cfg)
    assert maxima == BudgetMaxima(
        max_optimizer_executions=2,
        max_outer_meta_generations=1,
        max_inner_generate_requests=2,
        max_inner_evaluate_requests=2,
        max_provider_attempts=3,
    )


def test_larger_hypothetical_budget_formula():
    """Verify multi-population topology pop=3, gen=2, inner_gen=5, inner_eval=5."""
    cfg = ExperimentProtocolConfig(
        algorithm=FullMoHAlgorithmConfig(
            population_size=3,
            generations=2,
            max_inner_generate_requests=5,
            max_inner_evaluate_requests=5,
        ),
        llm=LLMRuntimeConfig(provider_attempt_limit_per_request=1),
    )
    maxima = calculate_budget_maxima(cfg)
    assert maxima == BudgetMaxima(
        max_optimizer_executions=5,
        max_outer_meta_generations=2,
        max_inner_generate_requests=25,
        max_inner_evaluate_requests=25,
        max_provider_attempts=27,
    )


def test_prompt_fingerprints():
    fp = compute_prompt_fingerprints()
    assert "outer_program_prompt_sha256" in fp
    assert "inner_heuristic_prompt_sha256" in fp
    assert len(fp["outer_program_prompt_sha256"]) == 64
    assert len(fp["inner_heuristic_prompt_sha256"]) == 64


def test_manifest_creation_valid():
    manifest = ExperimentManifest(
        run_id="run_20260927_001",
        timestamp="2026-09-27T12:00:00Z",
        git_commit="9979c02ce8df37466595fe223f4887cf19c47bd2",
        git_dirty=False,
        method="full_moh",
        config_hash="a" * 64,
        requested_model="ag/gemini-3.6-flash-low",
        route_provider="antigravity",
        proxy="9router",
        proxy_version="0.5.81",
        api_mode="chat_completions",
        resolved_proxy_host="172.25.160.1",
        task_config={"family": "tsp", "sizes": [10]},
        algorithm_config={"population_size": 2, "generations": 2},
        llm_config={"sdk_retries": 0},
        prompt_fingerprints={"outer": "b" * 64},
        work_counts={
            "outer_programs_evaluated": 4,
            "outer_offspring_generated": 2,
            "inner_generate_requests": 20,
            "inner_evaluate_requests": 20,
        },
        token_counts={"outer_total_tokens": 1000, "inner_total_tokens": 5000},
        status="COMPLETED_VALID",
        best_utility=-2.637129,
        best_program_id="o000002",
    )
    assert manifest.status == "COMPLETED_VALID"
    assert manifest.best_utility == -2.637129


def test_manifest_safety_reject_secret():
    data = {
        "run_id": "run_001",
        "api_key": "sk-proj-secret123",
    }
    with pytest.raises(ValueError, match="forbidden secret key"):
        validate_manifest_safety(data)


def test_manifest_safety_reject_bool_in_count():
    data = {
        "run_id": "run_001",
        "work_counts": {
            "inner_generate_requests": True,
        },
    }
    with pytest.raises(ValueError, match="boolean value not allowed for integer count"):
        validate_manifest_safety(data)


def test_manifest_safety_reject_non_finite_float():
    data = {
        "run_id": "run_001",
        "utility": float("nan"),
    }
    with pytest.raises(ValueError, match="non-finite float"):
        validate_manifest_safety(data)


def test_failure_manifest_support():
    manifest = ExperimentManifest(
        run_id="run_failed_001",
        timestamp="2026-09-27T12:00:00Z",
        git_commit="9979c02ce8df37466595fe223f4887cf19c47bd2",
        git_dirty=False,
        method="full_moh",
        config_hash="a" * 64,
        requested_model="ag/gemini-3.6-flash-low",
        route_provider="antigravity",
        proxy="9router",
        proxy_version="0.5.81",
        api_mode="chat_completions",
        resolved_proxy_host="172.25.160.1",
        task_config={"family": "tsp", "sizes": [10]},
        algorithm_config={"population_size": 2, "generations": 2},
        llm_config={"sdk_retries": 0},
        prompt_fingerprints={"outer": "b" * 64},
        work_counts={
            "outer_programs_evaluated": 1,
            "outer_offspring_generated": 0,
            "inner_generate_requests": 1,
            "inner_evaluate_requests": 0,
        },
        token_counts={"outer_total_tokens": 100},
        status="PROVIDER_FAILURE",
        failure_stage="outer_meta_generation",
        error_code="HTTP_500",
        best_utility=None,
        best_program_id=None,
    )
    assert manifest.status == "PROVIDER_FAILURE"
    assert manifest.failure_stage == "outer_meta_generation"
    assert manifest.best_utility is None
