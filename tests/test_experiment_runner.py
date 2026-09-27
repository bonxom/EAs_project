"""Offline Unit Tests for M7B-A Experiment Runner & Artifact Management."""

import json
from pathlib import Path

from moh.experiments.protocol import (
    ExperimentProtocolConfig,
    FullMoHAlgorithmConfig,
    LLMRuntimeConfig,
    TaskConfig,
    calculate_budget_maxima,
    compare_protocol_fairness,
)
from moh.experiments.runner import (
    generate_dry_run_preview,
    load_experiment_config,
    run_experiment,
)


def test_initial_population_accounting():
    """Verify initial P=2 predefined seed programs consume 0 outer meta LLM calls."""
    cfg = ExperimentProtocolConfig(
        algorithm=FullMoHAlgorithmConfig(population_size=2, generations=2)
    )
    maxima = calculate_budget_maxima(cfg)
    assert maxima.max_optimizer_executions == 4
    assert maxima.max_outer_meta_generations == 2  # Only G=2 generations consume meta LLM calls


def test_task_instance_evaluation_count():
    """Verify 20 logical evaluate requests on 3 instances map to 60 task-instance evaluations."""
    cfg = ExperimentProtocolConfig(
        task=TaskConfig(sizes=(10,), instances_per_task=3),
        algorithm=FullMoHAlgorithmConfig(
            population_size=2,
            generations=2,
            max_inner_generate_requests=5,
            max_inner_evaluate_requests=5,
        ),
    )
    maxima = calculate_budget_maxima(cfg)
    assert maxima.max_inner_evaluate_requests == 20
    assert maxima.max_task_instance_evaluations == 60


def test_real_api_opt_in_guard():
    """Verify real provider requires allow_real_api=True, returning dry-run preview otherwise."""
    cfg = ExperimentProtocolConfig(
        llm=LLMRuntimeConfig(provider="openai", requested_model="ag/gemini-3.6-flash-low")
    )
    result = run_experiment(cfg, allow_real_api=False)
    assert result["status"] == "dry_run_preview"
    assert result["allow_real_api"] is False
    assert result["per_run_budget_maxima"]["max_provider_attempts"] == 22


def test_dry_run_preview_structure():
    cfg = ExperimentProtocolConfig()
    preview = generate_dry_run_preview(cfg, replicates=3)
    assert preview["status"] == "dry_run_preview"
    assert preview["method"] == "full_moh"
    assert preview["per_run_budget_maxima"]["max_optimizer_executions"] == 4
    assert preview["per_run_budget_maxima"]["max_outer_meta_generations"] == 2
    assert preview["per_run_budget_maxima"]["max_inner_generate_requests"] == 20
    assert preview["per_run_budget_maxima"]["max_inner_evaluate_requests"] == 20
    assert preview["per_run_budget_maxima"]["max_task_instance_evaluations"] == 60
    assert preview["per_run_budget_maxima"]["max_provider_attempts"] == 22
    assert preview["campaign_sanity_budget_maxima"]["campaign_max_provider_attempts"] == 66


def test_offline_fake_experiment_e2e(tmp_path: Path):
    """Run full offline experiment with FakeLLM and verify artifacts."""
    cfg = ExperimentProtocolConfig(
        output_dir=tmp_path,
        algorithm=FullMoHAlgorithmConfig(
            population_size=1,
            generations=1,
            max_inner_generate_requests=1,
            max_inner_evaluate_requests=1,
        ),
        llm=LLMRuntimeConfig(provider="fake"),
    )
    res = run_experiment(cfg, allow_real_api=False)
    assert res["status"] in ("COMPLETED_VALID", "COMPLETED_NO_VALID_UTILITY")
    
    manifest_path = Path(res["manifest_path"])
    assert manifest_path.exists()
    
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["method"] == "full_moh"
    assert manifest_data["work_counts"]["outer_programs_evaluated"] == 2
    assert manifest_data["work_counts"]["inner_generate_requests"] == 1
    assert manifest_data["work_counts"]["inner_evaluate_requests"] == 0

    # Check program artifacts exist
    run_dir = manifest_path.parent
    programs_dir = run_dir / "programs"
    assert programs_dir.exists()
    assert (programs_dir / "o000001.py").exists()

    # Check trajectory artifact exists
    assert (run_dir / "trajectory.json").exists()


def test_config_comparability_helper():
    """Verify compare_protocol_fairness catches mismatches and allows method changes."""
    cfg1 = ExperimentProtocolConfig(
        algorithm=FullMoHAlgorithmConfig(population_size=2, generations=2),
        llm=LLMRuntimeConfig(requested_model="ag/gemini-3.6-flash-low"),
    )
    cfg2 = ExperimentProtocolConfig(
        algorithm=FullMoHAlgorithmConfig(population_size=2, generations=2),
        llm=LLMRuntimeConfig(requested_model="ag/gemini-3.6-flash-low"),
    )
    
    is_fair, mismatches = compare_protocol_fairness(cfg1, cfg2)
    assert is_fair is True
    assert len(mismatches) == 0

    # Test mismatch in population size
    cfg3 = ExperimentProtocolConfig(
        algorithm=FullMoHAlgorithmConfig(population_size=3, generations=2)
    )
    is_fair, mismatches = compare_protocol_fairness(cfg1, cfg3)
    assert is_fair is False
    assert any("population_size mismatch" in m for m in mismatches)


def test_load_experiment_config_file(tmp_path: Path):
    yaml_content = """
protocol_version: "1.0.0"
run_seed: 100
replicate_index: 1

algorithm:
  method: "full_moh"
  population_size: 2
  generations: 2
  max_inner_generate_requests: 5
  max_inner_evaluate_requests: 5

llm:
  requested_model: "ag/gemini-3.6-flash-low"
  provider: "openai"

task:
  family: "tsp"
  sizes: [10]
  instances_per_task: 3
  root_seed: 42

output_dir: "outputs/test_run"
"""
    file_path = tmp_path / "test_config.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")

    config = load_experiment_config(file_path)
    assert config.run_seed == 100
    assert config.replicate_index == 1
    assert config.algorithm.population_size == 2
    assert config.llm.requested_model == "ag/gemini-3.6-flash-low"
    assert config.task.instances_per_task == 3
