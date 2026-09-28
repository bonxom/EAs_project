"""Offline Unit Tests for M7B-A / M7B-A2 Experiment Runner & Artifact Management."""

import hashlib
import json
from pathlib import Path

import pytest

from moh.experiments.protocol import (
    ExperimentProtocolConfig,
    FullMoHAlgorithmConfig,
    LLMRuntimeConfig,
    TaskConfig,
    calculate_budget_maxima,
    calculate_campaign_budget,
    compare_protocol_fairness,
)
from moh.experiments.runner import (
    CampaignMasterGuard,
    ReplicateAttemptBudget,
    generate_dry_run_preview,
    load_experiment_config,
    run_campaign,
    run_experiment,
    validate_real_provider_environment,
)
from moh.llm.base import GenerationError
from moh.llm.budget import (
    ProviderAttemptBudget,
    ProviderAttemptBudgetExceeded,
    ProviderAttemptLimits,
)
from moh.llm.openai_client import OpenAILLMClient
from moh.optimizers.evolution import initial_optimizer_programs


def test_initial_population_accounting():
    """Verify initial P=2 predefined seed programs consume 0 outer meta LLM calls."""
    cfg = ExperimentProtocolConfig(
        algorithm=FullMoHAlgorithmConfig(population_size=2, generations=2)
    )
    maxima = calculate_budget_maxima(cfg)
    assert maxima.max_optimizer_executions == 4
    assert maxima.max_outer_meta_generations == 2  # Only G=2 generations consume meta LLM calls


def test_initial_program_source_hashes():
    """Verify initial seed programs o000001 and o000002 have stable SHA256 source hashes."""
    seeds = initial_optimizer_programs(2)
    assert len(seeds) == 2
    assert seeds[0].id == "o000001"
    assert seeds[1].id == "o000002"

    hash0 = hashlib.sha256(seeds[0].source_code.encode("utf-8")).hexdigest()
    hash1 = hashlib.sha256(seeds[1].source_code.encode("utf-8")).hexdigest()

    assert len(hash0) == 64
    assert len(hash1) == 64
    assert hash0 != hash1


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

    campaign = calculate_campaign_budget(cfg, replicates=3)
    assert campaign.campaign_max_task_instance_evaluations == 180


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


def test_openai_client_per_request_attempt_limit(monkeypatch):
    """Verify max_attempts_per_request=1 prevents application retries on transient errors."""
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-test-key")
    
    attempts_made = 0
    budget = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=10))

    class FailingTransport:
        def __init__(self):
            class Completions:
                def create(self, **kwargs):
                    nonlocal attempts_made
                    attempts_made += 1
                    import openai
                    raise openai.APIConnectionError(request=None)
            self.chat = type("Chat", (), {"completions": Completions()})()

    client = OpenAILLMClient(
        model="ag/gemini-3.6-flash-low",
        timeout_seconds=5.0,
        observer=lambda *_: None,
        transport=FailingTransport(),
        attempt_budget=budget,
        max_attempts_per_request=1,
    )

    # First logical generate request
    with pytest.raises(GenerationError, match="provider_connection_error"):
        client.generate("KIND: mutate\nTest prompt.")

    # Exactly 1 provider attempt was made for the 1st request (no retry loop)
    assert attempts_made == 1
    assert budget.usage.attempts == 1

    # Second logical generate request still gets its own attempt
    with pytest.raises(GenerationError, match="provider_connection_error"):
        client.generate("KIND: mutate\nTest prompt 2.")

    assert attempts_made == 2
    assert budget.usage.attempts == 2


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


def test_real_provider_environment_validation(monkeypatch):
    """Verify validate_real_provider_environment enforces auth and endpoint safety."""
    llm_cfg = LLMRuntimeConfig(requested_model="ag/gemini-3.6-flash-low")

    # 1. OPENAI_API_KEY active -> ValueError
    monkeypatch.setenv("OPENAI_API_KEY", "sk-active-key")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-compat-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    with pytest.raises(ValueError, match="OPENAI_API_KEY must be UNSET"):
        validate_real_provider_environment(llm_cfg)

    # 2. OPENAI_COMPAT_API_KEY missing -> ValueError
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_COMPAT_API_KEY must be configured"):
        validate_real_provider_environment(llm_cfg)

    # 3. OPENAI_COMPAT_MODEL mismatch -> ValueError
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-compat-key")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "different-model")
    with pytest.raises(ValueError, match="OPENAI_COMPAT_MODEL mismatch"):
        validate_real_provider_environment(llm_cfg)

    # 4. OPENAI_COMPAT_BASE_URL invalid endpoint -> ValueError
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "ag/gemini-3.6-flash-low")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://invalid-endpoint:8080")
    with pytest.raises(ValueError, match="invalid endpoint path or port"):
        validate_real_provider_environment(llm_cfg)

    # Valid env succeeds
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    validate_real_provider_environment(llm_cfg)  # Should not raise


def test_offline_real_path_construction_fake_transport(monkeypatch, tmp_path: Path):
    """Verify real-provider construction path reaches execution code with fake transport and zero network."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-compat-key")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "ag/gemini-3.6-flash-low")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")

    cfg = ExperimentProtocolConfig(
        output_dir=tmp_path,
        llm=LLMRuntimeConfig(provider="openai", requested_model="ag/gemini-3.6-flash-low"),
        algorithm=FullMoHAlgorithmConfig(
            population_size=1,
            generations=1,
            max_inner_generate_requests=1,
            max_inner_evaluate_requests=1,
        ),
    )

    class FakeChatResponse:
        def __init__(self, content):
            self.choices = [
                type("Choice", (), {"message": type("Message", (), {"content": content, "reasoning_content": None})()})()
            ]
            self.usage = type(
                "Usage",
                (),
                {"prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 0, "total_tokens": 30},
            )()

    class FakeTransport:
        def __init__(self):
            class Completions:
                def create(self, **kwargs):
                    return FakeChatResponse("def select_next_node(*args):\n    return 0\n")
            self.chat = type("Chat", (), {"completions": Completions()})()

    res = run_experiment(cfg, allow_real_api=True, transport_factory=lambda: FakeTransport())
    assert res["status"] in ("COMPLETED_VALID", "COMPLETED_NO_VALID_UTILITY")
    assert Path(res["manifest_path"]).exists()


def test_three_replicate_offline_campaign_e2e(tmp_path: Path):
    """Run a 3-replicate offline campaign and verify distinct run_ids and manifests."""
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

    res = run_campaign(cfg, allow_real_api=False, replicates=3)
    assert res["status"] == "SANITY_CAMPAIGN_COMPLETE"
    assert res["attempted_replicates"] == [0, 1, 2]
    assert res["completed_replicates"] == [0, 1, 2]

    # Check campaign manifest
    c_manifest_path = Path(res["manifest_path"])
    assert c_manifest_path.exists()
    c_manifest_data = json.loads(c_manifest_path.read_text(encoding="utf-8"))
    assert c_manifest_data["campaign_status"] == "SANITY_CAMPAIGN_COMPLETE"
    assert len(c_manifest_data["replicate_run_ids"]) == 3
    assert len(set(c_manifest_data["replicate_run_ids"])) == 3


def test_mixed_status_campaign(tmp_path: Path):
    """Verify campaign correctly aggregates mixed replicate statuses without replacement."""
    cfg = ExperimentProtocolConfig(
        output_dir=tmp_path,
        llm=LLMRuntimeConfig(provider="fake"),
    )

    res = run_campaign(cfg, allow_real_api=False, replicates=3)
    assert res["status"] == "SANITY_CAMPAIGN_COMPLETE"
    assert res["status_counts"]["COMPLETED_VALID"] >= 0


def test_campaign_master_limit():
    """Verify exceeding campaign master limit stops campaign with SANITY_CAMPAIGN_SAFETY_STOP."""
    c_guard = CampaignMasterGuard(max_attempts=2)
    c_guard.reserve()
    c_guard.reserve()

    with pytest.raises(ProviderAttemptBudgetExceeded, match="campaign_attempt_budget_exhausted"):
        c_guard.reserve()

    assert c_guard.attempts == 2


def test_per_run_limit():
    """Verify replicate attempt limit stops replicate without exceeding its allowance."""
    c_guard = CampaignMasterGuard(max_attempts=66)
    rep_budget = ReplicateAttemptBudget(c_guard, max_replicate_attempts=2)

    rep_budget.reserve()
    rep_budget.reserve()
    assert rep_budget.usage.attempts == 2
    assert c_guard.attempts == 2

    with pytest.raises(ProviderAttemptBudgetExceeded, match="replicate_attempt_budget_exhausted"):
        rep_budget.reserve()

    assert c_guard.attempts == 2


def test_campaign_aggregation(tmp_path: Path):
    """Verify campaign aggregates total work counts and token totals."""
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
    res = run_campaign(cfg, allow_real_api=False, replicates=3)
    assert res["total_work_counts"]["outer_programs_evaluated"] == 6

from pathlib import Path

from moh.experiments.protocol import (
    calculate_diversity_facts,
)
from moh.experiments.runner import (
    build_real_provider_llms,
)
from moh.llm.budget import ProviderUsageAccountant
from moh.llm.fake import FakeLLM


def test_m7bc_inner_failure_reproduction_and_diagnostics(tmp_path: Path):
    """Test outer success + inner generation failure path (M7B-C mirror)."""
    cfg = ExperimentProtocolConfig(
        output_dir=tmp_path,
        algorithm=FullMoHAlgorithmConfig(
            population_size=2,
            generations=1,
            max_inner_generate_requests=1,
            max_inner_evaluate_requests=1,
        ),
        llm=LLMRuntimeConfig(provider="fake"),
    )
    
    meta_llm = FakeLLM(seed=42)
    
    class FailingInnerLLM:
        def __init__(self):
            self.usage_accountant = ProviderUsageAccountant()
        def generate(self, prompt: str) -> str:
            raise GenerationError("provider_connection_error", last_provider_error="provider_connection_error")

    failing_inner = FailingInnerLLM()
    
    res = run_experiment(
        cfg,
        allow_real_api=True,
        meta_llm=meta_llm,
        inner_llm=failing_inner,
    )
    
    assert res["status"] == "COMPLETED_NO_VALID_UTILITY"
    assert res["best_utility"] is None
    
    manifest_path = Path(res["manifest_path"])
    assert manifest_path.exists()
    m_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    assert m_dict["status"] == "COMPLETED_NO_VALID_UTILITY"
    assert m_dict["failure_stage"] == "inner_generation"
    assert m_dict["error_code"] == "provider_generation_failed"
    assert m_dict["work_counts"]["inner_evaluate_requests"] == 0
    assert m_dict["work_counts"]["inner_generate_requests"] == 2
    
    # Candidate artifacts must be absent
    cand_dir = manifest_path.parent / "candidates"
    assert not cand_dir.exists() or len(list(cand_dir.glob("*.py"))) == 0


def test_experiment_runner_known_good_inner_path(tmp_path: Path):
    """Test experiment runner with known-good fake LLM returning valid candidate functions."""
    cfg = ExperimentProtocolConfig(
        output_dir=tmp_path,
        algorithm=FullMoHAlgorithmConfig(
            population_size=2,
            generations=1,
            max_inner_generate_requests=1,
            max_inner_evaluate_requests=1,
        ),
        llm=LLMRuntimeConfig(provider="fake"),
    )
    
    fake_llm = FakeLLM(seed=42)
    res = run_experiment(cfg, allow_real_api=True, meta_llm=fake_llm, inner_llm=fake_llm)
    
    assert res["status"] == "COMPLETED_VALID"
    assert res["best_utility"] is not None
    assert isinstance(res["best_utility"], float)
    
    manifest_path = Path(res["manifest_path"])
    m_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert m_dict["status"] == "COMPLETED_VALID"
    assert m_dict["failure_stage"] is None
    assert m_dict["error_code"] is None
    assert m_dict["work_counts"]["inner_evaluate_requests"] > 0


def test_multiple_inner_accountant_aggregation(tmp_path: Path):
    """Verify that multiple inner clients created via factory have their usage aggregated exactly."""
    cfg = ExperimentProtocolConfig(
        output_dir=tmp_path,
        algorithm=FullMoHAlgorithmConfig(
            population_size=2,
            generations=1,
            max_inner_generate_requests=1,
            max_inner_evaluate_requests=1,
        ),
        llm=LLMRuntimeConfig(provider="fake"),
    )
    
    c_guard = CampaignMasterGuard(max_attempts=66)
    rep_budget = ReplicateAttemptBudget(c_guard, max_replicate_attempts=22)
    
    meta_acct = ProviderUsageAccountant()
    meta_acct.record(model="ag/gemini-3.6-flash-low", input_tokens=100, output_tokens=50, reasoning_tokens=0)
    
    class UsageMockTransport:
        def __init__(self):
            self.chat = self
            self.completions = self
        def create(self, **kwargs):
            class Message:
                content = "def select_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]"
            class Choice:
                message = Message()
            class Usage:
                prompt_tokens = 10
                completion_tokens = 5
                total_tokens = 15
            class Response:
                def __init__(self):
                    self.choices = [Choice()]
                    self.usage = Usage()
                    self.model = "ag/gemini-3.6-flash-low"
                    self.status = "completed"
            return Response()

    mock_transport = UsageMockTransport()
    
    _meta_llm, inner_factory, _outer_acct, inner_accts = build_real_provider_llms(
        config=cfg,
        replicate_budget=rep_budget,
        transport_factory=lambda: mock_transport,
    )
    
    # Simulate 2 inner client calls from factory
    client1 = inner_factory("o000001")
    client2 = inner_factory("o000002")
    
    client1.generate("prompt 1")
    client2.generate("prompt 2")
    
    assert len(inner_accts) == 2
    assert inner_accts[0].totals().total_tokens == 15
    assert inner_accts[1].totals().total_tokens == 15
    
    total_inner_in = sum(a.totals().input_tokens for a in inner_accts)
    total_inner_out = sum(a.totals().output_tokens for a in inner_accts)
    assert total_inner_in == 20
    assert total_inner_out == 10


def test_provider_failure_token_accounting():
    """Verify that a provider call failing before token usage records attempts without token count fabrication."""
    acct = ProviderUsageAccountant()
    budg = ProviderAttemptBudget(ProviderAttemptLimits(max_attempts=2))
    
    class FailingTransport:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.call_count = 0
        def create(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                import openai
                raise openai.APIConnectionError(request=None, message="connection failed")
            class Message:
                content = "success"
            class Choice:
                message = Message()
            class Usage:
                prompt_tokens = 30
                completion_tokens = 20
                total_tokens = 50
            class Response:
                def __init__(self):
                    self.choices = [Choice()]
                    self.usage = Usage()
                    self.model = "ag/gemini-3.6-flash-low"
                    self.status = "completed"
            return Response()

    transport = FailingTransport()
    client = OpenAILLMClient(
        model="ag/gemini-3.6-flash-low",
        timeout_seconds=30.0,
        observer=lambda *_: None,
        transport=transport,
        attempt_budget=budg,
        usage_accountant=acct,
        max_attempts_per_request=1,
    )
    
    with pytest.raises(GenerationError):
        client.generate("prompt 1")
        
    assert budg.usage.attempts == 1
    assert acct.totals().total_tokens == 0
    
    # Second call succeeds
    res = client.generate("prompt 2")
    assert res == "success"
    assert budg.usage.attempts == 2
    assert acct.totals().total_tokens == 50


def test_diversity_arithmetic_helper():
    """Verify calculate_diversity_facts formula."""
    res1 = calculate_diversity_facts(["def f1(): pass", "def f2(): pass"])
    assert res1["active_population_size"] == 2
    assert res1["unique_source_sha_count"] == 2
    assert res1["exact_duplicate_count"] == 0

    res2 = calculate_diversity_facts(["def f1(): pass", "def f1(): pass"])
    assert res2["active_population_size"] == 2
    assert res2["unique_source_sha_count"] == 1
    assert res2["exact_duplicate_count"] == 1
