"""Offline tests for MoH-aligned Dual-HiFo behavior (no real LLM)."""

import pytest

from utils.dual_hifo import GuidedLLM, HiFoConfig, HiFoController, structure_hash


class FakeLLM:
    batch_size = 2
    model = "offline"
    temperature = 0.7

    def __init__(self):
        self.messages = []

    def prompt(self, expertise, message, temperature=None):
        self.messages.append((expertise, message, temperature))
        return "ok"

    def prompt_batch(self, expertise, message_batch, temperature=None):
        self.messages.extend((expertise, message, temperature) for message in message_batch)
        return ["ok"] * len(message_batch)


CODE = """def improve_algorithm(population, utility, language_model, function_format, task):
    for _ in range(2):
        candidate = population.get_random_solution(task)
        response = language_model.prompt('x', candidate['best_sol'])
    return ('idea', response, utility(response))
"""


def test_modes_do_not_implicitly_enable_both_levels():
    assert not HiFoConfig(mode="full_moh").inner_enabled
    assert not HiFoConfig(mode="full_moh").outer_enabled
    assert HiFoConfig(mode="inner_hifo").inner_enabled
    assert not HiFoConfig(mode="inner_hifo").outer_enabled
    assert HiFoConfig(mode="dual_hifo_cross_level").cross_level_enabled
    assert not HiFoConfig(mode="dual_hifo_independent").cross_level_enabled
    with pytest.raises(ValueError):
        HiFoConfig(mode="typo")


def test_fitness_minimizes_gap_and_regime_transitions():
    c = HiFoController("inner", HiFoConfig(stagnation_patience=2, progress_patience=2, duplicate_threshold=1.0))
    assert c.observe(CODE, 10.0)["best_utility"] == 10.0
    assert c.observe(CODE + "#better", 9.0)["regime"] == "EXPLOIT"
    assert c.observe(CODE + "#bad", 9.5)["best_utility"] == 9.0
    assert c.observe(CODE + "#bad2", 1e6)["regime"] == "EXPLORE"
    assert c.snapshot()["semantic_diversity"] is None


def test_exact_ast_duplicate_is_not_semantic_diversity():
    assert structure_hash("x=1\n") == structure_hash("x = 1 # whitespace\n")
    c = HiFoController("outer", HiFoConfig(stagnation_patience=10, duplicate_threshold=0.5))
    c.observe("x=1", 10)
    state = c.observe("x = 1 # same AST", None)
    assert state["exact_duplicate_rate"] == 0.5
    assert state["regime"] == "EXPLORE"
    assert state["semantic_diversity"] is None


def test_hindsight_usage_credit_pruning_and_format_compatibility():
    c = HiFoController("outer", HiFoConfig(pool_size=2, insight_top_k=1))
    c.observe(CODE, 10.0)
    assert c.insights
    base = FakeLLM()
    wrapped = GuidedLLM(base, c)
    assert wrapped.batch_size == 2
    wrapped.prompt("writer", "write optimizer")
    assert "[DUAL-HIFO" in base.messages[-1][1]
    assert "Previously successful abstract mechanisms" in base.messages[-1][1]
    wrapped.prompt_batch("writer", ["a", "b"], temperature=0.9)
    assert len(base.messages) == 3
    assert all(t == 0.9 for _, _, t in base.messages[1:])
    c.observe("def optimize(): return 0", 9.0)
    assert len(c.insights) <= 2
    assert all(-1 <= tip.effectiveness <= 1 for tip in c.insights.values())


def test_validator_prompt_is_not_modified():
    llm = FakeLLM()
    controller = HiFoController("outer", HiFoConfig())
    guided = GuidedLLM(llm, controller)
    directive = "Please count the iterations in this code"
    expertise = "You will act as a professional coder and analyze a given function provided as a string."
    guided.prompt(expertise, directive)
    assert llm.messages[-1][1] == directive
    assert controller.logical_prompt_requests == 0


def test_independent_mode_does_not_inject_cross_level_context():
    llm = FakeLLM()
    ctrl = HiFoController("outer", HiFoConfig(mode="dual_hifo_independent"))
    guided = GuidedLLM(llm, ctrl)
    guided.prompt("writer", "generate code")
    assert "Completed inner-search evidence" not in llm.messages[0][1]


def test_controller_snapshot_does_not_execute_generated_source(tmp_path):
    marker = tmp_path / "should_not_exist"
    malicious = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    c = HiFoController("inner", HiFoConfig())
    c.observe(malicious, 3.0)
    assert not marker.exists()


def test_config_validation():
    with pytest.raises(ValueError):
        HiFoConfig(improvement_epsilon=-1)
    with pytest.raises(ValueError):
        HiFoConfig(credit_ema=0)
    with pytest.raises(ValueError):
        HiFoConfig(pool_size=0)
