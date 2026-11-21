"""Global-controller hooks on the real DualHiFoMoH class (FakeLLM, no subprocess, no API)."""

import json
import re
from pathlib import Path
from typing import ClassVar

import moh as moh_module
import numpy as np
import pytest
from dual_hifo import DualHiFoMoH
from moh import MoH
from omegaconf import OmegaConf

MOH_ROOT = Path(__file__).resolve().parents[1]

STRONG_OPTIMIZER = '''def improve_algorithm(population, utility, language_model, function_format, task):
    # STRONG optimizer
    response = language_model.prompt("writer", "strong heuristic")
    return "idea", response, utility(response, "idea", task)
'''

BASE_HEURISTIC = "def update_edge_distance(d):\n    score = 1.0\n    return d\n"


class FakeLLM:
    batch_size = 2
    model = "fake"
    temperature = 0.7

    def __init__(self):
        self.messages = []

    def prompt(self, expertise, message, temperature=None):
        self.messages.append((expertise, message))
        if "analyze a given function provided as a string" in expertise:
            return "```txt\n2\n```"
        return "ok"

    def prompt_batch(self, expertise, message_batch, temperature=None):
        return [self.prompt(expertise, m, temperature) for m in message_batch]


class FakeGeneratedImprover:
    """Parent-side stand-in for GeneratedImprover (no child process).

    Weak optimizers (seed) produce heuristics worse than the base population;
    STRONG optimizers produce strictly improving heuristics.
    """

    counter = 0
    fail_tasks: ClassVar[set[str]] = set()

    def __init__(self, code, timeout=3600, seed=0):
        self.code = code

    def validate(self):
        return None

    def __call__(self, population, utility, language_model, function_format, task):
        if task in FakeGeneratedImprover.fail_tasks:
            raise RuntimeError(f"generated optimizer crashed on {task}")
        language_model.prompt("writer", "write heuristic")
        FakeGeneratedImprover.counter += 1
        n = FakeGeneratedImprover.counter
        u = 0.5 / n if "STRONG" in self.code else 5.0 + 0.01 * n
        code = f"def update_edge_distance(d):\n    score = {u!r}\n    step = {n}\n    return d\n"
        return "idea", code, utility(code, "idea", task)


def _fake_eval(self, algorithm_str, problem_type, mode="val"):
    match = re.search(r"score = ([0-9.e-]+)", algorithm_str or "")
    return float(match.group(1)) if match else 1e6


def _fake_read_base(self, subtask):
    if self.subtask_pop.get_subtask_size(subtask) == 0:
        self.subtask_pop.save_solution(subtask, "base", BASE_HEURISTIC, 1.0)


def _outer_improver(population, utility, language_model, function_format, task):
    language_model.prompt("writer", "generate optimizer")
    return "strong", STRONG_OPTIMIZER, utility(STRONG_OPTIMIZER, "strong", task)


@pytest.fixture
def make_optimizer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MoH, "evaluate_heuristic", _fake_eval)
    monkeypatch.setattr(MoH, "read_base_algorithm", _fake_read_base)
    monkeypatch.setattr(moh_module, "GeneratedImprover", FakeGeneratedImprover)
    FakeGeneratedImprover.counter = 0
    FakeGeneratedImprover.fail_tasks = set()

    def build(mode, **controller):
        cfg = OmegaConf.create({
            "n_iterations": 4, "seed": 0, "pop_size": 5, "timeout": 5,
            "optimizer_timeout": 5, "max_eval_calls": 100, "seed_max_attempts": 2,
            "isolate_base_solutions": True,
            "problem": {"problem_name": "tsp_gls", "problem_size": [20, 50], "threshold": {"default": 6}},
            "dual_hifo": {"mode": mode, "global_controller": controller},
        })
        np.random.seed(0)
        llm = FakeLLM()
        opt = DualHiFoMoH(cfg, str(MOH_ROOT), llm)
        opt.get_seed()
        return opt, llm
    return build


def _events(opt):
    path = Path(opt.run_logger.output_dir) / "dual_hifo_events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _decisions(opt):
    return [e for e in _events(opt) if e.get("event") == "decision"]


def _step(opt, i):
    opt._cur_iter = i
    return opt.try_improvement(_outer_improver, "previous-driver")


def test_inner_action_runs_incumbent_without_meta_generation(make_optimizer):
    opt, llm = make_optimizer("dual_hifo_global_rule")
    seed_optimizer = opt.improver_str
    sizes_before = {t: opt.subtask_pop.get_subtask_size(t) for t in opt.subtask_list}
    meta_before = opt._hifo_meters["meta"].calls
    n_messages = len(llm.messages)

    accepted, candidate, driver = _step(opt, 0)

    assert (accepted, candidate, driver) == (False, None, "previous-driver")
    assert opt.improver_str == seed_optimizer
    assert opt._hifo_meters["meta"].calls == meta_before
    new_messages = [m for _, m in llm.messages[n_messages:]]
    assert not any("generate optimizer" in m for m in new_messages)
    assert all("Regime: EXPLORE" in m and "GLOBAL_CONTROLLER:INNER_EXPLORE" in m for m in new_messages)
    assert all(opt.subtask_pop.get_subtask_size(t) == sizes_before[t] + 1 for t in opt.subtask_list)
    (decision,) = _decisions(opt)
    assert decision["action"] == "INNER_EXPLORE"
    assert decision["cost_llm_calls"] == len(opt.subtask_list)
    assert decision["accepted"] is False
    assert set(decision["inner_pass"]["subtasks"]) == set(opt.subtask_list)
    assert all(nav.regime_override is None for nav in opt.inner_hifo.values())


def test_rule_controller_escalates_weak_optimizer_to_outer_search(make_optimizer):
    opt, llm = make_optimizer("dual_hifo_global_rule", inner_retry_limit=2)
    results = [_step(opt, i) for i in range(4)]
    actions = [d["action"] for d in _decisions(opt)]
    assert actions == ["INNER_EXPLORE", "INNER_EXPLORE", "OUTER_EXPLORE", "INNER_EXPLOIT"]
    assert results[2][0] is True
    assert opt.improver_str == STRONG_OPTIMIZER
    outer_prompts = [m for _, m in llm.messages if m.startswith("generate optimizer")]
    assert len(outer_prompts) == 1
    assert "Level: OUTER; Regime: EXPLORE" in outer_prompts[0]
    assert "GLOBAL_CONTROLLER:OUTER_EXPLORE" in outer_prompts[0]
    decisions = _decisions(opt)
    assert decisions[2]["cost_llm_calls"] > decisions[0]["cost_llm_calls"]
    assert decisions[2]["q_after"] < decisions[2]["q_before"]
    assert "NEW_OPTIMIZER" in decisions[3]["reasons"]
    assert opt.outer_hifo.regime_override is None


def test_inner_pass_survives_one_failing_subtask(make_optimizer):
    opt, _ = make_optimizer("dual_hifo_global_rule")
    FakeGeneratedImprover.fail_tasks = {"tsp_gls-50"}
    accepted, _, _ = _step(opt, 0)
    assert accepted is False
    (decision,) = _decisions(opt)
    assert decision["inner_pass"]["subtasks"]["tsp_gls-50"] is None
    assert decision["inner_pass"]["subtasks"]["tsp_gls-20"] > 5.0
    assert decision["inner_pass"]["incumbent_remeasured_utility"] is None


def test_bandit_controller_logs_scores_and_persists_state(make_optimizer):
    opt, _ = make_optimizer("dual_hifo_global_bandit", ucb_alpha=0.5)
    for i in range(4):
        _step(opt, i)
    decisions = _decisions(opt)
    assert len(decisions) == 4
    for d in decisions:
        assert set(d["scores"]) == {"INNER_EXPLORE", "INNER_EXPLOIT", "OUTER_EXPLORE", "OUTER_EXPLOIT"}
        assert 0.0 <= d["reward"] <= 1.0
    state = json.loads((Path(opt.run_logger.output_dir) / "global_controller_state.json").read_text())
    assert state["kind"] == "bandit"
    assert sum(s["n"] for s in state["action_stats"].values()) == 4


def test_independent_mode_logs_progress_without_decisions(make_optimizer):
    opt, _ = make_optimizer("dual_hifo_independent")
    for i in range(2):
        _step(opt, i)
    events = _events(opt)
    assert not [e for e in events if e.get("event") == "decision"]
    progress = [e for e in events if e.get("event") == "progress"]
    assert [p["action"] for p in progress] == ["SEED", "OUTER_STEP", "OUTER_STEP"]
    calls = [p["llm_calls"] for p in progress]
    assert calls == sorted(calls) and calls[-1] > calls[0]
    assert all(p["mode"] == "dual_hifo_independent" for p in progress)
    assert progress[0]["elite_quality"] is not None
    assert progress[-1]["meta_calls"] > 0 and progress[-1]["heu_calls"] > 0
