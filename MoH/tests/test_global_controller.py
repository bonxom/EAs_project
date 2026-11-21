"""Offline tests for the cross-level global controller (no real LLM)."""

import json

import numpy as np
import pytest
from utils.dual_hifo import GuidedLLM, HiFoConfig, HiFoController, LLMCallMeter
from utils.global_controller import (
    ACTIONS,
    FEATURE_NAMES,
    GlobalController,
    GlobalControllerConfig,
    LinUCBPolicy,
    NavigatorSignals,
    compute_reward,
)


class FakeLLM:
    batch_size = 3
    model = "fake"
    temperature = 0.7

    def __init__(self):
        self.messages = []
        self.budget = "sentinel-budget"

    def prompt(self, expertise, message, temperature=None):
        self.messages.append((expertise, message, temperature))
        return "ok"

    def prompt_batch(self, expertise, message_batch, temperature=None):
        self.messages.extend((expertise, m, temperature) for m in message_batch)
        return ["ok"] * len(message_batch)


# ---------------------------------------------------------------------------
# Modes and configuration
# ---------------------------------------------------------------------------

def test_global_modes_enable_both_levels_without_prompt_injection():
    rule = HiFoConfig(mode="dual_hifo_global_rule")
    bandit = HiFoConfig(mode="dual_hifo_global_bandit")
    for cfg in (rule, bandit):
        assert cfg.inner_enabled and cfg.outer_enabled
        assert not cfg.cross_level_enabled
    assert rule.global_controller == "rule"
    assert bandit.global_controller == "bandit"
    for mode in ("full_moh", "inner_hifo", "outer_hifo", "dual_hifo_independent", "dual_hifo_cross_level"):
        assert HiFoConfig(mode=mode).global_controller is None


def test_global_controller_config_validation():
    GlobalControllerConfig()
    with pytest.raises(ValueError):
        GlobalControllerConfig(inner_retry_limit=0)
    with pytest.raises(ValueError):
        GlobalControllerConfig(ucb_alpha=-0.1)
    with pytest.raises(ValueError):
        GlobalControllerConfig(ridge_lambda=0)
    with pytest.raises(ValueError):
        GlobalControllerConfig(efficiency_scale=0)
    with pytest.raises(ValueError):
        GlobalControllerConfig(ema=0)
    with pytest.raises(ValueError):
        GlobalController("typo", GlobalControllerConfig())


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def test_reward_is_cost_normalised_and_bounded():
    assert compute_reward(1.0, 1.0, 10, 1e-3) == 0.0
    assert compute_reward(1.0, 1.2, 10, 1e-3) == 0.0  # never negative
    assert compute_reward(None, 0.5, 10, 1e-3) == 0.0
    assert compute_reward(1.0, None, 10, 1e-3) == 0.0
    cheap = compute_reward(1.0, 0.99, 10, 1e-3)
    costly = compute_reward(1.0, 0.99, 100, 1e-3)
    assert 0.0 < costly < cheap <= 1.0
    assert compute_reward(1.0, 0.0, 1, 1e-3) == 1.0


# ---------------------------------------------------------------------------
# Rule-based global policy: C1 / C2 diagnosis
# ---------------------------------------------------------------------------

def _feedback(ctrl, decision, q_before, q_after, accepted=False, cost=10):
    return ctrl.feedback(decision, cost=cost, q_before=q_before, q_after=q_after,
                         accepted=accepted, meta_before=None, meta_after=None)


def test_rule_policy_tries_cheap_inner_exploration_first():
    ctrl = GlobalController("rule", GlobalControllerConfig(inner_retry_limit=2), seed=0)
    d = ctrl.decide(NavigatorSignals())
    assert d.action == "INNER_EXPLORE"
    assert d.level == "inner" and d.regime == "EXPLORE"
    assert "DIAGNOSIS_LOCAL_OPTIMUM" in d.reasons


def test_rule_policy_exploits_after_progress():
    ctrl = GlobalController("rule", GlobalControllerConfig(), seed=0)
    d = ctrl.decide(NavigatorSignals())
    _feedback(ctrl, d, 1.0, 0.9)
    d2 = ctrl.decide(NavigatorSignals())
    assert d2.action == "INNER_EXPLOIT"
    assert "PROGRESS" in d2.reasons


def test_rule_policy_escalates_to_outer_after_failed_inner_exploration():
    ctrl = GlobalController("rule", GlobalControllerConfig(inner_retry_limit=2), seed=0)
    for _ in range(2):
        d = ctrl.decide(NavigatorSignals())
        assert d.level == "inner"
        _feedback(ctrl, d, 1.0, 1.0)
    d = ctrl.decide(NavigatorSignals(outer_progress=0))
    assert d.action == "OUTER_EXPLORE"
    assert "DIAGNOSIS_WEAK_OPTIMIZER" in d.reasons
    # Outer search progressing on its own -> refine optimizer instead.
    ctrl2 = GlobalController("rule", GlobalControllerConfig(inner_retry_limit=1, progress_patience=2), seed=0)
    d = ctrl2.decide(NavigatorSignals())
    _feedback(ctrl2, d, 1.0, 1.0)
    assert ctrl2.decide(NavigatorSignals(outer_progress=2)).action == "OUTER_EXPLOIT"


def test_rule_policy_exploits_new_optimizer_after_acceptance():
    ctrl = GlobalController("rule", GlobalControllerConfig(inner_retry_limit=1), seed=0)
    d = ctrl.decide(NavigatorSignals())
    _feedback(ctrl, d, 1.0, 1.0)
    d = ctrl.decide(NavigatorSignals())
    assert d.level == "outer"
    _feedback(ctrl, d, 1.0, 1.0, accepted=True, cost=500)
    d = ctrl.decide(NavigatorSignals())
    assert d.action == "INNER_EXPLOIT"
    assert "NEW_OPTIMIZER" in d.reasons


def test_decision_features_cover_cross_level_state_and_action_history():
    ctrl = GlobalController("bandit", GlobalControllerConfig(), seed=0)
    d = ctrl.decide(NavigatorSignals(inner_duplicate_rate=0.25, outer_duplicate_rate=0.5, outer_stagnation=3))
    assert set(d.features) == set(ACTIONS)
    for vector in d.features.values():
        assert len(vector) == len(FEATURE_NAMES)
        assert all(0.0 <= v <= 1.0 for v in vector)
    named = dict(zip(FEATURE_NAMES, d.features["INNER_EXPLORE"]))
    assert named["inner_diversity"] == pytest.approx(0.75)
    assert named["outer_diversity"] == pytest.approx(0.5)
    assert named["outer_stagnation"] == pytest.approx(3 / (3 + 3))
    assert named["bias"] == 1.0
    _feedback(ctrl, d, 1.0, 0.9, cost=100)
    d2 = ctrl.decide(NavigatorSignals())
    chosen = dict(zip(FEATURE_NAMES, d2.features[d.action]))
    other = next(a for a in ACTIONS if a != d.action)
    unseen = dict(zip(FEATURE_NAMES, d2.features[other]))
    assert chosen["action_cost"] == pytest.approx(100 / (100 + 100))
    assert chosen["action_efficacy"] > 0
    assert unseen["action_cost"] == 0.5 and unseen["action_efficacy"] == 0.0
    assert json.dumps(d2.to_dict(), allow_nan=False)


# ---------------------------------------------------------------------------
# LinUCB
# ---------------------------------------------------------------------------

def test_linucb_learns_context_dependent_arm():
    policy = LinUCBPolicy(("a", "b"), dim=3, alpha=0.1, ridge_lambda=1.0)
    rng = np.random.default_rng(0)
    x1, x2 = [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]
    for _ in range(30):
        policy.update("a", x1, 1.0)
        policy.update("b", x1, 0.0)
        policy.update("a", x2, 0.0)
        policy.update("b", x2, 1.0)
    assert policy.select({}, {"a": x1, "b": x1}, rng)[0] == "a"
    assert policy.select({}, {"a": x2, "b": x2}, rng)[0] == "b"


def test_bandit_controller_is_deterministic_under_seed():
    def run(seed):
        ctrl = GlobalController("bandit", GlobalControllerConfig(), seed=seed)
        actions = []
        for i in range(6):
            d = ctrl.decide(NavigatorSignals())
            actions.append(d.action)
            _feedback(ctrl, d, 1.0, 1.0 - 0.01 * (i % 2))
        return actions

    assert run(3) == run(3)


def test_bandit_state_roundtrip_reproduces_choices(tmp_path):
    ctrl = GlobalController("bandit", GlobalControllerConfig(), seed=1)
    for i in range(5):
        d = ctrl.decide(NavigatorSignals())
        _feedback(ctrl, d, 1.0, 0.95 if d.level == "inner" else 1.0)
    path = tmp_path / "state.json"
    ctrl.save(path)
    payload = json.loads(path.read_text())
    assert payload["kind"] == "bandit"
    warm = GlobalController("bandit", GlobalControllerConfig(warm_start_path=str(path)), seed=1)
    probe = NavigatorSignals(inner_duplicate_rate=0.1)
    assert warm.policy.scores(ctrl._contexts(ctrl._raw(probe))) == pytest.approx(
        ctrl.policy.scores(ctrl._contexts(ctrl._raw(probe)))
    )


class ToyTwoCauseEnv:
    """Synthetic C1/C2 world.

    Each new optimizer unlocks a few cheap inner gains (C1); once exhausted only
    an expensive outer step (new optimizer, C2) lifts the heuristic again.
    """

    def __init__(self):
        self.q = 1.0
        self.inner_left = 3

    def step(self, action):
        if action.startswith("INNER"):
            if self.inner_left > 0:
                self.q *= 0.97
                self.inner_left -= 1
            return 50, False
        self.q *= 0.95
        self.inner_left = 3
        return 500, True


def _calls_to_target(policy_fn, target=0.6, budget=40000):
    env = ToyTwoCauseEnv()
    calls = 0
    while env.q > target and calls < budget:
        action, finish = policy_fn()
        q_before = env.q
        cost, accepted = env.step(action)
        calls += cost
        finish(cost, q_before, env.q, accepted)
    return calls


def _controller_fn(kind, seed):
    ctrl = GlobalController(kind, GlobalControllerConfig(), seed=seed)
    holder = {}

    def policy_fn():
        holder["d"] = ctrl.decide(NavigatorSignals())

        def finish(cost, q_before, q_after, accepted):
            ctrl.feedback(holder["d"], cost=cost, q_before=q_before, q_after=q_after,
                          accepted=accepted, meta_before=None, meta_after=None)
        return holder["d"].action, finish
    return policy_fn


def test_global_controllers_need_fewer_calls_than_always_outer_in_toy_world():
    always_outer = _calls_to_target(lambda: ("OUTER_EXPLORE", lambda *a: None))
    rule = _calls_to_target(_controller_fn("rule", 0))
    assert rule < always_outer
    for seed in range(5):
        assert _calls_to_target(_controller_fn("bandit", seed)) < always_outer


# ---------------------------------------------------------------------------
# Navigator override and call meter
# ---------------------------------------------------------------------------

def test_navigator_regime_override_only_changes_guidance():
    nav = HiFoController("inner", HiFoConfig())
    nav.observe("def f():\n    return 1", 1.0)
    local = nav.regime
    nav.set_override("EXPLORE", "INNER_EXPLORE")
    llm = FakeLLM()
    GuidedLLM(llm, nav).prompt("writer", "make heuristic")
    text = llm.messages[-1][1]
    assert "Regime: EXPLORE" in text
    assert "GLOBAL_CONTROLLER:INNER_EXPLORE" in text
    snap = nav.snapshot()
    assert snap["regime"] == local
    assert snap["regime_override"] == "EXPLORE"
    nav.clear_override()
    assert nav.snapshot()["regime_override"] is None
    GuidedLLM(llm, nav).prompt("writer", "make heuristic")
    assert "GLOBAL_CONTROLLER" not in llm.messages[-1][1]
    with pytest.raises(ValueError):
        nav.set_override("SIDEWAYS", "x")


def test_llm_call_meter_counts_and_forwards():
    base = FakeLLM()
    meter = LLMCallMeter(base)
    assert meter.batch_size == 3 and meter.model == "fake"
    assert meter.budget == "sentinel-budget"
    meter.prompt("e", "m")
    meter.prompt("e", "m", 0.2)
    meter.prompt_batch("e", ["a", "b", "c"], temperature=0.9)
    assert meter.calls == 5
    assert base.messages[1][2] == 0.2
    assert base.messages[-1][2] == 0.9
    assert meter.messages is base.messages
