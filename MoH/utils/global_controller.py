"""Cross-level global controller for Dual-HiFo (provider-independent).

One controller observes both Evolutionary Navigators (inner heuristic search and
outer optimizer search) and picks one coordinated action per outer iteration:

- ``INNER_EXPLORE`` / ``INNER_EXPLOIT``: cheap inner pass that re-runs the
  incumbent heuristic-optimizer with the inner regime forced (diagnosis C1:
  heuristics sit in a local optimum, the optimizer is adequate).
- ``OUTER_EXPLORE`` / ``OUTER_EXPLOIT``: standard (expensive) MoH outer step with
  the outer regime forced (diagnosis C2: the optimizer itself is the bottleneck).

Two policies: a hand-designed diagnosis rule and a disjoint LinUCB contextual
bandit rewarded by relative elite-quality gain per LLM call. Lower quality is
better. No generated code is executed and no LLM is called here.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ACTIONS: tuple[str, ...] = ("INNER_EXPLORE", "INNER_EXPLOIT", "OUTER_EXPLORE", "OUTER_EXPLOIT")
CONTROLLER_KINDS = ("rule", "bandit")

FEATURE_NAMES: tuple[str, ...] = (
    "inner_improvement",   # recent relative elite-quality gain
    "outer_improvement",   # recent relative meta-utility gain
    "inner_stagnation",    # decisions since elite quality improved
    "outer_stagnation",    # outer navigator rounds without accepted optimizer
    "inner_diversity",     # 1 - mean inner exact-duplicate rate
    "outer_diversity",     # 1 - outer exact-duplicate rate
    "failed_inner_streak",  # consecutive inner actions without gain
    "action_cost",         # EMA LLM calls of this action (per arm)
    "action_efficacy",     # EMA reward of this action (per arm)
    "bias",
)

_IMPROVEMENT_HALF = 0.01  # 1 % relative gain maps to feature value 0.5
_EPS = 1e-12


def action_level(action: str) -> str:
    return "inner" if action.startswith("INNER") else "outer"


def action_regime(action: str) -> str:
    return action.split("_", 1)[1]


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _squash(value: float, half: float) -> float:
    value = max(0.0, float(value))
    return value / (value + half) if half > 0 else 0.0


def compute_reward(q_before: float | None, q_after: float | None, cost: float, efficiency_scale: float) -> float:
    """Relative elite-quality gain per LLM call, scaled to [0, 1]."""
    if not (_finite(q_before) and _finite(q_after)):
        return 0.0
    gain = max(0.0, q_before - q_after) / max(abs(q_before), _EPS)
    return float(min(1.0, gain / max(float(cost), 1.0) / efficiency_scale))


@dataclass(frozen=True)
class GlobalControllerConfig:
    inner_retry_limit: int = 2
    patience: int = 3
    progress_patience: int = 2
    window: int = 3
    ucb_alpha: float = 0.5
    ridge_lambda: float = 1.0
    cost_scale: float = 100.0
    efficiency_scale: float = 1e-3
    ema: float = 0.3
    warm_start_path: str | None = None

    def __post_init__(self):
        for name in ("inner_retry_limit", "patience", "progress_patience", "window"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not _finite(self.ucb_alpha) or self.ucb_alpha < 0:
            raise ValueError("ucb_alpha must be finite and nonnegative")
        for name in ("ridge_lambda", "cost_scale", "efficiency_scale"):
            value = getattr(self, name)
            if not _finite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not _finite(self.ema) or not (0 < self.ema <= 1):
            raise ValueError("ema must be in (0, 1]")


@dataclass(frozen=True)
class NavigatorSignals:
    """State exported by the two navigators at decision time."""

    inner_duplicate_rate: float = 0.0
    outer_duplicate_rate: float = 0.0
    outer_stagnation: int = 0
    outer_progress: int = 0


@dataclass(frozen=True)
class Decision:
    step: int
    action: str
    reasons: tuple[str, ...]
    features: dict[str, list[float]]
    scores: dict[str, float]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> str:
        return action_level(self.action)

    @property
    def regime(self) -> str:
        return action_regime(self.action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "level": self.level,
            "regime": self.regime,
            "reasons": list(self.reasons),
            "features": self.features,
            "feature_names": list(FEATURE_NAMES),
            "scores": self.scores,
            "raw": self.raw,
        }


class RuleBasedPolicy:
    """Hand-designed cross-level diagnosis: try cheap inner exploration, escalate to outer."""

    def __init__(self, config: GlobalControllerConfig):
        self.config = config

    def select(self, raw: dict[str, Any], contexts: dict[str, list[float]], rng: np.random.Generator):
        if raw["last_action"] is not None and (raw["last_improved"] or raw["last_accepted"]):
            return "INNER_EXPLOIT", ("NEW_OPTIMIZER" if raw["last_accepted"] else "PROGRESS",), {}
        if raw["failed_inner_streak"] >= self.config.inner_retry_limit:
            if raw["outer_progress"] >= self.config.progress_patience:
                return "OUTER_EXPLOIT", ("INNER_EXPLORATION_EXHAUSTED", "OUTER_PROGRESS"), {}
            return "OUTER_EXPLORE", ("INNER_EXPLORATION_EXHAUSTED", "DIAGNOSIS_WEAK_OPTIMIZER"), {}
        return "INNER_EXPLORE", ("TRY_CHEAP_INNER_EXPLORATION", "DIAGNOSIS_LOCAL_OPTIMUM"), {}

    def update(self, action: str, x, reward: float) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


class LinUCBPolicy:
    """Disjoint LinUCB (Li et al., 2010) with per-arm ridge regression."""

    def __init__(self, actions: tuple[str, ...], dim: int, alpha: float, ridge_lambda: float):
        self.actions = tuple(actions)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.A = {a: ridge_lambda * np.eye(self.dim) for a in self.actions}
        self.b = {a: np.zeros(self.dim) for a in self.actions}

    def scores(self, contexts: dict[str, list[float]]) -> dict[str, float]:
        result = {}
        for action in self.actions:
            x = np.asarray(contexts[action], dtype=float)
            a_inv = np.linalg.inv(self.A[action])
            theta = a_inv @ self.b[action]
            width = math.sqrt(max(float(x @ a_inv @ x), 0.0))
            result[action] = float(theta @ x + self.alpha * width)
        return result

    def select(self, raw: dict[str, Any], contexts: dict[str, list[float]], rng: np.random.Generator):
        scores = self.scores(contexts)
        best = max(scores.values())
        ties = [a for a in self.actions if scores[a] >= best - 1e-12]
        action = ties[int(rng.integers(len(ties)))]
        return action, ("LINUCB",), scores

    def update(self, action: str, x, reward: float) -> None:
        x = np.asarray(x, dtype=float)
        self.A[action] += np.outer(x, x)
        self.b[action] += float(reward) * x

    def state_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "A": {a: self.A[a].tolist() for a in self.actions},
            "b": {a: self.b[a].tolist() for a in self.actions},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("dim", -1)) != self.dim:
            raise ValueError("LinUCB state dimension mismatch")
        for action in self.actions:
            matrix = np.asarray(state["A"][action], dtype=float)
            vector = np.asarray(state["b"][action], dtype=float)
            if matrix.shape != (self.dim, self.dim) or vector.shape != (self.dim,):
                raise ValueError(f"LinUCB state shape mismatch for {action}")
            self.A[action] = matrix
            self.b[action] = vector


@dataclass
class ActionStats:
    n: int = 0
    successes: int = 0
    ema_cost: float | None = None
    ema_reward: float | None = None


class GlobalController:
    """Cross-level Foresight: one decision per outer iteration from both levels' state."""

    def __init__(self, kind: str, config: GlobalControllerConfig, seed: int = 0):
        if kind not in CONTROLLER_KINDS:
            raise ValueError(f"Unsupported global controller kind: {kind}")
        self.kind = kind
        self.config = config
        self.rng = np.random.default_rng(seed)
        if kind == "rule":
            self.policy = RuleBasedPolicy(config)
        else:
            self.policy = LinUCBPolicy(ACTIONS, len(FEATURE_NAMES), config.ucb_alpha, config.ridge_lambda)
        self.step = 0
        self.stats = {a: ActionStats() for a in ACTIONS}
        self.quality_history: list[float | None] = []
        self.meta_history: list[float | None] = []
        self.failed_inner_streak = 0
        self.inner_stagnation_rounds = 0
        self.last: dict[str, Any] = {"action": None, "improved": False, "accepted": False}
        if config.warm_start_path:
            self.load(config.warm_start_path)

    # ----- state -----

    def _recent_gain(self, history: list[float | None]) -> float:
        values = [v for v in history[-(self.config.window + 1):] if _finite(v)]
        if len(values) < 2:
            return 0.0
        return max(0.0, values[0] - values[-1]) / max(abs(values[0]), _EPS)

    def _raw(self, signals: NavigatorSignals) -> dict[str, Any]:
        return {
            "last_action": self.last["action"],
            "last_improved": bool(self.last["improved"]),
            "last_accepted": bool(self.last["accepted"]),
            "failed_inner_streak": self.failed_inner_streak,
            "inner_stagnation_rounds": self.inner_stagnation_rounds,
            "outer_stagnation": int(signals.outer_stagnation),
            "outer_progress": int(signals.outer_progress),
            "inner_duplicate_rate": float(signals.inner_duplicate_rate),
            "outer_duplicate_rate": float(signals.outer_duplicate_rate),
            "inner_recent_gain": self._recent_gain(self.quality_history),
            "outer_recent_gain": self._recent_gain(self.meta_history),
        }

    def _contexts(self, raw: dict[str, Any]) -> dict[str, list[float]]:
        cfg = self.config
        shared = [
            _squash(raw["inner_recent_gain"], _IMPROVEMENT_HALF),
            _squash(raw["outer_recent_gain"], _IMPROVEMENT_HALF),
            _squash(raw["inner_stagnation_rounds"], cfg.patience),
            _squash(raw["outer_stagnation"], cfg.patience),
            min(1.0, max(0.0, 1.0 - raw["inner_duplicate_rate"])),
            min(1.0, max(0.0, 1.0 - raw["outer_duplicate_rate"])),
            _squash(raw["failed_inner_streak"], cfg.inner_retry_limit),
        ]
        contexts = {}
        for action in ACTIONS:
            stats = self.stats[action]
            cost = 0.5 if stats.ema_cost is None else _squash(stats.ema_cost, cfg.cost_scale)
            efficacy = 0.0 if stats.ema_reward is None else min(1.0, max(0.0, stats.ema_reward))
            contexts[action] = [float(v) for v in (*shared, cost, efficacy, 1.0)]
        return contexts

    # ----- decision loop -----

    def decide(self, signals: NavigatorSignals) -> Decision:
        raw = self._raw(signals)
        contexts = self._contexts(raw)
        action, reasons, scores = self.policy.select(raw, contexts, self.rng)
        return Decision(self.step, action, tuple(reasons), contexts, dict(scores), raw)

    def feedback(
        self,
        decision: Decision,
        *,
        cost: float,
        q_before: float | None,
        q_after: float | None,
        accepted: bool = False,
        meta_before: float | None = None,
        meta_after: float | None = None,
    ) -> float:
        cfg = self.config
        improved = _finite(q_before) and _finite(q_after) and q_after < q_before - _EPS
        reward = compute_reward(q_before, q_after, cost, cfg.efficiency_scale)
        self.policy.update(decision.action, decision.features[decision.action], reward)

        stats = self.stats[decision.action]
        stats.n += 1
        stats.successes += int(improved)
        cost = max(0.0, float(cost))
        stats.ema_cost = cost if stats.ema_cost is None else (1 - cfg.ema) * stats.ema_cost + cfg.ema * cost
        stats.ema_reward = reward if stats.ema_reward is None else (1 - cfg.ema) * stats.ema_reward + cfg.ema * reward

        if decision.level == "inner":
            self.failed_inner_streak = 0 if improved else self.failed_inner_streak + 1
        else:
            self.failed_inner_streak = 0
        self.inner_stagnation_rounds = 0 if improved else self.inner_stagnation_rounds + 1
        if not self.quality_history:
            self.quality_history.append(q_before)
            self.meta_history.append(meta_before)
        self.quality_history.append(q_after)
        self.meta_history.append(meta_after)
        self.last = {"action": decision.action, "improved": improved, "accepted": bool(accepted)}
        self.step += 1
        return reward

    # ----- persistence (warm start across seeds/runs) -----

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "kind": self.kind,
            "step": self.step,
            "feature_names": list(FEATURE_NAMES),
            "action_stats": {a: asdict(s) for a, s in self.stats.items()},
            "policy": self.policy.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != self.kind:
            raise ValueError(f"Controller state kind {state.get('kind')!r} != {self.kind!r}")
        if list(state.get("feature_names", [])) != list(FEATURE_NAMES):
            raise ValueError("Controller state feature layout mismatch")
        self.policy.load_state_dict(state.get("policy", {}))
        for action, values in state.get("action_stats", {}).items():
            if action in self.stats:
                self.stats[action] = ActionStats(**values)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), allow_nan=False, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        logger.info("Global controller warm-started from %s", path)
