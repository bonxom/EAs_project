"""Dual-HiFo controllers for the original MoH training pipeline.

No generated code is executed here. GeneratedImprover still runs in a child;
LLM calls and evaluation remain parent-side services. Lower utility is better.

Research policy v1: deterministic AST mechanism extraction (no hidden LLM
calls); exact structural duplication, NOT semantic diversity; deterministic
Foresight directives and credit assignment. Equations are implementation
choices, not claimed as equations from the HiFo paper.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from utils.global_controller import (
    GlobalController,
    GlobalControllerConfig,
    NavigatorSignals,
)

logger = logging.getLogger(__name__)


# mode -> (inner navigator, outer navigator, cross-level prompt evidence, global controller)
MODES = {
    "full_moh": (False, False, False, None),
    "inner_hifo": (True, False, False, None),
    "outer_hifo": (False, True, False, None),
    "dual_hifo_independent": (True, True, False, None),
    "dual_hifo_cross_level": (True, True, True, None),
    "dual_hifo_global_rule": (True, True, False, "rule"),
    "dual_hifo_global_bandit": (True, True, False, "bandit"),
}
REGIMES = ("EXPLORE", "EXPLOIT", "BALANCE")


def _finite_or_none(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


@dataclass(frozen=True)
class HiFoConfig:
    mode: str = "dual_hifo_independent"
    stagnation_patience: int = 3
    progress_patience: int = 2
    duplicate_threshold: float = 0.50
    diversity_window: int = 8
    improvement_epsilon: float = 1e-9
    pool_size: int = 30
    insight_top_k: int = 3
    usage_weight: float = 0.10
    recency_weight: float = 0.20
    credit_ema: float = 0.30

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unsupported Dual-HiFo mode: {self.mode}")
        for name in ("stagnation_patience", "progress_patience", "diversity_window", "pool_size", "insight_top_k"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("improvement_epsilon", "usage_weight", "recency_weight"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not (0 <= self.duplicate_threshold <= 1):
            raise ValueError("duplicate_threshold must be in [0,1]")
        if not (0 < self.credit_ema <= 1):
            raise ValueError("credit_ema must be in (0,1]")

    @property
    def inner_enabled(self) -> bool:
        return MODES[self.mode][0]

    @property
    def outer_enabled(self) -> bool:
        return MODES[self.mode][1]

    @property
    def cross_level_enabled(self) -> bool:
        return MODES[self.mode][2]

    @property
    def global_controller(self) -> str | None:
        return MODES[self.mode][3]


def structure_hash(source: str) -> str:
    """Exact AST structure fingerprint; not a semantic similarity metric."""
    try:
        tree = ast.parse(source)
        normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    except (SyntaxError, TypeError, ValueError):
        normalized = str(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _design_mechanisms(source: str, level: str) -> tuple[str, ...]:
    """Conservative AST mechanism labels, not an LLM-generated summary."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return ()
    nodes = list(ast.walk(tree))
    calls = set()
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute):
            calls.add(fn.attr)
        elif isinstance(fn, ast.Name):
            calls.add(fn.id)
    mechanisms = []
    if "prompt_batch" in calls:
        mechanisms.append("Use batched candidate generation to compare alternatives")
    if "get_random_solution" in calls:
        mechanisms.append("Diversify by sampling from the solution population")
    if "get_best_solution" in calls or "get_solution_by_index" in calls:
        mechanisms.append("Refine high-ranked solutions from the population")
    if "min" in calls or "sorted" in calls or "nsmallest" in calls:
        mechanisms.append("Select candidates using minimum validated objective")
    if "max" in calls and level == "inner":
        mechanisms.append("Check objective direction when aggregating candidates")
    if any(isinstance(node, ast.If) for node in nodes):
        mechanisms.append("Use conditional decisions to adapt search behavior")
    if any(isinstance(node, (ast.For, ast.While)) for node in nodes):
        mechanisms.append("Iteratively evaluate a bounded collection of candidates")
    if level == "inner" and "copy" in calls:
        mechanisms.append("Preserve the input distance matrix by copying it")
    return tuple(mechanisms)


@dataclass
class Insight:
    insight_id: str
    text: str
    created_step: int
    last_used_step: int
    uses: int = 0
    effectiveness: float = 0.0

    def credit(self, step: int, cfg: HiFoConfig) -> float:
        return (
            self.effectiveness
            - cfg.usage_weight * math.log1p(self.uses)
            + cfg.recency_weight * math.exp(-max(0, step - self.last_used_step) / 3)
        )


@dataclass
class HiFoController:
    """One controller for one level and task; no states leaked across tasks."""

    level: str
    config: HiFoConfig
    best_utility: float | None = None
    progress_count: int = 0
    stagnation_count: int = 0
    step: int = 0
    regime: str = "BALANCE"
    reasons: tuple[str, ...] = ()
    insights: dict[str, Insight] = field(default_factory=dict)
    recent_hashes: deque[str] = field(default_factory=deque)
    selected_insight_ids: list[str] = field(default_factory=list)
    logical_prompt_requests: int = 0
    regime_override: str | None = None
    override_reason: str = ""

    def __post_init__(self):
        if self.level not in ("inner", "outer"):
            raise ValueError("level must be inner or outer")
        self.recent_hashes = deque(maxlen=self.config.diversity_window)

    def set_override(self, regime: str, reason: str) -> None:
        """Let a global controller force the prompt regime; local state tracking is unchanged."""
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        self.regime_override = regime
        self.override_reason = str(reason)

    def clear_override(self) -> None:
        self.regime_override = None
        self.override_reason = ""

    @property
    def exact_duplicate_rate(self) -> float:
        values = list(self.recent_hashes)
        return (len(values) - len(set(values))) / len(values) if values else 0.0

    def _choose_regime(self) -> None:
        reasons = []
        if self.stagnation_count >= self.config.stagnation_patience:
            reasons.append("STAGNATION")
        if len(self.recent_hashes) >= 2 and self.exact_duplicate_rate >= self.config.duplicate_threshold:
            reasons.append("EXACT_DUPLICATES")
        if reasons:
            self.regime = "EXPLORE"
        elif self.progress_count >= self.config.progress_patience:
            self.regime = "EXPLOIT"
            reasons.append("CONSECUTIVE_PROGRESS")
        else:
            self.regime = "BALANCE"
            reasons.append("STABLE_OR_INSUFFICIENT_EVIDENCE")
        self.reasons = tuple(reasons)

    def _reward_selected(self, reward: float) -> None:
        for insight_id in set(self.selected_insight_ids):
            insight = self.insights.get(insight_id)
            if insight is not None:
                a = self.config.credit_ema
                insight.effectiveness = max(-1.0, min(1.0, (1 - a) * insight.effectiveness + a * reward))
        self.selected_insight_ids.clear()

    def observe(self, code: str | None, utility: float | None) -> dict[str, Any]:
        """Observe verified evaluation. Failed/invalid utility never counts as success."""
        self.step += 1
        previous = self.best_utility
        valid = (type(utility) in (float, int) and math.isfinite(utility) and 0 <= utility < 1e6)
        improvement = 0.0
        if valid and (previous is None or utility < previous - self.config.improvement_epsilon):
            improvement = 1.0 if previous is None else min(1.0, (previous - utility) / max(1.0, abs(previous)))
            self.best_utility = float(utility)
            self.progress_count += 1
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
            self.progress_count = 0
        self._reward_selected(improvement if improvement > 0 else -0.1)
        if code:
            self.recent_hashes.append(structure_hash(code))
        if improvement > 0 and code:
            for label in _design_mechanisms(code, self.level):
                insight_id = hashlib.sha256(f"{self.level}:{label}".encode()).hexdigest()[:16]
                if insight_id not in self.insights:
                    self.insights[insight_id] = Insight(insight_id, label, self.step, self.step)
            self._prune()
        self._choose_regime()
        return self.snapshot()

    def _prune(self) -> None:
        while len(self.insights) > self.config.pool_size:
            worst_id = min(
                self.insights,
                key=lambda key: (self.insights[key].credit(self.step, self.config), self.insights[key].created_step, key),
            )
            del self.insights[worst_id]

    def _retrieve(self) -> list[Insight]:
        top = sorted(self.insights.values(), key=lambda item: (-item.credit(self.step, self.config), item.insight_id))
        chosen = top[: self.config.insight_top_k]
        for item in chosen:
            item.uses += 1
            item.last_used_step = self.step
            self.selected_insight_ids.append(item.insight_id)
        return chosen

    def guidance(self, *, cross_level_context: str = "") -> str:
        """Construct prompt guidance and record insight usage (no provider call)."""
        self.logical_prompt_requests += 1
        instructions = {
            "EXPLORE": "Make a meaningful STRUCTURAL change rather than only adjusting constants; consider a distinct search strategy.",
            "EXPLOIT": "Refine and simplify proven mechanisms, tuning only relevant components while preserving correctness.",
            "BALANCE": "Balance novel mechanisms with reuse of validated, effective ideas.",
        }
        regime = self.regime_override or self.regime
        if self.regime_override:
            reasons = f"GLOBAL_CONTROLLER:{self.override_reason}"
        else:
            reasons = ",".join(self.reasons) or "INITIAL"
        sections = [
            "[DUAL-HIFO RESEARCH GUIDANCE; existing task/output constraints still apply]",
            f"Level: {self.level.upper()}; Regime: {regime}; Reasons: {reasons}.",
            instructions[regime],
            "Smaller validation utility is better; never use unverified or test-set scores for selection.",
        ]
        chosen = self._retrieve()
        if chosen:
            sections.append("Previously successful abstract mechanisms (adapt, do not paste code):")
            sections.extend(f"- {item.text}" for item in chosen)
        if cross_level_context:
            sections.append("Completed inner-search evidence (not direct optimizer fitness):")
            sections.append(cross_level_context)
        sections.append("[/DUAL-HIFO RESEARCH GUIDANCE]")
        return "\n".join(sections)

    def snapshot(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "step": self.step,
            "best_utility": self.best_utility,
            "progress_count": self.progress_count,
            "stagnation_count": self.stagnation_count,
            "regime": self.regime,
            "regime_override": self.regime_override,
            "reason_codes": list(self.reasons),
            "exact_duplicate_rate": self.exact_duplicate_rate,
            "semantic_diversity": None,  # do NOT conflate AST duplicates with semantic diversity
            "active_insights": len(self.insights),
            "logical_prompt_requests": self.logical_prompt_requests,
        }


class GuidedLLM:
    """Parent-side prompt wrapper; preserves BaseClient's public contract."""

    def __init__(self, base: Any, controller: HiFoController, context_factory=None):
        self.base = base
        self.controller = controller
        self.context_factory = context_factory
        for name in ("batch_size", "model", "temperature"):
            if hasattr(base, name):
                setattr(self, name, getattr(base, name))

    def _message(self, expertise: str, message: str) -> str:
        # A validator requests analysis of code, not evolutionary generation.
        if "analyze a given function provided as a string" in expertise:
            return message
        cross = self.context_factory() if self.context_factory else ""
        return f"{message}\n\n{self.controller.guidance(cross_level_context=cross)}"

    def prompt(self, expertise: str, message: str, temperature=None):
        new_message = self._message(expertise, message)
        if temperature is None:
            return self.base.prompt(expertise, new_message)
        return self.base.prompt(expertise, new_message, temperature=temperature)

    def prompt_batch(self, expertise: str, message_batch: list[str], temperature=None):
        new_messages = [self._message(expertise, m) for m in message_batch]
        if temperature is None:
            return self.base.prompt_batch(expertise, new_messages)
        return self.base.prompt_batch(expertise, new_messages, temperature=temperature)


class LLMCallMeter:
    """Transparent logical-prompt counter (prompt = 1, prompt_batch = len(batch)).

    Forwards every other attribute (``budget``, ``batch_size``, ...) to the wrapped
    client, so provider-budget checks and generated-optimizer services are unchanged.
    """

    def __init__(self, base: Any):
        self.base = base
        self.calls = 0
        self._lock = Lock()

    def __getattr__(self, name: str):
        if name in ("base", "calls", "_lock"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def _count(self, n: int) -> None:
        with self._lock:
            self.calls += n

    def prompt(self, expertise: str, message: str, temperature=None):
        self._count(1)
        if temperature is None:
            return self.base.prompt(expertise, message)
        return self.base.prompt(expertise, message, temperature=temperature)

    def prompt_batch(self, expertise: str, message_batch: list[str], temperature=None):
        self._count(len(message_batch))
        if temperature is None:
            return self.base.prompt_batch(expertise, message_batch)
        return self.base.prompt_batch(expertise, message_batch, temperature=temperature)


class DualHiFoMoHMixin:
    """Mix into the original MoH class, rather than porting the old mini-MoH."""

    def init_dual_hifo(
        self,
        config: HiFoConfig,
        controller_config: GlobalControllerConfig | None = None,
        seed: int = 0,
    ):
        self.hifo_config = config
        self._hifo_seed = int(seed)
        self.outer_hifo = HiFoController("outer", config) if config.outer_enabled else None
        self.inner_hifo = {
            task: HiFoController("inner", config) for task in self.subtask_list
        } if config.inner_enabled else {}
        self.hifo_inner_state_by_task: dict[str, dict[str, Any]] = {}
        # full_moh keeps exact MoH parity: no meters, no telemetry.
        self._hifo_meters: dict[str, LLMCallMeter] = {}
        if config.mode != "full_moh":
            self.heu_llm = LLMCallMeter(self.heu_llm)
            self.meta_llm = LLMCallMeter(self.meta_llm)
            self._hifo_meters = {"heu": self.heu_llm, "meta": self.meta_llm}
        self.global_controller = None
        if config.global_controller is not None:
            self.global_controller = GlobalController(
                config.global_controller, controller_config or GlobalControllerConfig(), seed=self._hifo_seed,
            )

    # ------------------------------------------------------------------
    # Cross-level measurements (parent-side, no generated code executed)
    # ------------------------------------------------------------------

    def _llm_calls(self) -> int:
        return sum(meter.calls for meter in self._hifo_meters.values())

    def _elite_quality(self) -> float | None:
        """Size-weighted mean of the best validated utility per training subtask."""
        pop = getattr(self, "subtask_pop", None)
        if pop is None:
            return None
        best = []
        for task in self.subtask_list:
            try:
                entries = pop.get_population(task)
            except (KeyError, AttributeError):
                return None
            valid = [e.get("utility") for e in entries if _finite_or_none(e.get("utility")) is not None]
            valid = [u for u in valid if 0 <= u < 1e6]
            if not valid:
                return None
            best.append(min(valid))
        weights = getattr(self, "size_weights", None)
        weights = [float(w) for w in weights] if weights is not None and len(weights) == len(best) else [1.0] * len(best)
        if sum(weights) <= 0:
            return None
        return sum(w * v for w, v in zip(weights, best)) / sum(weights)

    def _navigator_signals(self) -> NavigatorSignals:
        inner = list(self.inner_hifo.values())
        outer = self.outer_hifo
        return NavigatorSignals(
            inner_duplicate_rate=sum(c.exact_duplicate_rate for c in inner) / len(inner) if inner else 0.0,
            outer_duplicate_rate=outer.exact_duplicate_rate if outer else 0.0,
            outer_stagnation=outer.stagnation_count if outer else 0,
            outer_progress=outer.progress_count if outer else 0,
        )

    def _hifo_progress(self, action: str) -> None:
        if self.hifo_config.mode == "full_moh":
            return
        self._hifo_record("global", "all", {
            "event": "progress",
            "mode": self.hifo_config.mode,
            "seed": self._hifo_seed,
            "action": action,
            "llm_calls": self._llm_calls(),
            "heu_calls": self._hifo_meters["heu"].calls,
            "meta_calls": self._hifo_meters["meta"].calls,
            "eval_calls": int(getattr(self, "_total_eval_calls", 0)),
            "elite_quality": self._elite_quality(),
            "meta_utility": _finite_or_none(getattr(self, "meta_utility_val", None)),
        })

    def _hifo_record(self, level: str, task: str, event: dict[str, Any]):
        output = getattr(getattr(self, "run_logger", None), "output_dir", None)
        if not output:
            return
        filename = Path(output) / "dual_hifo_events.jsonl"
        filename.parent.mkdir(parents=True, exist_ok=True)
        payload = {"level": level, "task": task, "outer_iteration": getattr(self, "_cur_iter", -1), **event}
        try:
            with filename.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Dual-HiFo telemetry write failed (search continues): %s", exc)

    def evaluate_heuristic(self, algorithm_str, problem_type, mode="val"):
        controller = self.inner_hifo.get(problem_type)
        if controller is None or mode != "val":
            return super().evaluate_heuristic(algorithm_str, problem_type, mode)
        score = super().evaluate_heuristic(algorithm_str, problem_type, mode)
        snapshot = controller.observe(algorithm_str, score)
        self._hifo_record("inner", problem_type, {"event": "evaluated", "utility": score, "state": snapshot})
        self.hifo_inner_state_by_task[problem_type] = snapshot
        return score

    def get_improver(self, improve_str, subtask_str, subtask):
        controller = self.inner_hifo.get(subtask)
        if controller is None:
            return super().get_improver(improve_str, subtask_str, subtask)
        original = self.heu_llm
        self.heu_llm = GuidedLLM(original, controller)
        try:
            return super().get_improver(improve_str, subtask_str, subtask)
        finally:
            self.heu_llm = original

    def _cross_level_context(self) -> str:
        if not self.hifo_config.cross_level_enabled:
            return ""
        if not self.inner_hifo:
            return ""

        # 1. Collect top verified mechanisms across inner subtasks
        mechanism_tasks: dict[str, list[str]] = {}
        mechanism_scores: dict[str, float] = {}

        for task, controller in sorted(self.inner_hifo.items()):
            for insight in controller.insights.values():
                score = insight.credit(controller.step, self.hifo_config)
                if insight.effectiveness >= 0:
                    text = insight.text
                    if text not in mechanism_tasks:
                        mechanism_tasks[text] = []
                        mechanism_scores[text] = score
                    else:
                        mechanism_scores[text] = max(mechanism_scores[text], score)
                    if task not in mechanism_tasks[text]:
                        mechanism_tasks[text].append(task)

        # 2. Collect subtask search states
        subtask_summaries: list[str] = []
        for task, controller in sorted(self.inner_hifo.items()):
            state = self.hifo_inner_state_by_task.get(task)
            regime = state["regime"] if state else controller.regime
            best_u = state["best_utility"] if state else controller.best_utility
            stag = state["stagnation_count"] if state else controller.stagnation_count

            if best_u is not None and math.isfinite(best_u):
                u_str = f"{best_u:.6f}"
            else:
                u_str = "None"
            subtask_summaries.append(f"{task}: best_utility={u_str}, regime={regime}, stagnation={stag}")

        lines = ["[Inner Search Evidence & Distilled Mechanisms]"]
        if mechanism_tasks:
            lines.append("Validated heuristic mechanisms across subtasks:")
            sorted_mechs = sorted(
                mechanism_tasks.keys(),
                key=lambda m: (len(mechanism_tasks[m]), mechanism_scores.get(m, 0.0)),
                reverse=True,
            )
            for m in sorted_mechs[:4]:
                proven_tasks = ", ".join(mechanism_tasks[m])
                lines.append(f"  * Mechanism: \"{m}\" (active in: {proven_tasks})")
        else:
            lines.append("Validated heuristic mechanisms: None registered yet (early search phase).")

        lines.append("Subtask optimization states:")
        for s in subtask_summaries:
            lines.append(f"  - {s}")
        lines.append(
            "Directive: Evolve the HeuristicOptimizer to promote cross-task reuse of these successful mechanisms "
            "while exploring novel search strategies on stagnant subtasks."
        )
        lines.append("[/Inner Search Evidence & Distilled Mechanisms]")
        return "\n".join(lines)

    def get_seed(self):
        result = super().get_seed()
        if self.outer_hifo is not None:
            snap = self.outer_hifo.observe(self.improver_str, self.meta_utility_val)
            self._hifo_record("outer", "meta-optimizer", {"event": "seed", "state": snap})
        self._hifo_progress("SEED")
        return result

    def _outer_step(self, improve_algorithm_func, previous_algo):
        """Standard MoH outer step, guided by the outer navigator when enabled."""
        controller = self.outer_hifo
        if controller is None:
            return super().try_improvement(improve_algorithm_func, previous_algo)
        original = self.meta_llm
        factory = self._cross_level_context if self.hifo_config.cross_level_enabled else None
        self.meta_llm = GuidedLLM(original, controller, factory)
        try:
            accepted, candidate, driver = super().try_improvement(improve_algorithm_func, previous_algo)
        finally:
            self.meta_llm = original
        snap = controller.observe(candidate, self.meta_utility_val if accepted else None)
        self._hifo_record("outer", "meta-optimizer", {"event": "selection", "accepted": accepted, "candidate_sha256": structure_hash(candidate) if candidate else None, "state": snap})
        return accepted, candidate, driver

    def try_improvement(self, improve_algorithm_func, previous_algo):
        if self.global_controller is None:
            result = self._outer_step(improve_algorithm_func, previous_algo)
            self._hifo_progress("OUTER_STEP")
            return result
        return self._global_step(improve_algorithm_func, previous_algo)

    # ------------------------------------------------------------------
    # Cross-level global controller
    # ------------------------------------------------------------------

    def _run_inner_pass(self, regime: str, reason: str) -> dict[str, Any]:
        """Re-run the incumbent optimizer once per subtask (no meta generation, no validator).

        The incumbent still runs in a GeneratedImprover child via ``get_improver``.
        A failing subtask is logged and skipped; it never aborts the pass.
        """
        subtasks: dict[str, float | None] = {}
        utilities: list[float] = []
        for task in self.subtask_list:
            navigator = self.inner_hifo.get(task)
            if navigator is not None:
                navigator.set_override(regime, reason)
            try:
                self.read_base_algorithm(task)
                result = self.get_improver(self.improver_str, self.subtask_form, task)
                if isinstance(result, Exception):
                    raise result
                idea, code, value = result
                if not code:
                    raise ValueError("incumbent optimizer returned an empty heuristic")
                self.subtask_pop.save_solution(task, idea, code, value)
                self.run_logger.log_utility(getattr(self, "_cur_iter", -1), task, value)
                self.subtask_pop.save_subtask_to_file(task, self._solution_path(task))
                subtasks[task] = _finite_or_none(value)
                if subtasks[task] is not None:
                    utilities.append(subtasks[task])
            except Exception as exc:  # noqa: BLE001 — isolate generated-optimizer failure per subtask
                logger.warning("Inner pass failed on %s (search continues): %s", task, exc)
                subtasks[task] = None
            finally:
                if navigator is not None:
                    navigator.clear_override()
        remeasured = None
        weights = [float(w) for w in getattr(self, "size_weights", [])]
        if len(utilities) == len(self.subtask_list) and len(weights) == len(utilities) and sum(weights) > 0:
            remeasured = sum(w * v for w, v in zip(weights, utilities)) / sum(weights)
        return {"regime": regime, "subtasks": subtasks, "incumbent_remeasured_utility": remeasured}

    def _save_controller_state(self) -> None:
        output = getattr(getattr(self, "run_logger", None), "output_dir", None)
        if not output:
            return
        try:
            self.global_controller.save(Path(output) / "global_controller_state.json")
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Global controller state write failed (search continues): %s", exc)

    def _global_step(self, improve_algorithm_func, previous_algo):
        controller = self.global_controller
        decision = controller.decide(self._navigator_signals())
        logger.info("Global controller (%s) step %d: %s %s", controller.kind, decision.step,
                    decision.action, ",".join(decision.reasons))
        q_before = self._elite_quality()
        meta_before = _finite_or_none(self.meta_utility_val)
        calls_before = self._llm_calls()
        evals_before = int(getattr(self, "_total_eval_calls", 0))
        accepted, candidate, driver = False, None, previous_algo
        inner_report = None
        try:
            if decision.level == "inner":
                inner_report = self._run_inner_pass(decision.regime, decision.action)
            else:
                if self.outer_hifo is not None:
                    self.outer_hifo.set_override(decision.regime, decision.action)
                try:
                    accepted, candidate, driver = self._outer_step(improve_algorithm_func, previous_algo)
                finally:
                    if self.outer_hifo is not None:
                        self.outer_hifo.clear_override()
        except Exception as exc:  # noqa: BLE001 — one failed action must not crash the experiment
            logger.warning("Global action %s failed (search continues): %s", decision.action, exc)
            accepted, candidate, driver = False, None, previous_algo
        cost = self._llm_calls() - calls_before
        q_after = self._elite_quality()
        meta_after = _finite_or_none(self.meta_utility_val)
        reward = controller.feedback(
            decision, cost=cost, q_before=q_before, q_after=q_after, accepted=bool(accepted),
            meta_before=meta_before, meta_after=meta_after,
        )
        self._hifo_record("global", "all", {
            "event": "decision",
            "mode": self.hifo_config.mode,
            "controller": controller.kind,
            **decision.to_dict(),
            "cost_llm_calls": cost,
            "cost_eval_calls": int(getattr(self, "_total_eval_calls", 0)) - evals_before,
            "q_before": q_before,
            "q_after": q_after,
            "meta_before": meta_before,
            "meta_after": meta_after,
            "accepted": bool(accepted),
            "reward": reward,
            "inner_pass": inner_report,
        })
        self._save_controller_state()
        self._hifo_progress(decision.action)
        return accepted, candidate, driver
