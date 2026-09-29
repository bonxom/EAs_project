"""Schema-only contracts for Dual-HiFo research.

M8-A intentionally contains no HiFo control policy, prompt generation,
semantic-diversity algorithm, credit equation, insight extraction, reuse,
or pruning logic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any


class HiFoExperimentMode(str, Enum):
    """Ablation modes for later controlled HiFo experiments."""

    FULL_MOH = "full_moh"
    INNER_HIFO = "inner_hifo"
    OUTER_HIFO = "outer_hifo"
    DUAL_HIFO_INDEPENDENT = "dual_hifo_independent"
    DUAL_HIFO_CROSS_LEVEL = "dual_hifo_cross_level"

    @property
    def inner_hindsight_enabled(self) -> bool:
        return self in {
            self.INNER_HIFO,
            self.DUAL_HIFO_INDEPENDENT,
            self.DUAL_HIFO_CROSS_LEVEL,
        }

    @property
    def inner_foresight_enabled(self) -> bool:
        return self.inner_hindsight_enabled

    @property
    def outer_hindsight_enabled(self) -> bool:
        return self in {
            self.OUTER_HIFO,
            self.DUAL_HIFO_INDEPENDENT,
            self.DUAL_HIFO_CROSS_LEVEL,
        }

    @property
    def outer_foresight_enabled(self) -> bool:
        return self.outer_hindsight_enabled

    @property
    def cross_level_enabled(self) -> bool:
        return self is self.DUAL_HIFO_CROSS_LEVEL


class HiFoLevel(str, Enum):
    INNER = "inner"
    OUTER = "outer"


class ForesightMode(str, Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    BALANCE = "balance"


class InsightStatus(str, Enum):
    ACTIVE = "active"
    PRUNED = "pruned"


def _require_non_empty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_optional_non_empty_str(name: str, value: str | None) -> None:
    if value is not None:
        _require_non_empty_str(name, value)


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_optional_non_negative_int(name: str, value: int | None) -> None:
    if value is not None:
        _require_non_negative_int(name, value)


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_optional_finite(name: str, value: float | None) -> None:
    if value is not None:
        _require_finite(name, value)


def _require_string_tuple(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    for value in values:
        _require_non_empty_str(name, value)


def _require_count_pairs(
    name: str,
    values: tuple[tuple[str, int], ...],
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")

    seen: set[str] = set()
    for key, count in values:
        _require_non_empty_str(f"{name} key", key)
        _require_non_negative_int(f"{name}[{key!r}]", count)
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        seen.add(key)


@dataclass(frozen=True)
class EvidenceRef:
    """Stable pointer to raw evidence; never an inferred HiFo interpretation."""

    run_id: str
    level: HiFoLevel
    logical_event_seq: int | None = None
    logical_step: int | None = None
    optimizer_program_id: str | None = None
    candidate_source_hash: str | None = None
    generation_kind: str | None = None
    event_kind: str | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str("run_id", self.run_id)
        _require_optional_non_negative_int(
            "logical_event_seq",
            self.logical_event_seq,
        )
        _require_optional_non_negative_int("logical_step", self.logical_step)
        _require_optional_non_empty_str(
            "optimizer_program_id",
            self.optimizer_program_id,
        )
        _require_optional_non_empty_str(
            "candidate_source_hash",
            self.candidate_source_hash,
        )
        _require_optional_non_empty_str("generation_kind", self.generation_kind)
        _require_optional_non_empty_str("event_kind", self.event_kind)
        _require_optional_non_empty_str("failure_code", self.failure_code)


@dataclass(frozen=True)
class ProgressObservation:
    """Caller-supplied Foresight progress/stagnation observation."""

    logical_step: int
    progress_counter: int
    stagnation_counter: int
    current_best_value: float | None = None
    previous_best_value: float | None = None
    last_improvement_step: int | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("logical_step", self.logical_step)
        _require_non_negative_int("progress_counter", self.progress_counter)
        _require_non_negative_int("stagnation_counter", self.stagnation_counter)
        _require_optional_finite("current_best_value", self.current_best_value)
        _require_optional_finite("previous_best_value", self.previous_best_value)
        _require_optional_non_negative_int(
            "last_improvement_step",
            self.last_improvement_step,
        )
        if (
            self.last_improvement_step is not None
            and self.last_improvement_step > self.logical_step
        ):
            raise ValueError(
                "last_improvement_step cannot exceed logical_step"
            )


@dataclass(frozen=True)
class DiversityObservation:
    """Exact duplication and semantic diversity are distinct evidence domains."""

    population_size: int
    exact_duplicate_count: int
    exact_duplicate_rate: float
    semantic_diversity: float | None = None
    semantic_diversity_method: str | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        _require_non_negative_int("population_size", self.population_size)
        _require_non_negative_int(
            "exact_duplicate_count",
            self.exact_duplicate_count,
        )
        if self.exact_duplicate_count > self.population_size:
            raise ValueError(
                "exact_duplicate_count cannot exceed population_size"
            )

        _require_finite("exact_duplicate_rate", self.exact_duplicate_rate)
        if not 0.0 <= float(self.exact_duplicate_rate) <= 1.0:
            raise ValueError("exact_duplicate_rate must be in [0, 1]")

        _require_optional_finite(
            "semantic_diversity",
            self.semantic_diversity,
        )
        _require_optional_non_empty_str(
            "semantic_diversity_method",
            self.semantic_diversity_method,
        )


@dataclass(frozen=True)
class CapabilityBudgetObservation:
    """Snapshot of existing capability accounting; does not redefine it."""

    generate_used: int
    generate_limit: int
    evaluate_used: int
    evaluate_limit: int
    provider_attempts_used: int | None = None
    provider_attempts_limit: int | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("generate_used", self.generate_used)
        _require_non_negative_int("generate_limit", self.generate_limit)
        _require_non_negative_int("evaluate_used", self.evaluate_used)
        _require_non_negative_int("evaluate_limit", self.evaluate_limit)

        if self.generate_used > self.generate_limit:
            raise ValueError("generate_used cannot exceed generate_limit")
        if self.evaluate_used > self.evaluate_limit:
            raise ValueError("evaluate_used cannot exceed evaluate_limit")

        one_provider_value_missing = (
            self.provider_attempts_used is None
        ) != (
            self.provider_attempts_limit is None
        )
        if one_provider_value_missing:
            raise ValueError(
                "provider attempt usage and limit must both be set or both be None"
            )

        if self.provider_attempts_used is not None:
            _require_non_negative_int(
                "provider_attempts_used",
                self.provider_attempts_used,
            )
            _require_non_negative_int(
                "provider_attempts_limit",
                self.provider_attempts_limit,
            )
            if self.provider_attempts_used > self.provider_attempts_limit:
                raise ValueError(
                    "provider_attempts_used cannot exceed provider_attempts_limit"
                )


@dataclass(frozen=True)
class ForesightState:
    """Policy-neutral state snapshot consumed by future Foresight logic."""

    level: HiFoLevel
    logical_step: int
    progress: ProgressObservation
    diversity: DiversityObservation
    budget: CapabilityBudgetObservation
    recent_failure_codes: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    active_insight_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_negative_int("logical_step", self.logical_step)
        _require_string_tuple(
            "recent_failure_codes",
            self.recent_failure_codes,
        )
        _require_string_tuple("active_insight_ids", self.active_insight_ids)


@dataclass(frozen=True)
class ForesightDecision:
    """Representable Foresight decision; M8-A contains no decision engine."""

    level: HiFoLevel
    logical_step: int
    mode: ForesightMode
    reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    verbal_gradient: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("logical_step", self.logical_step)
        _require_string_tuple("reason_codes", self.reason_codes)
        _require_optional_non_empty_str(
            "verbal_gradient",
            self.verbal_gradient,
        )


@dataclass(frozen=True)
class CreditComponents:
    """Unaggregated Hindsight credit inputs plus optional caller result."""

    effectiveness: float | None = None
    usage_penalty: float | None = None
    recency_bonus: float | None = None
    aggregate_credit: float | None = None
    credit_rule_version: str | None = None

    def __post_init__(self) -> None:
        _require_optional_finite("effectiveness", self.effectiveness)
        _require_optional_finite("usage_penalty", self.usage_penalty)
        _require_optional_finite("recency_bonus", self.recency_bonus)
        _require_optional_finite("aggregate_credit", self.aggregate_credit)
        _require_optional_non_empty_str(
            "credit_rule_version",
            self.credit_rule_version,
        )


@dataclass(frozen=True)
class InsightRecord:
    """Stored Hindsight insight without extraction/reuse/pruning behavior."""

    insight_id: str
    level: HiFoLevel
    abstract_insight: str
    source_evidence_refs: tuple[EvidenceRef, ...]
    credit: CreditComponents
    usage_count: int
    created_step: int
    last_used_step: int | None = None
    status: InsightStatus = InsightStatus.ACTIVE
    tags: tuple[str, ...] = ()
    mechanism_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_str("insight_id", self.insight_id)
        _require_non_empty_str("abstract_insight", self.abstract_insight)
        _require_non_negative_int("usage_count", self.usage_count)
        _require_non_negative_int("created_step", self.created_step)
        _require_optional_non_negative_int(
            "last_used_step",
            self.last_used_step,
        )
        if (
            self.last_used_step is not None
            and self.last_used_step < self.created_step
        ):
            raise ValueError(
                "last_used_step cannot precede created_step"
            )
        _require_string_tuple("tags", self.tags)
        _require_string_tuple("mechanism_labels", self.mechanism_labels)


@dataclass(frozen=True)
class InnerEvaluationPoint:
    """Chronological inner evaluation fact."""

    logical_step: int
    candidate_source_hash: str
    evidence_ref: EvidenceRef
    generation_kind: str | None = None
    score: float | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("logical_step", self.logical_step)
        _require_non_empty_str(
            "candidate_source_hash",
            self.candidate_source_hash,
        )
        _require_optional_non_empty_str("generation_kind", self.generation_kind)
        _require_optional_finite("score", self.score)
        _require_optional_non_empty_str("error_code", self.error_code)

        if self.score is None and self.error_code is None:
            raise ValueError(
                "inner evaluation point requires a score or an error_code"
            )


@dataclass(frozen=True)
class InnerTrajectorySummary:
    """Policy-neutral summary of an optimizer's chronological inner search."""

    run_id: str
    optimizer_program_id: str
    evaluation_points: tuple[InnerEvaluationPoint, ...]
    generate_requests: int
    evaluate_requests: int
    valid_numeric_evaluations: int
    invalid_evaluations: int
    candidate_source_hashes: tuple[str, ...] = ()
    failure_code_counts: tuple[tuple[str, int], ...] = ()
    generation_intent_counts: tuple[tuple[str, int], ...] = ()
    optimizer_source_hash: str | None = None
    best_valid_score: float | None = None
    final_optimizer_utility: float | None = None
    capability_budget_snapshot: CapabilityBudgetObservation | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str("run_id", self.run_id)
        _require_non_empty_str(
            "optimizer_program_id",
            self.optimizer_program_id,
        )
        _require_optional_non_empty_str(
            "optimizer_source_hash",
            self.optimizer_source_hash,
        )

        _require_non_negative_int("generate_requests", self.generate_requests)
        _require_non_negative_int("evaluate_requests", self.evaluate_requests)
        _require_non_negative_int(
            "valid_numeric_evaluations",
            self.valid_numeric_evaluations,
        )
        _require_non_negative_int(
            "invalid_evaluations",
            self.invalid_evaluations,
        )

        if (
            self.valid_numeric_evaluations + self.invalid_evaluations
            > self.evaluate_requests
        ):
            raise ValueError(
                "valid + invalid evaluations cannot exceed evaluate_requests"
            )

        _require_string_tuple(
            "candidate_source_hashes",
            self.candidate_source_hashes,
        )
        _require_count_pairs(
            "failure_code_counts",
            self.failure_code_counts,
        )
        _require_count_pairs(
            "generation_intent_counts",
            self.generation_intent_counts,
        )
        _require_optional_finite("best_valid_score", self.best_valid_score)
        _require_optional_finite(
            "final_optimizer_utility",
            self.final_optimizer_utility,
        )

        previous_step: int | None = None
        for point in self.evaluation_points:
            if previous_step is not None and point.logical_step < previous_step:
                raise ValueError(
                    "evaluation_points must preserve chronological order"
                )
            previous_step = point.logical_step


@dataclass(frozen=True)
class OuterGenerationSummary:
    """Policy-neutral snapshot of one outer steady-state generation."""

    outer_generation_index: int
    parent_program_ids: tuple[str, ...]
    parent_utilities: tuple[float | None, ...]
    offspring_program_id: str
    offspring_utility: float | None
    surviving_program_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...] = ()
    best_utility_after_selection: float | None = None
    diversity_observation: DiversityObservation | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(
            "outer_generation_index",
            self.outer_generation_index,
        )
        _require_string_tuple(
            "parent_program_ids",
            self.parent_program_ids,
        )
        _require_non_empty_str(
            "offspring_program_id",
            self.offspring_program_id,
        )
        _require_string_tuple(
            "surviving_program_ids",
            self.surviving_program_ids,
        )

        if len(self.parent_program_ids) != len(self.parent_utilities):
            raise ValueError(
                "parent_program_ids and parent_utilities must have equal length"
            )

        for utility in self.parent_utilities:
            _require_optional_finite("parent utility", utility)
        _require_optional_finite(
            "offspring_utility",
            self.offspring_utility,
        )
        _require_optional_finite(
            "best_utility_after_selection",
            self.best_utility_after_selection,
        )


@dataclass(frozen=True)
class CrossLevelExperience:
    """Research-extension link from inner trajectory to outer outcome."""

    run_id: str
    outer_generation: int
    optimizer_program_id: str
    survived_outer_selection: bool
    inner_trajectory_summary: InnerTrajectorySummary
    optimizer_source_hash: str | None = None
    final_optimizer_utility: float | None = None
    outer_evidence_refs: tuple[EvidenceRef, ...] = ()
    outer_insight_ids: tuple[str, ...] = ()
    inner_insight_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_str("run_id", self.run_id)
        _require_non_negative_int("outer_generation", self.outer_generation)
        _require_non_empty_str(
            "optimizer_program_id",
            self.optimizer_program_id,
        )
        _require_optional_non_empty_str(
            "optimizer_source_hash",
            self.optimizer_source_hash,
        )
        _require_optional_finite(
            "final_optimizer_utility",
            self.final_optimizer_utility,
        )
        _require_string_tuple("outer_insight_ids", self.outer_insight_ids)
        _require_string_tuple("inner_insight_ids", self.inner_insight_ids)

        if self.inner_trajectory_summary.run_id != self.run_id:
            raise ValueError(
                "inner trajectory run_id must match cross-level run_id"
            )
        if (
            self.inner_trajectory_summary.optimizer_program_id
            != self.optimizer_program_id
        ):
            raise ValueError(
                "inner trajectory optimizer_program_id must match "
                "cross-level optimizer_program_id"
            )

        inner_utility = self.inner_trajectory_summary.final_optimizer_utility
        if (
            self.final_optimizer_utility is not None
            and inner_utility is not None
            and self.final_optimizer_utility != inner_utility
        ):
            raise ValueError(
                "cross-level final_optimizer_utility must match "
                "inner trajectory final_optimizer_utility"
            )


def to_json_dict(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation.

    The function performs representation only. It does not derive HiFo state,
    calculate policy, assign credit, or mutate the supplied object.
    """

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_dict(getattr(value, field.name))
            for field in sorted(fields(value), key=lambda item: item.name)
        }

    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("JSON-compatible mapping keys must be strings")
            converted[key] = to_json_dict(value[key])
        return converted

    if isinstance(value, (tuple, list)):
        return [to_json_dict(item) for item in value]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not JSON-compatible")
        return value

    if value is None or isinstance(value, (str, int, bool)):
        return value

    raise TypeError(
        f"unsupported JSON-compatible value type: {type(value).__name__}"
    )
