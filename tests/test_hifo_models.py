"""CPU-only tests for M8-A Dual-HiFo schema contracts."""

from dataclasses import fields

import pytest

from moh.hifo.models import (
    CapabilityBudgetObservation,
    CreditComponents,
    CrossLevelExperience,
    DiversityObservation,
    EvidenceRef,
    ForesightMode,
    HiFoExperimentMode,
    HiFoLevel,
    InnerEvaluationPoint,
    InnerTrajectorySummary,
    InsightRecord,
    InsightStatus,
    OuterGenerationSummary,
    ProgressObservation,
    to_json_dict,
)


def _evidence(
    *,
    level: HiFoLevel = HiFoLevel.INNER,
    step: int = 0,
) -> EvidenceRef:
    return EvidenceRef(
        run_id="run-001",
        level=level,
        logical_event_seq=step,
        logical_step=step,
    )


def _budget() -> CapabilityBudgetObservation:
    return CapabilityBudgetObservation(
        generate_used=2,
        generate_limit=5,
        evaluate_used=2,
        evaluate_limit=5,
    )


def _point(
    step: int,
    *,
    score: float | None = None,
    error_code: str | None = None,
) -> InnerEvaluationPoint:
    return InnerEvaluationPoint(
        logical_step=step,
        candidate_source_hash=f"hash-{step}",
        evidence_ref=_evidence(step=step),
        generation_kind="mutation",
        score=score,
        error_code=error_code,
    )


def _summary(
    points: tuple[InnerEvaluationPoint, ...] = (),
    *,
    valid: int = 0,
    invalid: int = 0,
    best: float | None = None,
    utility: float | None = None,
) -> InnerTrajectorySummary:
    return InnerTrajectorySummary(
        run_id="run-001",
        optimizer_program_id="o000001",
        optimizer_source_hash="optimizer-hash",
        evaluation_points=points,
        generate_requests=2,
        evaluate_requests=max(2, valid + invalid),
        valid_numeric_evaluations=valid,
        invalid_evaluations=invalid,
        candidate_source_hashes=tuple(
            point.candidate_source_hash for point in points
        ),
        best_valid_score=best,
        final_optimizer_utility=utility,
        capability_budget_snapshot=_budget(),
    )


def test_experiment_modes_exactly_five():
    assert tuple(HiFoExperimentMode) == (
        HiFoExperimentMode.FULL_MOH,
        HiFoExperimentMode.INNER_HIFO,
        HiFoExperimentMode.OUTER_HIFO,
        HiFoExperimentMode.DUAL_HIFO_INDEPENDENT,
        HiFoExperimentMode.DUAL_HIFO_CROSS_LEVEL,
    )


def test_full_moh_predicate_matrix():
    mode = HiFoExperimentMode.FULL_MOH
    assert mode.inner_hindsight_enabled is False
    assert mode.inner_foresight_enabled is False
    assert mode.outer_hindsight_enabled is False
    assert mode.outer_foresight_enabled is False
    assert mode.cross_level_enabled is False


def test_inner_hifo_predicate_matrix():
    mode = HiFoExperimentMode.INNER_HIFO
    assert mode.inner_hindsight_enabled is True
    assert mode.inner_foresight_enabled is True
    assert mode.outer_hindsight_enabled is False
    assert mode.outer_foresight_enabled is False
    assert mode.cross_level_enabled is False


def test_outer_hifo_predicate_matrix():
    mode = HiFoExperimentMode.OUTER_HIFO
    assert mode.inner_hindsight_enabled is False
    assert mode.inner_foresight_enabled is False
    assert mode.outer_hindsight_enabled is True
    assert mode.outer_foresight_enabled is True
    assert mode.cross_level_enabled is False


def test_dual_independent_predicate_matrix():
    mode = HiFoExperimentMode.DUAL_HIFO_INDEPENDENT
    assert mode.inner_hindsight_enabled is True
    assert mode.inner_foresight_enabled is True
    assert mode.outer_hindsight_enabled is True
    assert mode.outer_foresight_enabled is True
    assert mode.cross_level_enabled is False


def test_dual_cross_level_predicate_matrix():
    mode = HiFoExperimentMode.DUAL_HIFO_CROSS_LEVEL
    assert mode.inner_hindsight_enabled is True
    assert mode.inner_foresight_enabled is True
    assert mode.outer_hindsight_enabled is True
    assert mode.outer_foresight_enabled is True
    assert mode.cross_level_enabled is True


def test_foresight_modes_exactly_three():
    assert tuple(ForesightMode) == (
        ForesightMode.EXPLORE,
        ForesightMode.EXPLOIT,
        ForesightMode.BALANCE,
    )


def test_evidence_ref_preserves_caller_identity():
    ref = EvidenceRef(
        run_id="run-001",
        level=HiFoLevel.INNER,
        logical_event_seq=7,
        logical_step=3,
        optimizer_program_id="o000002",
        candidate_source_hash="abc",
    )
    assert ref.logical_event_seq == 7
    assert ref.optimizer_program_id == "o000002"


def test_evidence_ref_rejects_negative_sequence():
    with pytest.raises(ValueError):
        EvidenceRef(
            run_id="run-001",
            level=HiFoLevel.INNER,
            logical_event_seq=-1,
        )


def test_progress_preserves_supplied_counters():
    obs = ProgressObservation(
        logical_step=8,
        progress_counter=5,
        stagnation_counter=3,
    )
    assert obs.progress_counter == 5
    assert obs.stagnation_counter == 3


def test_progress_does_not_auto_update_counters():
    obs = ProgressObservation(
        logical_step=9,
        progress_counter=0,
        stagnation_counter=0,
        current_best_value=10.0,
        previous_best_value=1.0,
    )
    assert obs.progress_counter == 0
    assert obs.stagnation_counter == 0


def test_progress_rejects_negative_counter():
    with pytest.raises(ValueError):
        ProgressObservation(
            logical_step=0,
            progress_counter=-1,
            stagnation_counter=0,
        )


def test_progress_rejects_nan():
    with pytest.raises(ValueError):
        ProgressObservation(
            logical_step=0,
            progress_counter=0,
            stagnation_counter=0,
            current_best_value=float("nan"),
        )


def test_progress_rejects_positive_infinity():
    with pytest.raises(ValueError):
        ProgressObservation(
            logical_step=0,
            progress_counter=0,
            stagnation_counter=0,
            current_best_value=float("inf"),
        )


def test_progress_rejects_negative_infinity():
    with pytest.raises(ValueError):
        ProgressObservation(
            logical_step=0,
            progress_counter=0,
            stagnation_counter=0,
            current_best_value=float("-inf"),
        )


def test_diversity_keeps_exact_and_semantic_values_independent():
    obs = DiversityObservation(
        population_size=10,
        exact_duplicate_count=2,
        exact_duplicate_rate=0.2,
        semantic_diversity=7.5,
        semantic_diversity_method="future-method",
    )
    assert obs.exact_duplicate_rate == 0.2
    assert obs.semantic_diversity == 7.5


def test_diversity_rejects_rate_below_zero():
    with pytest.raises(ValueError):
        DiversityObservation(
            population_size=1,
            exact_duplicate_count=0,
            exact_duplicate_rate=-0.1,
        )


def test_diversity_rejects_rate_above_one():
    with pytest.raises(ValueError):
        DiversityObservation(
            population_size=1,
            exact_duplicate_count=0,
            exact_duplicate_rate=1.1,
        )


def test_diversity_rejects_duplicate_count_above_population():
    with pytest.raises(ValueError):
        DiversityObservation(
            population_size=2,
            exact_duplicate_count=3,
            exact_duplicate_rate=1.0,
        )


def test_budget_snapshot_accepts_existing_accounting():
    budget = CapabilityBudgetObservation(
        generate_used=3,
        generate_limit=5,
        evaluate_used=4,
        evaluate_limit=5,
        provider_attempts_used=7,
        provider_attempts_limit=10,
    )
    assert budget.generate_used == 3
    assert budget.provider_attempts_used == 7


def test_budget_snapshot_rejects_overrun():
    with pytest.raises(ValueError):
        CapabilityBudgetObservation(
            generate_used=6,
            generate_limit=5,
            evaluate_used=0,
            evaluate_limit=5,
        )


def test_credit_components_do_not_compute_aggregate():
    credit = CreditComponents(
        effectiveness=2.0,
        usage_penalty=1.0,
        recency_bonus=0.5,
    )
    assert credit.aggregate_credit is None


def test_credit_components_reject_non_finite_value():
    with pytest.raises(ValueError):
        CreditComponents(effectiveness=float("nan"))


def test_insight_record_supports_active_status():
    record = InsightRecord(
        insight_id="i-001",
        level=HiFoLevel.INNER,
        abstract_insight="Prefer structural changes after repeated failure.",
        source_evidence_refs=(_evidence(),),
        credit=CreditComponents(),
        usage_count=0,
        created_step=0,
    )
    assert record.status is InsightStatus.ACTIVE


def test_insight_record_supports_pruned_status_without_pruning_logic():
    record = InsightRecord(
        insight_id="i-002",
        level=HiFoLevel.OUTER,
        abstract_insight="Example historical insight.",
        source_evidence_refs=(),
        credit=CreditComponents(),
        usage_count=3,
        created_step=1,
        status=InsightStatus.PRUNED,
    )
    assert record.status is InsightStatus.PRUNED
    assert record.usage_count == 3


def test_insight_record_rejects_negative_usage():
    with pytest.raises(ValueError):
        InsightRecord(
            insight_id="i-003",
            level=HiFoLevel.INNER,
            abstract_insight="invalid",
            source_evidence_refs=(),
            credit=CreditComponents(),
            usage_count=-1,
            created_step=0,
        )


def test_inner_evaluation_point_accepts_valid_score():
    point = _point(0, score=-3.5)
    assert point.score == -3.5
    assert point.error_code is None


def test_inner_evaluation_point_accepts_failure():
    point = _point(0, error_code="evaluator_exception")
    assert point.score is None
    assert point.error_code == "evaluator_exception"


def test_inner_evaluation_point_requires_result_or_error():
    with pytest.raises(ValueError):
        _point(0)


def test_inner_trajectory_preserves_chronology():
    points = (
        _point(1, score=1.0),
        _point(2, score=2.0),
    )
    summary = _summary(points, valid=2, best=2.0, utility=2.0)
    assert summary.evaluation_points == points


def test_inner_trajectory_rejects_reverse_chronology():
    points = (
        _point(2, score=2.0),
        _point(1, score=1.0),
    )
    with pytest.raises(ValueError):
        _summary(points, valid=2, best=2.0)


def test_zero_valid_inner_trajectory_is_representable():
    summary = _summary(
        (_point(0, error_code="invalid_return"),),
        valid=0,
        invalid=1,
    )
    assert summary.valid_numeric_evaluations == 0
    assert summary.best_valid_score is None
    assert summary.final_optimizer_utility is None


def test_mixed_inner_trajectory_is_representable():
    summary = _summary(
        (
            _point(0, error_code="invalid_return"),
            _point(1, score=-2.0),
        ),
        valid=1,
        invalid=1,
        best=-2.0,
        utility=-2.0,
    )
    assert summary.valid_numeric_evaluations == 1
    assert summary.invalid_evaluations == 1


def test_outer_generation_summary_represents_none_and_numeric_utilities():
    summary = OuterGenerationSummary(
        outer_generation_index=1,
        parent_program_ids=("o000001", "o000002"),
        parent_utilities=(None, -3.0),
        offspring_program_id="o000003",
        offspring_utility=-2.5,
        surviving_program_ids=("o000002", "o000003"),
        best_utility_after_selection=-2.5,
    )
    assert summary.parent_utilities == (None, -3.0)
    assert summary.best_utility_after_selection == -2.5


def test_cross_level_experience_links_inner_and_outer_outcomes():
    inner = _summary(
        (_point(0, score=-2.0),),
        valid=1,
        best=-2.0,
        utility=-2.0,
    )
    experience = CrossLevelExperience(
        run_id="run-001",
        outer_generation=1,
        optimizer_program_id="o000001",
        survived_outer_selection=True,
        inner_trajectory_summary=inner,
        optimizer_source_hash="optimizer-hash",
        final_optimizer_utility=-2.0,
    )
    assert experience.inner_trajectory_summary is inner
    assert experience.survived_outer_selection is True


def test_serialization_is_deterministic_and_emits_enum_values():
    ref = EvidenceRef(
        run_id="run-001",
        level=HiFoLevel.INNER,
        logical_step=1,
    )
    first = to_json_dict(ref)
    second = to_json_dict(ref)

    assert first == second
    assert first["level"] == "inner"
    assert list(first) == sorted(first)


def test_serialization_converts_tuples_to_lists():
    result = to_json_dict(
        {
            "modes": (
                ForesightMode.EXPLORE,
                ForesightMode.BALANCE,
            )
        }
    )
    assert result == {"modes": ["explore", "balance"]}


def test_serialization_rejects_non_finite_float():
    with pytest.raises(ValueError):
        to_json_dict({"bad": float("inf")})


def test_contracts_have_no_implicit_timestamp_or_random_id_fields():
    names = {
        field.name
        for cls in (
            EvidenceRef,
            ProgressObservation,
            DiversityObservation,
            InnerTrajectorySummary,
            CrossLevelExperience,
        )
        for field in fields(cls)
    }
    assert "timestamp" not in names
    assert "created_at" not in names
    assert "uuid" not in names


def test_insight_rejects_last_use_before_creation():
    with pytest.raises(ValueError):
        InsightRecord(
            insight_id="i-bad",
            level=HiFoLevel.INNER,
            abstract_insight="invalid chronology",
            source_evidence_refs=(),
            credit=CreditComponents(),
            usage_count=1,
            created_step=5,
            last_used_step=2,
        )


def test_cross_level_rejects_conflicting_utility():
    inner = _summary(
        (_point(0, score=-2.0),),
        valid=1,
        best=-2.0,
        utility=-2.0,
    )

    with pytest.raises(ValueError):
        CrossLevelExperience(
            run_id="run-001",
            outer_generation=1,
            optimizer_program_id="o000001",
            survived_outer_selection=True,
            inner_trajectory_summary=inner,
            final_optimizer_utility=-9.0,
        )
