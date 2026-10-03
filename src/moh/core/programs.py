"""Immutable records for program search with nonnegative, minimizing fitness."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from moh.core.models import Heuristic, Status, WorkCounts

if TYPE_CHECKING:
    from moh.core.program_population import ProgramPopulation


def _safe_id(value: str) -> None:
    """IDs can be used as artifact basenames without path traversal."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value
    ):
        raise ValueError(
            "expected a safe nonempty identifier of at most 128 characters"
        )


def _number(value: float, *, positive: bool = False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            "expected a finite nonnegative number"
            if not positive
            else "expected a finite positive number"
        )


def _outcome(status: Status, utility: float | None, error: str | None) -> None:
    if status == "success":
        _number(utility)
        if error is not None:
            raise ValueError("success requires no error")
    elif status == "failed":
        if (
            utility is not None
            or not isinstance(error, str)
            or not 0 < len(error) <= 1024
        ):
            raise ValueError("failure requires null utility and a bounded error")
    else:
        raise ValueError("invalid status")


def _program(candidate: Heuristic | OptimizerProgram) -> None:
    if not isinstance(candidate, (Heuristic, OptimizerProgram)):
        raise TypeError("expected a heuristic or optimizer program")
    _safe_id(candidate.id)
    if not isinstance(candidate.source_code, str) or not candidate.source_code:
        raise ValueError("program source must be a nonempty string")
    try:
        candidate.source_code.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("program source must encode as UTF-8") from error
    if candidate.idea is not None and not isinstance(candidate.idea, str):
        raise ValueError("idea must be a string or None")


@dataclass(frozen=True)
class OptimizerProgram:
    id: str
    source_code: str
    idea: str | None = None

    def __post_init__(self):
        _program(self)


@dataclass(frozen=True)
class GapEvaluation:
    candidate_id: str
    task_id: str
    split: Literal["validation", "test"]
    status: Status
    utility: float | None
    costs: tuple[float, ...]
    gaps: tuple[float, ...]
    tours: tuple[tuple[int, ...], ...]
    error: str | None = None
    counts: WorkCounts = field(default_factory=WorkCounts)

    def __post_init__(self):
        _safe_id(self.candidate_id)
        _safe_id(self.task_id)
        object.__setattr__(self, "costs", tuple(self.costs))
        object.__setattr__(self, "gaps", tuple(self.gaps))
        object.__setattr__(self, "tours", tuple(tuple(tour) for tour in self.tours))
        if self.split not in ("validation", "test"):
            raise ValueError("split must be validation or test")
        _outcome(self.status, self.utility, self.error)
        if not isinstance(self.counts, WorkCounts):
            raise TypeError("counts must be WorkCounts")
        for cost in self.costs:
            _number(cost, positive=True)
        for gap in self.gaps:
            _number(gap)
        if not len(self.costs) == len(self.gaps) == len(self.tours):
            raise ValueError("costs, gaps and tours must have matching lengths")
        for tour in self.tours:
            if not tour or any(type(city) is not int or city < 0 for city in tour):
                raise ValueError("tour must contain nonnegative integer cities")
        if self.status == "success" and (
            not self.gaps
            or not math.isclose(
                self.utility,
                math.fsum(gap / len(self.gaps) for gap in self.gaps),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("success requires nonempty results and mean-gap utility")


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    selected: ScoredProgram | None

    def __post_init__(self):
        _safe_id(self.task_id)
        if self.selected is not None and (
            not isinstance(self.selected, ScoredProgram)
            or not isinstance(self.selected.candidate, Heuristic)
            or not isinstance(self.selected.evaluation, GapEvaluation)
            or self.selected.evaluation.task_id != self.task_id
            or self.selected.evaluation.status != "success"
            or self.selected.evaluation.split != "validation"
        ):
            raise ValueError("selected heuristic must succeed on this validation task")


@dataclass(frozen=True)
class ProgramEvaluation:
    candidate_id: str
    status: Status
    utility: float | None
    task_results: tuple[TaskOutcome, ...]
    error: str | None = None
    counts: WorkCounts = field(default_factory=WorkCounts)

    def __post_init__(self):
        _safe_id(self.candidate_id)
        object.__setattr__(self, "task_results", tuple(self.task_results))
        _outcome(self.status, self.utility, self.error)
        if not isinstance(self.counts, WorkCounts):
            raise TypeError("counts must be WorkCounts")
        if any(not isinstance(result, TaskOutcome) for result in self.task_results):
            raise ValueError("task results must be TaskOutcome records")
        if len({result.task_id for result in self.task_results}) != len(
            self.task_results
        ):
            raise ValueError("task results must have distinct task IDs")
        if self.status == "success" and (
            not self.task_results
            or any(result.selected is None for result in self.task_results)
        ):
            raise ValueError("program success requires every task to succeed")


@dataclass(frozen=True)
class ScoredProgram:
    candidate: Heuristic | OptimizerProgram
    evaluation: GapEvaluation | ProgramEvaluation

    def __post_init__(self):
        _program(self.candidate)
        if (
            isinstance(self.candidate, Heuristic)
            and not isinstance(self.evaluation, GapEvaluation)
            or isinstance(self.candidate, OptimizerProgram)
            and not isinstance(self.evaluation, ProgramEvaluation)
        ):
            raise TypeError("evaluation type must match program type")
        if self.candidate.id != self.evaluation.candidate_id:
            raise ValueError("candidate/evaluation identity mismatch")

    @property
    def id(self) -> str:
        return self.candidate.id

    @property
    def source_code(self) -> str:
        return self.candidate.source_code

    @property
    def idea(self) -> str | None:
        return self.candidate.idea

    @property
    def utility(self) -> float | None:
        return self.evaluation.utility


@dataclass(frozen=True)
class ProgramRunResult:
    status: Status
    winner: ScoredProgram | None
    active: ScoredProgram | None
    population: tuple[ScoredProgram, ...]
    task_populations: Mapping[str, ProgramPopulation]
    test_results: tuple[GapEvaluation, ...]
    counts: WorkCounts

    def __post_init__(self):
        object.__setattr__(self, "population", tuple(self.population))
        object.__setattr__(
            self, "task_populations", MappingProxyType(dict(self.task_populations))
        )
        object.__setattr__(self, "test_results", tuple(self.test_results))
        if self.status not in ("success", "failed") or not isinstance(
            self.counts, WorkCounts
        ):
            raise ValueError("invalid run status or counts")
        if any(item.evaluation.status != "success" for item in self.population):
            raise ValueError("run population must contain successful programs")
        for selected in (self.winner, self.active):
            if selected is not None and (selected.evaluation.status != "success"):
                raise ValueError("winner and active must be successfully evaluated")
        if self.winner is not None and self.winner not in self.population:
            raise ValueError("winner must be a population member")
        if self.status == "success" and (self.winner is None or self.active is None):
            raise ValueError("successful run requires winner and active programs")
        if self.status == "failed" and (
            self.winner is not None or self.active is not None or self.population
        ):
            raise ValueError("failed run cannot carry successful optimizer state")
        if any(result.split != "test" for result in self.test_results):
            raise ValueError("run test results must belong to held-out test split")
