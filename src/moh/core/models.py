"""Immutable search records. Scores belong to evaluations, never programs."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from moh.core.specs import OptimizerSpec

type Status = Literal["success", "failed"]
type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)
type EventSink = Callable[[str, dict[str, JSONValue]], None]


def _nonnegative(value: int):
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer")


def _outcome(status, utility, error):
    if status == "success":
        if (
            type(utility) not in (int, float)
            or not math.isfinite(utility)
            or error is not None
        ):
            raise ValueError("success requires finite utility and no error")
    elif status == "failed":
        if (
            utility is not None
            or not isinstance(error, str)
            or not 0 < len(error) <= 1024
        ):
            raise ValueError("failure requires null utility and a bounded error")
    else:
        raise ValueError("invalid status")


@dataclass(frozen=True)
class Heuristic:
    id: str
    source_code: str
    idea: str | None = None


@dataclass(frozen=True)
class EvaluationContext:
    task_id: str
    instance_seeds: tuple[int, ...]
    worker_seeds: tuple[int, ...]
    repetition: int = 0

    def __post_init__(self):
        object.__setattr__(self, "instance_seeds", tuple(self.instance_seeds))
        object.__setattr__(self, "worker_seeds", tuple(self.worker_seeds))
        _nonnegative(self.repetition)
        if not self.instance_seeds or len(self.instance_seeds) != len(
            self.worker_seeds
        ):
            raise ValueError("context needs matching nonempty seeds")
        for seed in (*self.instance_seeds, *self.worker_seeds):
            _nonnegative(seed)
            if seed >= 2**32:
                raise ValueError("worker seeds must fit uint32")


@dataclass(frozen=True)
class EvaluationResult:
    candidate_id: str
    context: EvaluationContext
    status: Status
    utility: float | None
    lengths: tuple[float, ...]
    error: str | None
    instances_attempted: int

    def __post_init__(self):
        object.__setattr__(self, "lengths", tuple(self.lengths))
        _outcome(self.status, self.utility, self.error)
        _nonnegative(self.instances_attempted)
        if (
            not len(self.lengths)
            <= self.instances_attempted
            <= len(self.context.instance_seeds)
        ):
            raise ValueError("inconsistent attempt count")
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0
            for x in self.lengths
        ):
            raise ValueError("invalid lengths")
        if self.status == "success" and (
            len(self.lengths) != len(self.context.instance_seeds)
            or self.utility != -math.fsum(x / len(self.lengths) for x in self.lengths)
        ):
            raise ValueError("success needs all lengths and negative mean utility")


@dataclass(frozen=True)
class ScoredHeuristic:
    heuristic: Heuristic
    evaluation: EvaluationResult

    def __post_init__(self):
        if self.heuristic.id != self.evaluation.candidate_id:
            raise ValueError("candidate/evaluation identity mismatch")


@dataclass(frozen=True)
class OptimizerCandidate:
    id: str
    spec: OptimizerSpec


@dataclass(frozen=True)
class WorkCounts:
    heuristic_evaluations: int = 0
    instance_attempts: int = 0
    llm_calls: int = 0

    def __post_init__(self):
        for count in (
            self.heuristic_evaluations,
            self.instance_attempts,
            self.llm_calls,
        ):
            _nonnegative(count)

    def __add__(self, other):
        return WorkCounts(
            self.heuristic_evaluations + other.heuristic_evaluations,
            self.instance_attempts + other.instance_attempts,
            self.llm_calls + other.llm_calls,
        )


@dataclass(frozen=True)
class InnerResult:
    population: tuple[ScoredHeuristic, ...]
    counts: WorkCounts

    def __post_init__(self):
        object.__setattr__(self, "population", tuple(self.population))


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    best: ScoredHeuristic | None
    counts: WorkCounts

    def __post_init__(self):
        if self.best is not None and (
            self.best.evaluation.context.task_id != self.task_id
            or self.best.evaluation.status != "success"
        ):
            raise ValueError("task best must be successful and belong to task")


@dataclass(frozen=True)
class OptimizerEvaluation:
    optimizer_id: str
    status: Status
    utility: float | None
    task_results: tuple[TaskResult, ...]
    counts: WorkCounts
    error: str | None

    def __post_init__(self):
        object.__setattr__(self, "task_results", tuple(self.task_results))
        _outcome(self.status, self.utility, self.error)
        if self.status == "success" and (
            not self.task_results or any(x.best is None for x in self.task_results)
        ):
            raise ValueError("optimizer success requires every task to succeed")


@dataclass(frozen=True)
class ScoredOptimizer:
    candidate: OptimizerCandidate
    evaluation: OptimizerEvaluation

    def __post_init__(self):
        if self.candidate.id != self.evaluation.optimizer_id:
            raise ValueError("optimizer/evaluation identity mismatch")


@dataclass(frozen=True)
class RunResult:
    status: Status
    winner: ScoredOptimizer | None
    population: tuple[ScoredOptimizer, ...]

    def __post_init__(self):
        object.__setattr__(self, "population", tuple(self.population))
        if self.status == "success":
            if (
                self.winner is None
                or self.winner not in self.population
                or self.winner.evaluation.status != "success"
            ):
                raise ValueError("successful run needs an evaluated winner")
        elif (
            self.status != "failed"
            or self.winner is not None
            or any(x.evaluation.status == "success" for x in self.population)
        ):
            raise ValueError("invalid failed run")
