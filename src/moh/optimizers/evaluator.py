"""M3 Evaluator Contract and Inner-Search Semantics."""

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from moh.core.models import OptimizerProgram
from moh.optimizers.capabilities import (
    CapabilityLimits,
    CapabilityUsage,
    OptimizerCapabilityController,
)
from moh.optimizers.runner import OptimizerProgramRunner, ProgramLimits


def classify_candidate_failure(error_msg: str) -> tuple[str, str]:
    msg = error_msg.lower()
    if "syntax" in msg:
        return "candidate_validation", "syntax_error"
    if "missing_function" in msg:
        return "candidate_validation", "missing_function"
    if "invalid_signature" in msg:
        return "candidate_validation", "invalid_signature"
    if "source_limit" in msg:
        return "candidate_validation", "source_limit_exceeded"
    if "invalid_return" in msg or "invalid_result" in msg:
        return "candidate_execution", "invalid_result"
    if "timeout" in msg:
        return "candidate_execution", "execution_timeout"
    if "exception" in msg or "execution" in msg:
        return "candidate_execution", "execution_failed"
    return "evaluator", "evaluator_failed"


@dataclass(frozen=True)
class CandidateEvaluationResult:
    candidate_id: str
    source_code: str
    score: float | None
    valid: bool
    error: str | None = None
    failure_stage: str | None = None
    candidate_error_code: str | None = None
    candidate_source_length: int | None = None
    candidate_source_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.source_code, str):
            raise TypeError("source_code must be a string")
        if type(self.valid) is not bool:
            raise TypeError("valid must be a bool")

        if self.candidate_source_length is None:
            object.__setattr__(self, "candidate_source_length", len(self.source_code))
        if self.candidate_source_sha256 is None:
            sha = hashlib.sha256(self.source_code.encode("utf-8")).hexdigest()
            object.__setattr__(self, "candidate_source_sha256", sha)

        if self.valid:
            if (
                type(self.score) is bool
                or not isinstance(self.score, (int, float))
                or not math.isfinite(self.score)
            ):
                raise ValueError("valid result requires a finite numeric score")
            if self.error is not None:
                raise ValueError("valid result must not have an error")
            object.__setattr__(self, "score", float(self.score))
            object.__setattr__(self, "failure_stage", None)
            object.__setattr__(self, "candidate_error_code", None)
        else:
            if self.score is not None:
                raise ValueError("invalid result must have null score")
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("invalid result requires a non-empty error message")
            if self.failure_stage is None or self.candidate_error_code is None:
                derived_stage, derived_code = classify_candidate_failure(self.error)
                if self.failure_stage is None:
                    object.__setattr__(self, "failure_stage", derived_stage)
                if self.candidate_error_code is None:
                    object.__setattr__(self, "candidate_error_code", derived_code)


@dataclass(frozen=True)
class InnerSearchResult:
    best_candidate_id: str | None
    best_score: float | None
    utility: float | None
    evaluations: tuple[CandidateEvaluationResult, ...]
    generated_count: int
    evaluated_count: int
    valid_evaluation_count: int
    invalid_evaluation_count: int

    def __post_init__(self):
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        for count_name, count_val in (
            ("generated_count", self.generated_count),
            ("evaluated_count", self.evaluated_count),
            ("valid_evaluation_count", self.valid_evaluation_count),
            ("invalid_evaluation_count", self.invalid_evaluation_count),
        ):
            if (
                type(count_val) is bool
                or not isinstance(count_val, int)
                or count_val < 0
            ):
                raise ValueError(f"{count_name} must be a non-negative integer")

        if self.best_score is not None:
            if (
                type(self.best_score) is bool
                or not isinstance(self.best_score, (int, float))
                or not math.isfinite(self.best_score)
            ):
                raise ValueError("best_score must be a finite float if present")
            object.__setattr__(self, "best_score", float(self.best_score))
            if (
                not isinstance(self.best_candidate_id, str)
                or not self.best_candidate_id.strip()
            ):
                raise ValueError(
                    "best_candidate_id must be a string when best_score is present"
                )
            if self.utility != self.best_score:
                raise ValueError(
                    "utility must match best_score when valid evaluations exist"
                )
        else:
            if self.best_candidate_id is not None:
                raise ValueError("best_candidate_id must be None when best_score is None")
            if self.utility is not None:
                raise ValueError(
                    "utility must be None when no valid evaluation exists"
                )


class FakeEvaluator:
    """Deterministic fake evaluator for tests."""

    def __init__(
        self,
        scores: dict[str, Any] | Callable[[str], Any] | list[Any] | None = None,
        default_score: float = 1.0,
    ):
        self.default_score = default_score
        if callable(scores):
            self._fn = scores
        elif isinstance(scores, dict):
            self._fn = lambda src: scores.get(src, self.default_score)
        elif isinstance(scores, list):
            seq = list(scores)
            self._fn = lambda src: seq.pop(0) if seq else self.default_score
        elif scores is None:
            self._fn = lambda src: self.default_score
        else:
            raise TypeError("scores must be a dict, callable, list, or None")

    def __call__(self, source_code: str) -> float:
        if not isinstance(source_code, str):
            raise TypeError("source_code must be a string")
        val = self._fn(source_code)
        if isinstance(val, Exception):
            raise val
        return val


class InnerSearchTracker:
    """Tracks candidate evaluations during an OptimizerProgram inner search."""

    def __init__(self, evaluator: Callable[[str], Any]):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self._evaluator = evaluator
        self._evaluations: list[CandidateEvaluationResult] = []
        self._best_candidate_id: str | None = None
        self._best_score: float | None = None
        self._valid_count = 0
        self._invalid_count = 0

    @property
    def evaluations(self) -> tuple[CandidateEvaluationResult, ...]:
        return tuple(self._evaluations)

    def evaluate_candidate(self, candidate_id: str, source_code: str) -> float:
        try:
            raw_score = self._evaluator(source_code)
        except Exception as exc:
            err_msg = f"evaluator_failed: {type(exc).__name__}"
            res = CandidateEvaluationResult(
                candidate_id=candidate_id,
                source_code=source_code,
                score=None,
                valid=False,
                error=err_msg,
            )
            self._evaluations.append(res)
            self._invalid_count += 1
            raise ValueError(err_msg) from exc

        if (
            type(raw_score) is bool
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(raw_score)
        ):
            err_msg = f"invalid_evaluation_score: {raw_score}"
            res = CandidateEvaluationResult(
                candidate_id=candidate_id,
                source_code=source_code,
                score=None,
                valid=False,
                error=err_msg,
            )
            self._evaluations.append(res)
            self._invalid_count += 1
            raise ValueError(err_msg)

        score_float = float(raw_score)
        res = CandidateEvaluationResult(
            candidate_id=candidate_id,
            source_code=source_code,
            score=score_float,
            valid=True,
            error=None,
        )
        self._evaluations.append(res)
        self._valid_count += 1

        # First-wins tie breaking: update ONLY if new_score > best_score
        if self._best_score is None or score_float > self._best_score:
            self._best_score = score_float
            self._best_candidate_id = candidate_id

        return score_float

    def get_result(
        self, generated_count: int, evaluated_count: int
    ) -> InnerSearchResult:
        return InnerSearchResult(
            best_candidate_id=self._best_candidate_id,
            best_score=self._best_score,
            utility=self._best_score,
            evaluations=tuple(self._evaluations),
            generated_count=generated_count,
            evaluated_count=evaluated_count,
            valid_evaluation_count=self._valid_count,
            invalid_evaluation_count=self._invalid_count,
        )


def run_optimizer_inner_search(
    program: OptimizerProgram,
    llm: Any,
    evaluator: Callable[[str], Any],
    capability_limits: CapabilityLimits,
    program_limits: ProgramLimits | None = None,
) -> tuple[dict[str, Any], CapabilityUsage, InnerSearchResult]:
    tracker = InnerSearchTracker(evaluator)

    def tracked_evaluator(req_id: str, source_code: str) -> float:
        return tracker.evaluate_candidate(req_id, source_code)

    controller = OptimizerCapabilityController(
        llm, tracked_evaluator, capability_limits
    )
    runner = OptimizerProgramRunner()
    worker_result = runner.run(program, controller.handle, limits=program_limits)

    inner_result = tracker.get_result(
        generated_count=controller.usage.generate_requests,
        evaluated_count=controller.usage.evaluate_requests,
    )
    return worker_result, controller.usage, inner_result
