"""Trusted parent capability controller and logical budget accounting."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from moh.core.models import OptimizerProgram
from moh.optimizers.protocol import decode_message
from moh.optimizers.runner import OptimizerProgramRunner, ProgramLimits


@dataclass(frozen=True)
class CapabilityLimits:
    max_generate_requests: int
    max_evaluate_requests: int

    def __post_init__(self):
        for name, val in (
            ("max_generate_requests", self.max_generate_requests),
            ("max_evaluate_requests", self.max_evaluate_requests),
        ):
            if type(val) is bool or not isinstance(val, int):
                raise TypeError(f"{name} must be an integer (not bool)")
            if val < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class CapabilityUsage:
    generate_requests: int
    evaluate_requests: int


class OptimizerCapabilityController:
    def __init__(
        self,
        llm: Any,
        evaluator: Callable[[str], float],
        limits: CapabilityLimits,
    ):
        if not isinstance(limits, CapabilityLimits):
            raise TypeError("limits must be a CapabilityLimits instance")

        self._llm = llm
        self._evaluator = evaluator
        self._limits = limits

        self._generate_requests = 0
        self._evaluate_requests = 0
        self._processed_ids: set[str] = set()

    @property
    def usage(self) -> CapabilityUsage:
        return CapabilityUsage(
            generate_requests=self._generate_requests,
            evaluate_requests=self._evaluate_requests,
        )

    def handle(self, raw_request: dict[str, Any] | str | bytes) -> dict[str, Any]:
        try:
            req = decode_message(raw_request)
        except (ValueError, TypeError) as exc:
            return {
                "type": "error",
                "request_id": "req-unknown",
                "code": "protocol_error",
                "message": f"malformed IPC request: {exc}",
            }

        req_id = req["request_id"]
        req_type = req["type"]

        if req_id in self._processed_ids:
            return {
                "type": "error",
                "request_id": req_id,
                "code": "duplicate_request",
                "message": f"request_id '{req_id}' has already been processed",
            }

        if req_type == "generate":
            return self._handle_generate(req_id, req["prompt"])
        if req_type == "evaluate":
            return self._handle_evaluate(req_id, req["source_code"])

        self._processed_ids.add(req_id)
        return {
            "type": "error",
            "request_id": req_id,
            "code": "unknown_request",
            "message": f"unknown request type '{req_type}'",
        }

    def _handle_generate(self, req_id: str, prompt: str) -> dict[str, Any]:
        self._processed_ids.add(req_id)

        if self._generate_requests >= self._limits.max_generate_requests:
            return {
                "type": "error",
                "request_id": req_id,
                "code": "generation_budget_exhausted",
                "message": "max generation requests limit reached",
            }

        # Consume budget BEFORE calling LLM dependency
        self._generate_requests += 1

        try:
            text = self._llm.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            return {
                "type": "error",
                "request_id": req_id,
                "code": "generation_failed",
                "message": f"LLM generation failed: {type(exc).__name__}",
            }

        if not isinstance(text, str):
            return {
                "type": "error",
                "request_id": req_id,
                "code": "generation_failed",
                "message": f"LLM returned invalid non-string response: {type(text).__name__}",
            }

        return {
            "type": "generate_result",
            "request_id": req_id,
            "text": text,
        }

    def _handle_evaluate(self, req_id: str, source_code: str) -> dict[str, Any]:
        self._processed_ids.add(req_id)

        if self._evaluate_requests >= self._limits.max_evaluate_requests:
            return {
                "type": "error",
                "request_id": req_id,
                "code": "evaluation_budget_exhausted",
                "message": "max evaluation requests limit reached",
            }

        # Consume budget BEFORE calling evaluator dependency
        self._evaluate_requests += 1

        try:
            score = self._evaluator(source_code)
        except Exception as exc:  # noqa: BLE001
            return {
                "type": "error",
                "request_id": req_id,
                "code": "evaluation_failed",
                "message": f"evaluation failed: {type(exc).__name__}",
            }

        if (
            type(score) is bool
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            return {
                "type": "error",
                "request_id": req_id,
                "code": "evaluation_failed",
                "message": f"evaluator returned invalid non-finite score: {score}",
            }

        return {
            "type": "evaluate_result",
            "request_id": req_id,
            "score": float(score),
        }


def run_optimizer_with_capabilities(
    program: OptimizerProgram,
    llm: Any,
    evaluator: Callable[[str], float],
    capability_limits: CapabilityLimits,
    program_limits: ProgramLimits | None = None,
) -> tuple[dict[str, Any], CapabilityUsage]:
    controller = OptimizerCapabilityController(llm, evaluator, capability_limits)
    runner = OptimizerProgramRunner()
    result = runner.run(program, controller.handle, limits=program_limits)
    return result, controller.usage
