"""Bounded JSON protocol: no candidate-controlled object deserialization."""

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_seconds: float = 5.0
    source_bytes: int = 65536
    result_bytes: int = 1048576
    output_bytes: int = 65536

    def __post_init__(self):
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout must be positive and finite")
        for limit in (self.source_bytes, self.result_bytes, self.output_bytes):
            if type(limit) is not int or limit <= 0:
                raise ValueError("byte limits must be positive integers")


@dataclass(frozen=True)
class WorkerResult:
    status: str
    tour: tuple[int, ...] | None
    error: str | None
    output: str = ""


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value")

    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON value")
        return number

    try:
        return json.loads(
            text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating
        )
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("invalid JSON") from exc


def decode_result(data, identity, size):
    result = strict_json(data)
    if (
        not isinstance(result, dict)
        or set(result) != {"id", "status", "tour", "error"}
        or result["id"] != identity
    ):
        raise ValueError("invalid protocol envelope")
    if result["status"] == "success":
        tour = result["tour"]
        if (
            result["error"] is not None
            or not isinstance(tour, list)
            or len(tour) != size + 1
            or any(type(x) is not int or not 0 <= x < size for x in tour)
        ):
            raise ValueError("invalid tour result")
        return WorkerResult("success", tuple(tour), None)
    if (
        result["status"] != "failed"
        or result["tour"] is not None
        or not isinstance(result["error"], str)
        or result["error"]
        not in {"syntax", "missing_function", "invalid_return", "exception"}
    ):
        raise ValueError("invalid failure result")
    return WorkerResult("failed", None, result["error"])
