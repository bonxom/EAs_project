"""Bounded JSON protocol for optimizer-program worker capabilities."""

import math
from dataclasses import dataclass
from typing import Any

from moh.execution.protocol import strict_json


class CapabilityError(Exception):
    """Domain exception for optimizer capability/protocol failures."""


def validate_request_id(request_id: Any) -> None:
    if not isinstance(request_id, str):
        raise TypeError("request_id must be a string")
    if not request_id:
        raise ValueError("request_id must be a non-empty string")


@dataclass(frozen=True)
class GenerateRequest:
    request_id: str
    prompt: str
    type: str = "generate"

    def __post_init__(self):
        validate_request_id(self.request_id)
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")


@dataclass(frozen=True)
class EvaluateRequest:
    request_id: str
    source_code: str
    type: str = "evaluate"

    def __post_init__(self):
        validate_request_id(self.request_id)
        if not isinstance(self.source_code, str):
            raise TypeError("source_code must be a string")


@dataclass(frozen=True)
class GenerateResponse:
    request_id: str
    text: str
    type: str = "generate_result"

    def __post_init__(self):
        validate_request_id(self.request_id)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True)
class EvaluateResponse:
    request_id: str
    score: float
    type: str = "evaluate_result"

    def __post_init__(self):
        validate_request_id(self.request_id)
        if (
            type(self.score) is bool
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise TypeError("score must be a finite real number (not bool)")


@dataclass(frozen=True)
class ErrorResponse:
    request_id: str
    code: str
    message: str
    type: str = "error"

    def __post_init__(self):
        validate_request_id(self.request_id)
        if not isinstance(self.code, str):
            raise TypeError("code must be a string")
        if not self.code:
            raise ValueError("code must be a non-empty string")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")


def validate_message_dict(d: dict[str, Any]) -> None:
    if not isinstance(d, dict):
        raise TypeError("message must be a dictionary")

    if "type" not in d:
        raise ValueError("missing required field: type")

    msg_type = d["type"]
    if not isinstance(msg_type, str):
        raise TypeError("type must be a string")

    if "request_id" not in d:
        raise ValueError("missing required field: request_id")
    validate_request_id(d["request_id"])

    if msg_type == "generate":
        expected_fields = {"type", "request_id", "prompt"}
        if set(d.keys()) != expected_fields:
            raise ValueError(
                f"invalid fields for generate: expected {expected_fields}, got {set(d.keys())}"
            )
        if not isinstance(d["prompt"], str):
            raise TypeError("prompt must be a string")

    elif msg_type == "evaluate":
        expected_fields = {"type", "request_id", "source_code"}
        if set(d.keys()) != expected_fields:
            raise ValueError(
                f"invalid fields for evaluate: expected {expected_fields}, got {set(d.keys())}"
            )
        if not isinstance(d["source_code"], str):
            raise TypeError("source_code must be a string")

    elif msg_type == "generate_result":
        expected_fields = {"type", "request_id", "text"}
        if set(d.keys()) != expected_fields:
            raise ValueError(
                f"invalid fields for generate_result: expected {expected_fields}, got {set(d.keys())}"
            )
        if not isinstance(d["text"], str):
            raise TypeError("text must be a string")

    elif msg_type == "evaluate_result":
        expected_fields = {"type", "request_id", "score"}
        if set(d.keys()) != expected_fields:
            raise ValueError(
                f"invalid fields for evaluate_result: expected {expected_fields}, got {set(d.keys())}"
            )
        score = d["score"]
        if (
            type(score) is bool
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise TypeError("score must be a finite real number (not bool)")

    elif msg_type == "error":
        expected_fields = {"type", "request_id", "code", "message"}
        if set(d.keys()) != expected_fields:
            raise ValueError(
                f"invalid fields for error: expected {expected_fields}, got {set(d.keys())}"
            )
        code = d["code"]
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if not code:
            raise ValueError("code must be a non-empty string")
        if not isinstance(d["message"], str):
            raise TypeError("message must be a string")

    else:
        raise ValueError(f"unknown message type: {msg_type}")


def encode_message(msg: Any) -> str:
    import json

    if isinstance(msg, dict):
        d = msg
    elif hasattr(msg, "__dataclass_fields__"):
        d = {k: getattr(msg, k) for k in msg.__dataclass_fields__}
    else:
        raise TypeError("unsupported message object")

    validate_message_dict(d)
    return json.dumps(d, sort_keys=True, allow_nan=False)


def decode_message(data: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, (str, bytes)):
        try:
            d = strict_json(data)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"malformed JSON: {exc}") from exc
    elif isinstance(data, dict):
        d = data
    else:
        raise TypeError("message data must be str, bytes, or dict")

    validate_message_dict(d)
    return d
