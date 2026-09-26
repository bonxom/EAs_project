"""Worker-side proxy for ImprovementAPI capability interface."""

import math
from collections.abc import Callable
from typing import Any

from moh.optimizers.protocol import (
    CapabilityError,
    EvaluateRequest,
    GenerateRequest,
    decode_message,
    encode_message,
)


class ImprovementAPI:
    def __init__(
        self,
        transport: Callable[[Any], Any],
        id_factory: Callable[[], str] | None = None,
    ):
        self._transport = transport
        if id_factory is None:
            self._counter = 0
            self._id_factory = self._default_id_factory
        else:
            self._id_factory = id_factory

    def _default_id_factory(self) -> str:
        self._counter += 1
        return f"req-{self._counter:06d}"

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise CapabilityError("prompt must be a string")

        req_id = self._id_factory()
        req = GenerateRequest(request_id=req_id, prompt=prompt)
        resp_dict = self._send(req)

        if resp_dict["request_id"] != req_id:
            raise CapabilityError(
                f"mismatched request_id: expected {req_id}, got {resp_dict['request_id']}"
            )

        if resp_dict["type"] == "error":
            raise CapabilityError(
                f"parent error [{resp_dict['code']}]: {resp_dict['message']}"
            )

        if resp_dict["type"] != "generate_result":
            raise CapabilityError(
                f"wrong response type: expected generate_result, got {resp_dict['type']}"
            )

        return resp_dict["text"]

    def evaluate(self, source_code: str) -> float:
        if not isinstance(source_code, str):
            raise CapabilityError("source_code must be a string")

        req_id = self._id_factory()
        req = EvaluateRequest(request_id=req_id, source_code=source_code)
        resp_dict = self._send(req)

        if resp_dict["request_id"] != req_id:
            raise CapabilityError(
                f"mismatched request_id: expected {req_id}, got {resp_dict['request_id']}"
            )

        if resp_dict["type"] == "error":
            raise CapabilityError(
                f"parent error [{resp_dict['code']}]: {resp_dict['message']}"
            )

        if resp_dict["type"] != "evaluate_result":
            raise CapabilityError(
                f"wrong response type: expected evaluate_result, got {resp_dict['type']}"
            )

        score = resp_dict["score"]
        if (
            type(score) is bool
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise CapabilityError(f"invalid score: {score}")

        return float(score)

    def _send(self, req: Any) -> dict[str, Any]:
        try:
            req_data = encode_message(req)
        except (ValueError, TypeError) as exc:
            raise CapabilityError(f"invalid request: {exc}") from exc

        try:
            raw_resp = self._transport(req_data)
        except CapabilityError:
            raise
        except Exception as exc:
            raise CapabilityError(f"transport error: {exc}") from exc

        try:
            return decode_message(raw_resp)
        except (ValueError, TypeError) as exc:
            raise CapabilityError(f"invalid response: {exc}") from exc
