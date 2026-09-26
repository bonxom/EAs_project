import pytest

from moh.optimizers.protocol import (
    CapabilityError,
    decode_message,
    encode_message,
)
from moh.optimizers.proxy import ImprovementAPI


def test_valid_generate():
    received_requests = []

    def transport(req_str):
        req = decode_message(req_str)
        received_requests.append(req)
        return encode_message(
            {
                "type": "generate_result",
                "request_id": req["request_id"],
                "text": "def improved(): pass",
            }
        )

    api = ImprovementAPI(transport)
    result = api.generate("prompt text")
    assert result == "def improved(): pass"
    assert len(received_requests) == 1
    assert received_requests[0] == {
        "type": "generate",
        "request_id": "req-000001",
        "prompt": "prompt text",
    }


def test_valid_evaluate():
    received_requests = []

    def transport(req_str):
        req = decode_message(req_str)
        received_requests.append(req)
        return encode_message(
            {
                "type": "evaluate_result",
                "request_id": req["request_id"],
                "score": -42.5,
            }
        )

    api = ImprovementAPI(transport)
    score = api.evaluate("def solve(): pass")
    assert score == -42.5
    assert isinstance(score, float)
    assert len(received_requests) == 1
    assert received_requests[0] == {
        "type": "evaluate",
        "request_id": "req-000001",
        "source_code": "def solve(): pass",
    }


def test_sequential_request_ids():
    ids = []

    def transport(req_str):
        req = decode_message(req_str)
        ids.append(req["request_id"])
        if req["type"] == "generate":
            return encode_message(
                {"type": "generate_result", "request_id": req["request_id"], "text": "ok"}
            )
        return encode_message(
            {"type": "evaluate_result", "request_id": req["request_id"], "score": 1.0}
        )

    api = ImprovementAPI(transport)
    api.generate("p1")
    api.evaluate("s1")
    api.generate("p2")

    assert ids == ["req-000001", "req-000002", "req-000003"]


def test_non_string_prompt_rejected():
    api = ImprovementAPI(lambda x: x)
    with pytest.raises(CapabilityError, match="prompt must be a string"):
        api.generate(123)  # type: ignore


def test_non_string_source_code_rejected():
    api = ImprovementAPI(lambda x: x)
    with pytest.raises(CapabilityError, match="source_code must be a string"):
        api.evaluate(None)  # type: ignore


@pytest.mark.parametrize(
    "bad_score", [float("nan"), float("inf"), float("-inf"), True, False]
)
def test_invalid_scores_rejected(bad_score):
    def transport(req_str):
        req = decode_message(req_str)
        return {
            "type": "evaluate_result",
            "request_id": req["request_id"],
            "score": bad_score,
        }

    api = ImprovementAPI(transport)
    with pytest.raises(CapabilityError):
        api.evaluate("def f(): pass")


def test_missing_required_field_rejected():
    def transport(req_str):
        req = decode_message(req_str)
        return {"type": "generate_result", "request_id": req["request_id"]}  # missing text

    api = ImprovementAPI(transport)
    with pytest.raises(CapabilityError):
        api.generate("prompt")


def test_unknown_message_type_rejected():
    def transport(req_str):
        req = decode_message(req_str)
        return {"type": "unknown_type", "request_id": req["request_id"]}

    api = ImprovementAPI(transport)
    with pytest.raises(CapabilityError):
        api.generate("prompt")


def test_mismatched_request_id_rejected():
    def transport(req_str):
        return encode_message(
            {"type": "generate_result", "request_id": "req-999999", "text": "text"}
        )

    api = ImprovementAPI(transport)
    with pytest.raises(CapabilityError, match="mismatched request_id"):
        api.generate("prompt")


def test_wrong_response_type_rejected():
    def transport(req_str):
        req = decode_message(req_str)
        return encode_message(
            {"type": "evaluate_result", "request_id": req["request_id"], "score": 10.0}
        )

    api = ImprovementAPI(transport)
    with pytest.raises(CapabilityError, match="wrong response type"):
        api.generate("prompt")


def test_bounded_error_response_converted_to_domain_exception():
    def transport(req_str):
        req = decode_message(req_str)
        return encode_message(
            {
                "type": "error",
                "request_id": req["request_id"],
                "code": "resource_exhausted",
                "message": "out of memory",
            }
        )

    api = ImprovementAPI(transport)
    with pytest.raises(CapabilityError, match=r"parent error \[resource_exhausted\]: out of memory"):
        api.generate("prompt")


def test_malformed_json_response_rejected():
    def transport(req_str):
        return "{malformed json"

    api = ImprovementAPI(transport)
    with pytest.raises(CapabilityError, match="invalid response"):
        api.generate("prompt")
