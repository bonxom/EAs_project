import pytest

from moh.optimizers.protocol import (
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    GenerateRequest,
    GenerateResponse,
    decode_message,
    encode_message,
)


def test_roundtrip_generate_request():
    req = GenerateRequest(request_id="req-000001", prompt="write a heuristic")
    encoded = encode_message(req)
    decoded = decode_message(encoded)
    assert decoded == {
        "type": "generate",
        "request_id": "req-000001",
        "prompt": "write a heuristic",
    }


def test_roundtrip_evaluate_request():
    req = EvaluateRequest(request_id="req-000002", source_code="def improve(): pass")
    encoded = encode_message(req)
    decoded = decode_message(encoded)
    assert decoded == {
        "type": "evaluate",
        "request_id": "req-000002",
        "source_code": "def improve(): pass",
    }


def test_roundtrip_generate_response():
    resp = GenerateResponse(request_id="req-000001", text="def candidate(): pass")
    encoded = encode_message(resp)
    decoded = decode_message(encoded)
    assert decoded == {
        "type": "generate_result",
        "request_id": "req-000001",
        "text": "def candidate(): pass",
    }


def test_roundtrip_evaluate_response():
    resp = EvaluateResponse(request_id="req-000002", score=-3.14159)
    encoded = encode_message(resp)
    decoded = decode_message(encoded)
    assert decoded == {
        "type": "evaluate_result",
        "request_id": "req-000002",
        "score": -3.14159,
    }


def test_roundtrip_error_response():
    resp = ErrorResponse(request_id="req-000003", code="eval_failed", message="timeout")
    encoded = encode_message(resp)
    decoded = decode_message(encoded)
    assert decoded == {
        "type": "error",
        "request_id": "req-000003",
        "code": "eval_failed",
        "message": "timeout",
    }


@pytest.mark.parametrize("req_id", ["", 123, None, []])
def test_invalid_request_id(req_id):
    with pytest.raises((ValueError, TypeError)):
        GenerateRequest(request_id=req_id, prompt="test")


@pytest.mark.parametrize("prompt", [123, None, ["a"]])
def test_invalid_generate_prompt(prompt):
    with pytest.raises((ValueError, TypeError)):
        GenerateRequest(request_id="req-001", prompt=prompt)


@pytest.mark.parametrize("source", [123, None, {"code": "x"}])
def test_invalid_evaluate_source(source):
    with pytest.raises((ValueError, TypeError)):
        EvaluateRequest(request_id="req-001", source_code=source)


@pytest.mark.parametrize(
    "score", [float("nan"), float("inf"), float("-inf"), True, False, "1.0", None]
)
def test_invalid_evaluate_score(score):
    with pytest.raises((ValueError, TypeError)):
        EvaluateResponse(request_id="req-001", score=score)


@pytest.mark.parametrize("code", ["", 123, None])
def test_invalid_error_code(code):
    with pytest.raises((ValueError, TypeError)):
        ErrorResponse(request_id="req-001", code=code, message="msg")


@pytest.mark.parametrize(
    "raw_json",
    [
        b"{}",
        b"NaN",
        b'{"type": "unknown", "request_id": "req-001"}',
        b'{"type": "generate", "request_id": "req-001"}',  # missing prompt
        b'{"type": "generate", "request_id": "req-001", "prompt": "p", "extra": 1}',  # extra field
        b'{"type": "generate", "request_id": "req-001", "prompt": "p", "type": "generate"}',  # duplicate key
    ],
)
def test_decode_message_failures(raw_json):
    with pytest.raises((ValueError, TypeError)):
        decode_message(raw_json)
