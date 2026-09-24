import pytest

from moh.execution.protocol import ExecutionLimits, decode_result


@pytest.mark.parametrize(
    "data",
    [
        b"{}",
        b"NaN",
        b'{"id":"x","id":"x"}',
        b'{"id":"x","status":"success","tour":[true],"error":null}',
        b'{"id":"wrong","status":"success","tour":[0,1,0],"error":null}',
    ],
)
def test_invalid_protocol(data):
    with pytest.raises(ValueError):
        decode_result(data, "x", 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"source_bytes": True},
        {"output_bytes": -1},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        ExecutionLimits(**kwargs)


@pytest.mark.parametrize("error", [{}, [], True, 123])
def test_wrong_type_failure_code_is_protocol_error(error):
    import json

    payload = json.dumps(
        {"id": "x", "status": "failed", "tour": None, "error": error}
    ).encode()
    with pytest.raises(ValueError):
        decode_result(payload, "x", 2)
