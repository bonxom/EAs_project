import os
import time

import pytest

from moh.core.models import OptimizerProgram
from moh.optimizers.runner import OptimizerProgramRunner, ProgramLimits


def dummy_handler(req):
    req_type = req.get("type")
    req_id = req.get("request_id", "req-unknown")
    if req_type == "generate":
        return {
            "type": "generate_result",
            "request_id": req_id,
            "text": f"generated-{req.get('prompt')}",
        }
    if req_type == "evaluate":
        return {
            "type": "evaluate_result",
            "request_id": req_id,
            "score": 42.0,
        }
    return {
        "type": "error",
        "request_id": req_id,
        "code": "unknown_request",
        "message": f"unknown request type: {req_type}",
    }


def test_trivial_successful_program():
    code = """
def improve_algorithm(api):
    return {"status": "ok", "value": 10}
"""
    prog = OptimizerProgram(id="opt-001", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "success"
    assert res["result"] == {"status": "ok", "value": 10}


def test_generate_capability_roundtrip():
    code = """
def improve_algorithm(api):
    return api.generate("make heuristic")
"""
    prog = OptimizerProgram(id="opt-002", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "success"
    assert res["result"] == "generated-make heuristic"


def test_evaluate_capability_roundtrip():
    code = """
def improve_algorithm(api):
    return api.evaluate("def solve(): pass")
"""
    prog = OptimizerProgram(id="opt-003", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "success"
    assert res["result"] == 42.0


def test_combined_generate_and_evaluate_roundtrip():
    code = """
def improve_algorithm(api):
    text = api.generate("make heuristic")
    score = api.evaluate(text)
    return {
        "text": text,
        "score": score
    }
"""
    prog = OptimizerProgram(id="opt-004", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "success"
    assert res["result"] == {"text": "generated-make heuristic", "score": 42.0}


def test_multiple_capability_requests_sequential_ids():
    received_ids = []

    def tracking_handler(req):
        received_ids.append(req["request_id"])
        return dummy_handler(req)

    code = """
def improve_algorithm(api):
    a = api.generate("a")
    b = api.generate("b")
    x = api.evaluate(a)
    y = api.evaluate(b)
    return [a, b, x, y]
"""
    prog = OptimizerProgram(id="opt-005", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, tracking_handler)

    assert res["status"] == "success"
    assert res["result"] == ["generated-a", "generated-b", 42.0, 42.0]
    assert received_ids == ["req-000001", "req-000002", "req-000003", "req-000004"]


def test_runtime_exception_containment():
    code = """
def improve_algorithm(api):
    raise ZeroDivisionError("division by zero")
"""
    prog = OptimizerProgram(id="opt-006", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "failed"
    assert res["code"] == "runtime_error"
    assert "ZeroDivisionError" in res["message"]


def test_worker_os_exit_containment():
    code = """
import os

def improve_algorithm(api):
    os._exit(7)
"""
    prog = OptimizerProgram(id="opt-007", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "failed"
    assert res["code"] == "worker_exit"


def test_infinite_loop_timeout():
    code = """
def improve_algorithm(api):
    while True:
        pass
"""
    prog = OptimizerProgram(id="opt-008", source_code=code)
    runner = OptimizerProgramRunner()
    start_time = time.monotonic()
    res = runner.run(prog, dummy_handler, limits=ProgramLimits(timeout_seconds=0.5))
    elapsed = time.monotonic() - start_time

    assert res["status"] == "failed"
    assert res["code"] == "timeout"
    assert elapsed < 2.0


def test_descendant_process_cleanup():
    code = """
import os
import time

def improve_algorithm(api):
    pid = os.fork()
    if pid == 0:
        time.sleep(60)
        os._exit(0)
    return pid
"""
    prog = OptimizerProgram(id="opt-009", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "success"
    child_pid = res["result"]
    assert isinstance(child_pid, int)

    # Verify descendant process is gone (sending signal 0 to reaped child raises OSError/ProcessLookupError)
    time.sleep(0.1)
    with pytest.raises(OSError):
        os.kill(child_pid, 0)


def test_malformed_python_source():
    code = "def improve_algorithm(api: syntax error"
    prog = OptimizerProgram(id="opt-010", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "failed"
    assert res["code"] == "invalid_program"


def test_missing_entrypoint():
    code = """
def other_function(api):
    return None
"""
    prog = OptimizerProgram(id="opt-011", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "failed"
    assert res["code"] == "missing_entrypoint"


def test_non_callable_entrypoint():
    code = "improve_algorithm = 123"
    prog = OptimizerProgram(id="opt-012", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "failed"
    assert res["code"] == "missing_entrypoint"


def test_invalid_return_object():
    code = """
def improve_algorithm(api):
    return object()
"""
    prog = OptimizerProgram(id="opt-013", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "failed"
    assert res["code"] == "invalid_return"


@pytest.mark.parametrize("bad_val", ["float('nan')", "float('inf')", "float('-inf')"])
def test_non_finite_return_rejected(bad_val):
    code = f"""
def improve_algorithm(api):
    return {bad_val}
"""
    prog = OptimizerProgram(id="opt-014", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "failed"
    assert res["code"] == "invalid_return"


def test_large_stdout_bounding():
    code = """
def improve_algorithm(api):
    print("x" * 100000)
    return "ok"
"""
    prog = OptimizerProgram(id="opt-015", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler, limits=ProgramLimits(max_output_bytes=4096))

    assert res["status"] == "success"
    assert res["result"] == "ok"
    assert len(res["output"]) <= 4096


def test_mismatched_request_id_from_parent():
    def bad_id_handler(req):
        return {
            "type": "generate_result",
            "request_id": "req-wrong",
            "text": "text",
        }

    code = """
def improve_algorithm(api):
    return api.generate("hello")
"""
    prog = OptimizerProgram(id="opt-016", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, bad_id_handler)

    assert res["status"] == "failed"
    assert res["code"] == "capability_error"
    assert "mismatched request_id" in res["message"]


def test_security_assertion_no_parent_objects():
    code = """
def improve_algorithm(api):
    return {
        "has_llm": "llm" in globals(),
        "has_evaluator": "evaluator" in globals(),
        "has_handler": "request_handler" in globals(),
        "has_runner": "runner" in globals(),
    }
"""
    prog = OptimizerProgram(id="opt-017", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "success"
    assert res["result"] == {
        "has_llm": False,
        "has_evaluator": False,
        "has_handler": False,
        "has_runner": False,
    }


def test_environment_isolation_no_credentials():
    os.environ["SECRET_API_KEY"] = "sk-super-secret-12345"
    code = """
import os

def improve_algorithm(api):
    return "SECRET_API_KEY" in os.environ
"""
    prog = OptimizerProgram(id="opt-018", source_code=code)
    runner = OptimizerProgramRunner()
    res = runner.run(prog, dummy_handler)

    assert res["status"] == "success"
    assert res["result"] is False
