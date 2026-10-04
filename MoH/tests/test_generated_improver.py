import multiprocessing
import os

import pytest
from utils.generated_improver import GeneratedImprover
from utils.population import Pop


class FakeLLM:
    batch_size = 2
    model = "fake"
    temperature = 0.0

    def __init__(self):
        self.calls = []

    def prompt(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "fake-response"

    def prompt_batch(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ["fake-response"]


def test_generated_improver_runs_in_child_and_calls_parent_services():
    code = '''import os
def improve_algorithm(population, utility, language_model, function_format, task):
    response = language_model.prompt("expert", function_format, temperature=0)
    score = utility(response, "idea", task)
    return os.getpid(), response, score
'''
    calls = []
    llm = FakeLLM()

    def utility(*args):
        calls.append(args)
        return 0.5

    pid, response, score = GeneratedImprover(code, timeout=10)(
        Pop(["task"], 2), utility, llm, "format", "task",
    )
    assert pid != os.getpid()
    assert response == "fake-response"
    assert score == 0.5
    assert calls == [("fake-response", "idea", "task")]
    assert llm.calls == [(("expert", "format"), {"temperature": 0})]


def test_loading_generated_code_has_timeout():
    with pytest.raises(TimeoutError):
        GeneratedImprover("while True:\n    pass", timeout=1).validate()


def test_running_generated_code_has_timeout():
    candidate = GeneratedImprover(
        "def improve_algorithm(*args):\n    while True:\n        pass", timeout=1,
    )
    with pytest.raises(TimeoutError):
        candidate(Pop(["task"], 1), lambda *_: 0, FakeLLM(), "format", "task")


def test_rejects_missing_function():
    with pytest.raises(RuntimeError, match="callable improve_algorithm"):
        GeneratedImprover("other_function = lambda: 1", timeout=10).validate()


def test_reports_child_crash():
    with pytest.raises(RuntimeError, match="exited"):
        GeneratedImprover("import os\nos._exit(3)", timeout=10).validate()


def test_seed_reproduces_generated_population_selection():
    code = '''import random
import numpy as np
def improve_algorithm(*args):
    return random.random(), float(np.random.random())
'''
    candidate = GeneratedImprover(code, timeout=10, seed=42)
    arguments = (Pop(["task"], 1), lambda *_: 0, FakeLLM(), "format", "task")
    assert candidate(*arguments) == candidate(*arguments)


def test_extraction_failure_from_child_is_logged_in_parent(caplog):
    code = '''from utils.utils import extract_code
def improve_algorithm(*args):
    return extract_code("No candidate code")
'''
    result = GeneratedImprover(code, timeout=10)(
        Pop(["task"], 1), lambda *_: 0, FakeLLM(), "format", "task",
    )
    assert result is None
    assert "No code extracted" in caplog.text


def test_watchdog_terminates_child_during_parent_callback():
    code = '''import os
def improve_algorithm(population, utility, language_model, function_format, task):
    return utility(os.getpid())
'''
    terminated = []

    def utility(pid):
        child = next(process for process in multiprocessing.active_children() if process.pid == pid)
        child.join(timeout=3)
        terminated.append(not child.is_alive())
        return 0

    with pytest.raises(TimeoutError):
        GeneratedImprover(code, timeout=2)(
            Pop(["task"], 1), utility, FakeLLM(), "format", "task",
        )
    assert terminated == [True]
