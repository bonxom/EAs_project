import os
import time
from pathlib import Path

import numpy as np
import pytest

from moh.core.models import EvaluationContext, Heuristic
from moh.core.seeds import derive_seed
from moh.execution.heuristic_runner import HeuristicRunner
from moh.execution.protocol import ExecutionLimits
from moh.problems.baselines import NEAREST_NEIGHBOR_SOURCE, RANDOM_CHOICE_SOURCE
from moh.problems.tsp import TSPTask


def evaluate(source, *, size=10, limits=None, task=None):
    task = task or TSPTask.create(size, 1, 42)
    context = EvaluationContext(
        task.id,
        task.instance_seeds,
        tuple(
            derive_seed(42, "task", size, "instance", i, "evaluation", 0)
            for i in range(len(task.instances))
        ),
    )
    return HeuristicRunner(limits or ExecutionLimits()).evaluate(
        Heuristic("h1", source), task, context
    )


@pytest.mark.parametrize("source", [NEAREST_NEIGHBOR_SOURCE, RANDOM_CHOICE_SOURCE])
@pytest.mark.parametrize("size", [10, 20, 50])
def test_baselines(source, size):
    a = evaluate(source, size=size)
    assert a.status == "success"
    assert a == evaluate(source, size=size)


@pytest.mark.parametrize(
    "source,code",
    [
        ("!syntax", "syntax"),
        ("x=1", "missing_function"),
        ("select_next_node=2", "missing_function"),
        ("raise RuntimeError()", "exception"),
        ("import os; os._exit(9)", "process_exit"),
        ("while True: pass", "timeout"),
        ("def select_next_node(c,u,x):\n while True: pass", "timeout"),
        ('while True: print("x"*10000)', "output_limit"),
    ]
    + [
        (f"def select_next_node(c,u,x): return {value}", "invalid_return")
        for value in ["True", "1.0", "999", "0"]
    ]
    + [("def select_next_node(c,u,x): raise ValueError()", "exception")],
)
def test_failures(source, code):
    result = evaluate(source, limits=ExecutionLimits(timeout_seconds=1.0))
    assert result.status == "failed"
    assert result.utility is None
    assert result.error == code


def test_module_pid_and_fresh_globals(tmp_path):
    marker = tmp_path / "pid"
    source = f'import os\nopen({str(marker)!r}, "w").write(str(os.getpid()))\ncounter=0\ndef select_next_node(c,u,x):\n global counter\n counter+=1\n assert counter<10\n return min(u)'
    assert evaluate(source).status == "success"
    assert int(marker.read_text()) != os.getpid()
    assert evaluate(source).status == "success"


def test_mutating_inputs_cannot_change_score():
    source = "def select_next_node(c,u,x):\n x[:]=0\n result=min(u)\n u.clear()\n return result"
    task = TSPTask.create(10, 1, 42)
    assert evaluate(source, task=task).lengths == (
        task.score_tour(0, list(range(10)) + [0]),
    )


def test_tie_smallest():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]])
    task = TSPTask("square", (points,), (1,))
    assert evaluate(NEAREST_NEIGHBOR_SOURCE, task=task).lengths == (6.0,)
    from moh.execution.sandbox import run_worker

    result = run_worker(
        {
            "id": "tie",
            "source": NEAREST_NEIGHBOR_SOURCE,
            "coordinates": points.tolist(),
            "seed": 1,
        },
        ExecutionLimits(),
    )
    assert result.tour == (0, 1, 3, 2, 0)


@pytest.mark.parametrize(
    "limits,source,code",
    [
        (ExecutionLimits(source_bytes=2), "abc", "source_limit"),
        (ExecutionLimits(result_bytes=2), NEAREST_NEIGHBOR_SOURCE, "result_limit"),
    ],
)
def test_limits(limits, source, code):
    assert evaluate(source, limits=limits).error == code


def test_descendant_reaped(tmp_path):
    marker = tmp_path / "child"
    source = f'import os,time\npid=os.fork()\nif pid==0:\n open({str(marker)!r},"w").write(str(os.getpid()))\n time.sleep(60)\nelse:\n os._exit(0)'
    started = time.monotonic()
    try:
        result = evaluate(source, limits=ExecutionLimits(timeout_seconds=1.0))
        assert result.status == "failed"
        assert time.monotonic() - started < 5
        if marker.exists():
            pid = int(marker.read_text())
            assert not Path(f"/proc/{pid}").exists()
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), 9)
            except ProcessLookupError:
                pass


def test_malformed_failure_envelope_does_not_escape_runner():
    source = (
        "import os,sys\n"
        'os.write(int(sys.argv[1]), b\'{"id":"h1","status":"failed","tour":null,"error":{}}\')\n'
        "os._exit(0)"
    )
    result = evaluate(source)
    assert result.status == "failed"
    assert result.error == "protocol"


def test_invalid_unicode_source_is_candidate_failure():
    result = evaluate("\ud800")
    assert result.status == "failed"
    assert result.utility is None
    assert result.error == "syntax"
