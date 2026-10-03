import pytest

from moh.core.models import Heuristic
from moh.execution.gls_runner import GLSRunner
from moh.execution.process import Deadline, ProcessSupervisor, ProgramLimits
from moh.problems.tsp_gls.baselines import IDENTITY_SOURCE
from moh.problems.tsp_gls.dataset import synthetic_task
from moh.problems.tsp_gls.evaluation import gap_percent
from moh.problems.tsp_gls.solver import GLSOptions


def evaluate(source=IDENTITY_SOURCE, *, limits=None, split='validation', task=None):
    task = synthetic_task(4, 2, 1, 42) if task is None else task
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(10)
    with supervisor.scope(deadline) as scope:
        return GLSRunner(supervisor, limits or ProgramLimits(), GLSOptions(iterations=2)).evaluate(
            Heuristic('h1', source), task, split=split, root_seed=42,
            deadline=deadline, scope=scope,
        )


def test_identity_worker_has_valid_mean_gap():
    result = evaluate()
    assert result.status == 'success'
    assert result.utility == pytest.approx(sum(result.gaps) / 2)
    assert result.counts.instance_attempts == 2
    assert result.counts.heuristic_evaluations == 1


def test_gap_uses_minimization_and_rejects_bad_reference():
    assert gap_percent(105, 100) == pytest.approx(5)
    with pytest.raises(ValueError, match='reference'):
        gap_percent(90, 100)


@pytest.mark.parametrize('source,error', [
    ('def broken(', 'syntax'),
    ('x = 1', 'missing_function'),
    ('raise SystemExit(3)', 'exception'),
    ('import os\nos._exit(7)', 'process_exit'),
    ('def update_edge_distance(*args):\n    raise RuntimeError("bad")', 'exception'),
    ('def update_edge_distance(d, t, p):\n    return d * float("nan")', 'exception'),
    ('def update_edge_distance(d, t, p):\n    return d * float("inf")', 'exception'),
    ('def update_edge_distance(d, t, p):\n    d[0, 1] += 1\n    return d', 'exception'),
    ('def update_edge_distance(d, t, p):\n    return d[:-1]', 'exception'),
    ('def update_edge_distance(d, t, p):\n    return -d', 'exception'),
    ('def update_edge_distance(d, t, p):\n    return "wrong"', 'exception'),
])
def test_bad_candidates_fail_without_crashing_experiment(source, error):
    result = evaluate(source)
    assert result.status == 'failed'
    assert result.error == error
    assert result.utility is None
    assert result.counts.instance_attempts == 1
    assert result.counts.heuristic_evaluations == 1
    assert result.costs == result.gaps == result.tours == ()
    assert evaluate().status == 'success'


@pytest.mark.parametrize('source', [
    'while True:\n    pass',
    'def update_edge_distance(*args):\n    while True:\n        pass',
])
def test_module_and_callback_infinite_loops_timeout(source):
    result = evaluate(source, limits=ProgramLimits(timeout_seconds=0.5))
    assert result.status == 'failed'
    assert result.error == 'timeout'
    assert result.counts.instance_attempts == 1
    assert evaluate().status == 'success'


def test_source_limit_rejected_before_any_instance():
    result = evaluate(limits=ProgramLimits(source_bytes=1))
    assert result.error == 'source_limit'
    assert result.counts.instance_attempts == 0
    assert result.counts.heuristic_evaluations == 1


def test_request_limit_rejected_before_any_instance():
    result = evaluate(limits=ProgramLimits(request_bytes=1))
    assert result.error == 'request_limit'
    assert result.counts.instance_attempts == 0


def test_heuristic_mutation_cannot_change_trusted_inputs():
    task = synthetic_task(6, 2, 1, 42)
    matrices = [instance.distances.copy() for instance in task.validation]
    source = '''import numpy as np

def update_edge_distance(d, t, p):
    d[:] = 0
    t[:] = 0
    p[:] = 100000
    return np.zeros_like(d)
'''
    result = evaluate(source, task=task)
    assert result.status == 'success'
    from moh.problems.tsp_gls.tour import tour_cost
    for original, instance, tour, cost in zip(
        matrices, task.validation, result.tours, result.costs, strict=True
    ):
        assert (instance.distances == original).all()
        assert cost == tour_cost(original, tour)
        assert cost > 0


def test_generated_module_executes_only_in_worker(tmp_path):
    import os
    marker = tmp_path / 'pid'
    source = f'import os\nfrom pathlib import Path\nPath({str(marker)!r}).write_text(str(os.getpid()))\n' + IDENTITY_SOURCE
    assert evaluate(source).status == 'success'
    assert int(marker.read_text()) != os.getpid()


def test_gls_worker_forbidden_callback():
    source = '''import os, json, sys
message = {'id': 1, 'op': 'llm_prompt', 'payload': {
    'expertise': '', 'message': '', 'temperature': 1}}
os.write(int(sys.argv[1]), (json.dumps(message) + '\\n').encode())
'''
    result = evaluate(source)
    assert result.status == 'failed'
    assert result.error == 'protocol'
    assert result.counts.llm_calls == 0


class ScriptedSupervisor(ProcessSupervisor):
    def __init__(self, results):
        self.results = iter(results)
        self.requests = []
        self.deadlines = []

    def exchange(self, module, request, dispatch, *, limits, deadline, scope):
        self.requests.append(request)
        self.deadlines.append((deadline, scope))
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


def scripted_evaluate(supervisor, task=None, *, deadline=None, split='validation'):
    task = synthetic_task(4, 2, 1, 42) if task is None else task
    deadline = Deadline.after(10) if deadline is None else deadline
    with supervisor.scope(deadline) as scope:
        return GLSRunner(supervisor, ProgramLimits(timeout_seconds=2), GLSOptions()).evaluate(
            Heuristic('h', IDENTITY_SOURCE), task, split=split, root_seed=42,
            deadline=deadline, scope=scope,
        )


def test_failure_retains_completed_results_and_counts_attempts():
    task = synthetic_task(4, 2, 1, 42)
    supervisor = ScriptedSupervisor([
        {'status': 'success', 'tour': list(task.validation[0].optimal_tour), 'error': None},
        {'status': 'failed', 'tour': None, 'error': 'exception'},
    ])
    result = scripted_evaluate(supervisor, task)
    assert result.status == 'failed'
    assert len(result.costs) == len(result.gaps) == len(result.tours) == 1
    assert result.utility is None
    assert result.counts.instance_attempts == 2


@pytest.mark.parametrize('result', [
    {'status': 'success', 'tour': [0, 1, 2, 3, 0], 'error': None, 'cost': 0},
    {'status': 'success', 'tour': [0, 1, 2, 2, 0], 'error': None},
    {'status': 'success', 'tour': [0, True, 2, 3, 0], 'error': None},
    {'status': 'success', 'tour': [0, 1.0, 2, 3, 0], 'error': None},
    {'status': 'success', 'tour': [0, 1, 2, 10**400, 0], 'error': None},
    {'status': 'success', 'tour': None, 'error': None},
    {'status': 'failed', 'tour': [0], 'error': 'exception'},
    {'status': 'failed', 'tour': None, 'error': ''},
    {'status': 'unknown', 'tour': None, 'error': None},
])
def test_untrusted_finish_is_rejected(result):
    evaluated = scripted_evaluate(ScriptedSupervisor([result]))
    assert evaluated.status == 'failed'
    assert evaluated.utility is None
    assert evaluated.counts.instance_attempts == 1


def test_parent_reference_inconsistency_propagates():
    task = synthetic_task(4, 2, 1, 42)
    # Simulate corrupted infrastructure after the dataset validator.
    object.__setattr__(task.validation[0], 'optimal_cost', task.validation[0].optimal_cost * 2)
    supervisor = ScriptedSupervisor([
        {'status': 'success', 'tour': list(task.validation[0].optimal_tour), 'error': None},
    ])
    with pytest.raises(ValueError, match='reference'):
        scripted_evaluate(supervisor, task)


def test_test_split_uses_held_out_instances_and_derived_seeds():
    from moh.core.seeds import derive_seed
    task = synthetic_task(4, 2, 1, 42)
    supervisor = ScriptedSupervisor([
        {'status': 'success', 'tour': list(task.test[0].optimal_tour), 'error': None},
    ])
    deadline = Deadline.after(1)
    result = scripted_evaluate(supervisor, task, deadline=deadline, split='test')
    assert result.status == 'success'
    assert result.split == 'test'
    assert result.counts.instance_attempts == 1
    request = supervisor.requests[0]
    assert request['distances'] == task.test[0].distances.tolist()
    assert request['seed'] == derive_seed(42, 'gls', task.id, 'test', task.test[0].id, 'worker')
    invocation, scope = supervisor.deadlines[0]
    assert invocation.expires_at <= deadline.expires_at
    assert scope.parent is not None


@pytest.mark.parametrize('cost,optimum', [(0, 1), (1, 0), (float('nan'), 1),
                                        (1, float('inf')), (True, 1), (10**400, 1)])
def test_gap_rejects_invalid_costs(cost, optimum):
    with pytest.raises(ValueError):
        gap_percent(cost, optimum)


def test_gap_clamps_roundoff_below_reference():
    assert gap_percent(100 - 1e-7, 100) == 0


def test_weighted_gap_requires_selected_successes_and_positive_weights():
    from moh.core.programs import ScoredProgram, TaskOutcome
    from moh.problems.tsp_gls.evaluation import weighted_gap
    result = evaluate()
    outcome = TaskOutcome(result.task_id, ScoredProgram(Heuristic('h1', IDENTITY_SOURCE), result))
    assert weighted_gap((outcome,), (2,)) == result.utility
    assert weighted_gap((outcome, outcome), (1e308, 1e308)) == result.utility
    for weights in [(), (0,), (-1,), (float('nan'),), (True,), (10**400,)]:
        with pytest.raises(ValueError):
            weighted_gap((outcome,), weights)
    with pytest.raises(ValueError):
        weighted_gap((TaskOutcome(result.task_id, None),), (1,))
