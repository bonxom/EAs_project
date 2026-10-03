import hashlib

import pytest

from moh.core.programs import OptimizerProgram
from moh.execution.budgets import WorkBudget
from moh.execution.gls_runner import GLSRunner
from moh.execution.optimizer_runner import OptimizerRunner
from moh.execution.process import Deadline, ProcessSupervisor, ProgramLimits
from moh.llm.fake import FakeLLM
from moh.llm.program_adapter import ProgramLLM
from moh.optimizers.program_inner import ProgramInner, initialize_task_population
from moh.optimizers.program_meta import ProgramEvaluator, ProgramMeta
from moh.problems.tsp_gls.dataset import synthetic_task
from moh.problems.tsp_gls.solver import GLSOptions

GENERATOR = '''from moh.optimizers.helpers import extract_code, extract_idea

def improve_algorithm(population, utility, language_model, function_format, task):
    response = language_model.prompt("initial marker", function_format)
    code, idea = extract_code(response), extract_idea(response)
    return idea, code, utility(code, idea, task)
'''
VARIANT = GENERATOR.replace('initial marker', 'variant marker')
BEST = '''def improve_algorithm(population, utility, language_model, function_format, task):
    item = population.get_best_solution(task)
    return item["idea"] or "best", item["best_sol"], item["utility"]
'''
FAIL = '''def improve_algorithm(population, utility, language_model, function_format, task):
    raise RuntimeError("bad optimizer")
'''
HEURISTIC = 'import numpy as np\ndef update_edge_distance(d,t,p): return d.copy() + 0.1*p'
FAIL_SECOND = f'''def improve_algorithm(population, utility, language_model, function_format, task):
    if task.endswith("5"):
        raise RuntimeError("second task")
    code = {HEURISTIC!r}
    return "new", code, utility(code, "new", task)
'''


def response(source):
    return f'# {{scripted optimizer}}\n```python\n{source}\n```'


def fixture_parts(tasks=2, limits=None, budget=None):
    supervisor, events = ProcessSupervisor(), []
    budget = budget or WorkBudget(100, 100)
    limits = limits or ProgramLimits()
    task_list = tuple(synthetic_task(size, 1, 1, 42) for size in (4, 5)[:tasks])
    optimizer = OptimizerRunner(supervisor, limits)
    gls = GLSRunner(supervisor, limits, GLSOptions(iterations=2))
    observed = []

    class ObservedFake(FakeLLM):
        def generate_request(self, request):
            observed.append((request.expertise, request.message))
            return super().generate_request(request)

    factory = lambda scope: ProgramLLM(ObservedFake(42), batch_size=2)
    emit = lambda event, data: events.append({'event': event, **data})
    inner = ProgramInner(optimizer, gls, budget, emit)
    evaluator = ProgramEvaluator(task_list, tuple(float(task.size) for task in task_list),
                                 inner, factory, budget, emit)
    return supervisor, budget, optimizer, gls, factory, evaluator, events, emit, observed


def run_meta_fixture(meta_sources=None, iterations=2, tasks=2, seeds=None, capacity=3, budget=None):
    parts = fixture_parts(tasks, budget=budget)
    supervisor, budget, optimizer, gls, factory, evaluator, events, emit, observed = parts
    sources = meta_sources if meta_sources is not None else [VARIANT] * 10
    meta = ProgramLLM(FakeLLM(42, {'optimizer_program': [response(s) for s in sources]}),
                      batch_size=2)
    deadline = Deadline.after(30)
    with supervisor.scope(deadline) as scope:
        populations = {task.id: initialize_task_population(
            task, gls, factory((task.id,)), budget, capacity=3, seed_attempts=0,
            threshold=None, root_seed=42, deadline=deadline, scope=scope, emit=emit)
            for task in evaluator.tasks}
        result = ProgramMeta(optimizer, evaluator, budget, emit).search(
            seeds if seeds is not None else (OptimizerProgram('seed1', GENERATOR),
                                             OptimizerProgram('seed2', BEST)),
            populations, meta, iterations=iterations, capacity=capacity,
            root_seed=42, deadline=deadline, scope=scope)
    return result, events, observed


def test_next_round_executes_selected_program():
    result, events, observed = run_meta_fixture()
    rounds = [e for e in events if e['event'] == 'meta_round_started']
    changed = [e for e in events if e['event'] == 'active_optimizer_changed']
    assert len(rounds) == 2
    assert rounds[1]['active_id'] == changed[0]['optimizer_id']
    assert rounds[1]['source_hash'] == changed[0]['source_hash']
    assert rounds[1]['source_hash'] != rounds[0]['source_hash']
    assert rounds[1]['source_hash'] == hashlib.sha256(VARIANT.strip().encode()).hexdigest()
    assert any(expertise == 'variant marker' for expertise, _ in observed)
    assert result.winner.utility == min(item.utility for item in result.population)


def test_failed_second_task_rolls_back_first():
    result, events, _ = run_meta_fixture(meta_sources=[FAIL_SECOND] * 10)
    failure = next(e for e in events if e['event'] == 'optimizer_evaluated'
                   and e['status'] == 'failed')
    assert failure['population_before'] == failure['population_after']
    assert failure['counts']['heuristic_evaluations'] > 0
    assert any(e['event'] == 'meta_round_finished' for e in events)
    assert result.status == 'success'


def test_zero_rounds_and_all_bad_seeds():
    result, events, _ = run_meta_fixture(iterations=0)
    assert result.status == 'success'
    assert not any(e['event'] == 'meta_round_started' for e in events)
    result, events, _ = run_meta_fixture(iterations=1,
                                       seeds=(OptimizerProgram('bad', FAIL),))
    assert result.status == 'failed'
    assert result.population == ()
    assert result.winner is result.active is None


def test_invalid_optimizer_then_recovery():
    result, events, _ = run_meta_fixture(meta_sources=[FAIL, VARIANT], iterations=2,
                                       seeds=(OptimizerProgram('seed1', GENERATOR),))
    rounds = [e for e in events if e['event'] == 'meta_round_finished']
    assert [e['status'] for e in rounds] == ['failed', 'success']
    assert result.active.source_code == VARIANT.strip()


@pytest.mark.parametrize('forged', [True, False])
def test_outer_callback_then_failure_discards_all_inner_commits(forged):
    outer = GENERATOR.replace('    return idea, code, utility(code, idea, task)',
                              '    score = utility(code, idea, task)\n' +
                              ('    return idea, code, score + 1\n' if forged else
                               '    raise RuntimeError("abort after assessment")\n'))
    # The seed must succeed at the inner level; only the outer path aborts.
    outer = outer.replace('    score = utility(code, idea, task)',
                          '    score = utility(code, idea, task)\n'
                          '    if task != "meta-optimizer": return idea, code, score')
    result, events, _ = run_meta_fixture(meta_sources=[FAIL_SECOND.replace(
        '    if task.endswith("5"):', '    if False:')], iterations=1,
        seeds=(OptimizerProgram('seed1', outer),))
    round_end = next(e for e in events if e['event'] == 'meta_round_finished')
    assert round_end['status'] == 'failed'
    assert round_end['population_before'] == round_end['population_after']
    assert result.active.id == 'seed1'
    assert not any(item.candidate.source_code == HEURISTIC
                   for pop in result.task_populations.values() for item in pop.members)


def test_duplicate_optimizer_callback_assessed_once():
    duplicate = GENERATOR.replace('    return idea, code, utility(code, idea, task)',
                                  '    score = utility(code, idea, task)\n'
                                  '    assert utility(code, idea, task) == score\n'
                                  '    return idea, code, score')
    result, events, _ = run_meta_fixture(iterations=1, tasks=1,
                                       seeds=(OptimizerProgram('duplicate', duplicate),))
    assert result.status == 'success'
    assessments = [e for e in events if e['event'] == 'optimizer_evaluated'
                   and e['invocation_id'].startswith('meta-round')]
    assert len(assessments) == 1
    assert any(e['event'] == 'cache_hit' and e.get('kind') == 'optimizer' for e in events)


def test_global_llm_budget_is_not_renewed_between_tasks_or_rounds():
    budget = WorkBudget(1, 100)
    result, events, _ = run_meta_fixture(iterations=3, budget=budget)
    assert result.status == 'success'
    assert result.active.id == 'seed2'
    assert result.counts.llm_calls == 1
    assert not any(e['event'] == 'meta_round_started' for e in events)
    failure = next(e for e in events if e['event'] == 'optimizer_evaluated'
                   and e['status'] == 'failed')
    assert failure['evaluation']['error'] == 'llm_budget'


def test_weighted_task_gap_and_counts():
    from moh.core.models import Heuristic, WorkCounts
    from moh.core.program_population import ProgramPopulation
    from moh.core.programs import GapEvaluation, ScoredProgram
    from moh.optimizers.program_inner import InnerProgramResult

    tasks = tuple(synthetic_task(size, 1, 1, 42) for size in (4, 5))
    budget, supervisor = WorkBudget(10, 10), ProcessSupervisor()
    scores = {tasks[0].id: 2.0, tasks[1].id: 8.0}

    class FixedInner:
        def search(self, program, task, population, llm, **kwargs):
            budget.charge_evaluation()
            budget.charge_instances(1)
            heuristic = Heuristic(f'h-{task.size}', HEURISTIC)
            gap = GapEvaluation(heuristic.id, task.id, 'validation', 'success', scores[task.id],
                                (1.0,), (scores[task.id],), (tuple(range(task.size)),),
                                counts=WorkCounts(1, 1, 0))
            item = ScoredProgram(heuristic, gap)
            return InnerProgramResult('success', item, population.add(item), (item,), None,
                                      WorkCounts(1, 1, 0))

    evaluator = ProgramEvaluator(tasks, (4.0, 5.0), FixedInner(),
                                 lambda labels: ProgramLLM(FakeLLM(42), batch_size=2),
                                 budget, lambda event, payload: None)
    deadline = Deadline.after(10)
    with supervisor.scope(deadline) as scope:
        evaluation, populations = evaluator.evaluate(
            OptimizerProgram('best', BEST), {task.id: ProgramPopulation(3) for task in tasks},
            root_seed=42, invocation_id='weighted', deadline=deadline, scope=scope)
    assert evaluation.utility == pytest.approx((4 * 2 + 5 * 8) / 9)
    assert evaluation.counts == WorkCounts(2, 2, 0)
    assert budget.counts == WorkCounts(2, 2, 0)
    assert all(pop.best() is not None for pop in populations.values())


def test_worse_valid_active_is_retained_outside_best_population():
    from moh.core.programs import ProgramEvaluation, ScoredProgram, TaskOutcome

    parts = fixture_parts(tasks=1)
    supervisor, budget, optimizer, gls, factory, evaluator, _events, emit, _ = parts
    deadline = Deadline.after(20)
    with supervisor.scope(deadline) as scope:
        populations = {task.id: initialize_task_population(
            task, gls, factory((task.id,)), budget, capacity=3, seed_attempts=0,
            threshold=None, root_seed=42, deadline=deadline, scope=scope, emit=emit)
            for task in evaluator.tasks}

        class FixedEvaluator:
            def evaluate(self, program, populations, **kwargs):
                # Ranking is isolated from GLS performance; the outer optimizer
                # still runs in a real worker and chooses the verified callback score.
                outcome = TaskOutcome(evaluator.tasks[0].id, next(iter(populations.values())).best())
                score = 1.0 if program.id == 'seed1' else 9.0
                evaluation = ProgramEvaluation(program.id, 'success', score, (outcome,))
                return evaluation, dict(populations)

        meta_llm = ProgramLLM(FakeLLM(42, {'optimizer_program': [response(VARIANT)]}), batch_size=2)
        result = ProgramMeta(optimizer, FixedEvaluator(), budget, emit).search(
            (OptimizerProgram('seed1', GENERATOR),), populations, meta_llm,
            iterations=1, capacity=1, root_seed=42, deadline=deadline, scope=scope)
    assert result.active.utility == 9
    assert result.winner.utility == 1
    assert result.active not in result.population
    assert isinstance(result.active, ScoredProgram)


def test_nested_deadline_kills_inner_process_and_rolls_back(tmp_path):
    import os
    import time

    pid_path = tmp_path / 'inner.pid'
    sleeping = (f'import os, time\nopen({str(pid_path)!r}, "w").write(str(os.getpid()))\n'
                'time.sleep(60)\n' + BEST)
    outer = BEST.replace('    item = population.get_best_solution(task)',
                         f'    if task == "meta-optimizer":\n'
                         f'        return "sleep", {sleeping!r}, utility({sleeping!r}, "sleep", task)\n'
                         '    item = population.get_best_solution(task)')
    parts = fixture_parts(tasks=1, limits=ProgramLimits(timeout_seconds=0.8))
    supervisor, budget, optimizer, gls, factory, evaluator, events, emit, _ = parts
    deadline = Deadline.after(20)
    started = time.monotonic()
    with supervisor.scope(deadline) as scope:
        populations = {task.id: initialize_task_population(
            task, gls, factory((task.id,)), budget, capacity=3, seed_attempts=0,
            threshold=None, root_seed=42, deadline=deadline, scope=scope, emit=emit)
            for task in evaluator.tasks}
        original = dict(populations)
        result = ProgramMeta(optimizer, evaluator, budget, emit).search(
            (OptimizerProgram('sleep-driver', outer),), populations,
            ProgramLLM(FakeLLM(42), batch_size=2), iterations=1, capacity=3,
            root_seed=42, deadline=deadline, scope=scope)
    assert time.monotonic() - started < 6
    assert pid_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)
    assert dict(result.task_populations) == original
    assert any(e['event'] == 'meta_round_finished' and e['status'] == 'failed' for e in events)


@pytest.mark.parametrize('outer_body', [
    f'        return "unassessed", {FAIL!r}, 0.0',
    f'        utility({BEST!r}, "wrong task", "tsp_gls4")',
])
def test_outer_unassessed_source_and_wrong_task_are_contained(outer_body):
    source = BEST.replace('    item = population.get_best_solution(task)',
                          '    if task == "meta-optimizer":\n' + outer_body + '\n'
                          '    item = population.get_best_solution(task)')
    result, events, _ = run_meta_fixture(iterations=1, tasks=1,
                                       seeds=(OptimizerProgram('seed1', source),))
    round_end = next(e for e in events if e['event'] == 'meta_round_finished')
    assert round_end['status'] == 'failed'
    assert result.active.id == 'seed1'


def test_failed_callback_can_recover_in_same_outer_invocation():
    outer = BEST.replace('    item = population.get_best_solution(task)',
                         f'    if task == "meta-optimizer":\n'
                         f'        assert utility({FAIL!r}, "fail", task) == 1e6\n'
                         f'        assert utility({FAIL!r}, "fail again", task) == 1e6\n'
                         f'        return "valid", {VARIANT!r}, utility({VARIANT!r}, "valid", task)\n'
                         '    item = population.get_best_solution(task)')
    result, events, _ = run_meta_fixture(iterations=1, tasks=1,
                                       seeds=(OptimizerProgram('seed1', outer),))
    assert result.active.source_code == VARIANT
    candidates = [e for e in events if e['event'] == 'optimizer_evaluated'
                  and e['invocation_id'].startswith('meta-round')]
    assert [e['status'] for e in candidates] == ['failed', 'success']
    assert any(e['event'] == 'cache_hit' and e.get('kind') == 'optimizer' for e in events)


def test_evaluator_contains_cancelled_scope_but_propagates_infrastructure():
    parts = fixture_parts(tasks=1)
    supervisor, budget, _, _, factory, evaluator, events, _, _ = parts
    from moh.core.program_population import ProgramPopulation

    task = evaluator.tasks[0]
    populations = {task.id: ProgramPopulation(3)}
    deadline = Deadline.after(10)
    with supervisor.scope(deadline) as scope:
        scope.abort('timeout')
        result, after = evaluator.evaluate(
            OptimizerProgram('best', BEST), populations, root_seed=42,
            invocation_id='cancelled', deadline=deadline, scope=scope)
    assert result.status == 'failed'
    assert result.error == 'timeout'
    assert after == populations
    assert result.counts == budget.counts

    class BrokenInner:
        def search(self, *args, **kwargs):
            raise OSError('trusted infrastructure')

    broken = ProgramEvaluator(evaluator.tasks, evaluator.weights, BrokenInner(), factory,
                              budget, lambda event, payload: None)
    with (supervisor.scope(deadline) as scope,
          pytest.raises(OSError, match='trusted infrastructure')):
        broken.evaluate(OptimizerProgram('best', BEST), populations, root_seed=42,
                        invocation_id='infra', deadline=deadline, scope=scope)
    assert populations[task.id].members == ()
    assert any(e['event'] == 'optimizer_evaluated' for e in events)
