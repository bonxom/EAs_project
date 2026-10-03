import pytest

from moh.core.models import Heuristic
from moh.core.program_population import ProgramPopulation
from moh.core.programs import OptimizerProgram, ScoredProgram
from moh.execution.budgets import WorkBudget
from moh.execution.gls_runner import GLSRunner
from moh.execution.optimizer_runner import OptimizerRunner
from moh.execution.process import Deadline, ProcessSupervisor, ProgramLimits
from moh.llm.fake import FakeLLM
from moh.llm.program_adapter import ProgramLLM
from moh.optimizers.program_inner import ProgramInner
from moh.problems.tsp_gls.baselines import IDENTITY_SOURCE
from moh.problems.tsp_gls.dataset import synthetic_task
from moh.problems.tsp_gls.solver import GLSOptions


def run_inner(body, *, fake=None, llm_limit=30, eval_limit=30):
    task = synthetic_task(4, 1, 1, 42)
    budget, supervisor, events = WorkBudget(llm_limit, eval_limit), ProcessSupervisor(), []
    deadline = Deadline.after(15)
    limits = ProgramLimits()
    gls = GLSRunner(supervisor, limits, GLSOptions(iterations=2))
    emit = lambda event, payload: events.append({'event': event, **payload})
    with supervisor.scope(deadline) as scope:
        baseline = Heuristic('baseline', IDENTITY_SOURCE)
        budget.charge_evaluation()
        evaluated = gls.evaluate(baseline, task, split='validation', root_seed=42,
                                 deadline=deadline, scope=scope)
        budget.charge_instances(evaluated.counts.instance_attempts)
        original = ProgramPopulation(3).add(ScoredProgram(baseline, evaluated))
        source = ('def improve_algorithm(population, utility, language_model, '
                  'function_format, task):\n' + body)
        result = ProgramInner(OptimizerRunner(supervisor, limits), gls, budget, emit).search(
            OptimizerProgram('o1', source), task, original,
            ProgramLLM(fake or FakeLLM(42), batch_size=2), root_seed=42,
            invocation_id='inner1', deadline=deadline, scope=scope)
    return result, original, budget, events


CODE = 'import numpy as np\ndef update_edge_distance(d,t,p): return d.copy()'


def test_forged_score_rolls_back():
    result, original, _, _ = run_inner(
        f'    code = {CODE!r}\n    utility(code, "identity", task)\n'
        '    return "identity", code, 12345.0\n')
    assert result.status == 'failed'
    assert result.error == 'unverified_result'
    assert result.population == original
    assert result.counts.heuristic_evaluations == 1
    assert len(result.evaluated) == 1


@pytest.mark.parametrize('body', [
    f'    return "invented", {CODE!r}, 0.0\n',
    ('    item = population.get_best_solution(task)\n    item["utility"] = 12345.0\n'
     '    return item["idea"] or "base", item["best_sol"], item["utility"]\n'),
    f'    utility({CODE!r}, "identity", "wrong")\n',
    f'    utility({CODE!r}, "identity", task)\n    raise RuntimeError()\n',
    '    language_model.prompt("", "hello")\n',
    '    language_model.prompt("expert", "\\ud800")\n',
])
def test_bad_worker_is_contained(body):
    result, original, _, _ = run_inner(body)
    assert result.status == 'failed'
    assert result.population == original


def test_duplicate_source_reuses_authoritative_zero():
    result, _, _, events = run_inner(
        f'    code = {CODE!r}\n    score = utility(code, "identity", task)\n'
        '    assert utility(code, "again", task) == score\n'
        '    return "identity", code, score\n')
    assert result.status == 'success'
    assert result.selected.utility == 0
    assert result.counts.heuristic_evaluations == 1
    assert result.counts.instance_attempts == 1
    assert any(event['event'] == 'cache_hit' for event in events)


def test_budget_failure_rolls_back():
    result, original, _, _ = run_inner(
        f'    utility({CODE!r}, "identity", task)\n', eval_limit=1)
    assert result.status == 'failed'
    assert result.population == original
    assert result.counts.heuristic_evaluations == 0


def test_failed_heuristic_penalty_cached_but_cannot_be_selected():
    result, original, _, events = run_inner(
        '    code = "def update_edge_distance(d,t,p): raise RuntimeError()"\n'
        '    score = utility(code, "bad", task)\n'
        '    assert score == 1e6\n    assert utility(code, "bad", task) == score\n'
        '    return "bad", code, score\n')
    assert result.status == 'failed'
    assert result.error == 'unverified_result'
    assert result.population == original
    assert result.evaluated == ()
    assert result.counts.heuristic_evaluations == 1
    assert result.counts.instance_attempts == 1
    assert any(event['event'] == 'heuristic_evaluated' and event['evaluation']['status'] == 'failed'
               for event in events)


def test_generation_failure_is_charged_and_contained():
    from moh.llm.base import GenerationError

    class BrokenFake(FakeLLM):
        def generate_request(self, request):
            raise GenerationError('failed transport')

    result, original, _, _ = run_inner('    language_model.prompt("expert", "hi")\n',
                                       fake=BrokenFake(42))
    assert result.status == 'failed'
    assert result.error == 'generation'
    assert result.population == original
    assert result.counts.llm_calls == 1


def test_snapshot_selection_keeps_parent_idea_and_fitness():
    result, original, _, _ = run_inner(
        '    item = population.get_best_solution(task)\n'
        '    item["idea"] = "forged"\n'
        '    return "forged", item["best_sol"], item["utility"]\n')
    assert result.status == 'success'
    assert result.selected == original.best()
    assert result.selected.idea is None
    assert result.counts.heuristic_evaluations == 0


def test_initialization_seed_attempts_are_finite_and_keep_baselines():
    from moh.optimizers.program_inner import initialize_task_population

    class InvalidCodeFake(FakeLLM):
        def generate_request(self, request):
            if 'KIND: directions' in request.message:
                return '```json\n{"insights": ["break"]}\n```'
            return '# {broken}\n```python\ndef invalid(:\n```'

    task, supervisor, budget = synthetic_task(4, 1, 1, 42), ProcessSupervisor(), WorkBudget(20, 20)
    deadline = Deadline.after(15)
    gls = GLSRunner(supervisor, ProgramLimits(), GLSOptions(iterations=2))
    events = []
    with supervisor.scope(deadline) as scope:
        population = initialize_task_population(
            task, gls, ProgramLLM(InvalidCodeFake(42), batch_size=2), budget,
            capacity=3, seed_attempts=3, threshold=0, root_seed=42,
            deadline=deadline, scope=scope, emit=lambda event, data: events.append((event, data)))
    assert len(population.members) == 3
    assert budget.counts.llm_calls == 6
    assert budget.counts.heuristic_evaluations == 4
    assert budget.counts.instance_attempts == 4


def test_initializer_contains_wrong_direction_json_shape():
    from moh.optimizers.program_inner import initialize_task_population

    class MalformedFake(FakeLLM):
        def generate_request(self, request):
            return '```json\n["unexpected"]\n```'

    supervisor, budget = ProcessSupervisor(), WorkBudget(10, 10)
    deadline = Deadline.after(15)
    with supervisor.scope(deadline) as scope:
        population = initialize_task_population(
            synthetic_task(4, 1, 1, 42),
            GLSRunner(supervisor, ProgramLimits(), GLSOptions(iterations=2)),
            ProgramLLM(MalformedFake(42), batch_size=2), budget,
            capacity=3, seed_attempts=2, threshold=0, root_seed=42,
            deadline=deadline, scope=scope, emit=lambda event, payload: None)
    assert len(population.members) == 3
    assert budget.counts.llm_calls == 2
