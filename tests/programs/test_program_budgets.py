import pytest

from moh.core.models import WorkCounts
from moh.execution.budgets import WorkBudget
from moh.execution.process import CandidateFailure


def test_batch_precheck_is_atomic():
    budget = WorkBudget(1, 2)
    with pytest.raises(CandidateFailure, match='budget'):
        budget.require_llm(2)
    assert budget.counts == WorkCounts()


def test_global_limits_and_delta():
    budget = WorkBudget(1, 1)
    before = budget.counts
    budget.charge_llm()
    budget.charge_evaluation()
    budget.charge_instances(2)
    assert budget.delta(before) == WorkCounts(1, 2, 1)
    for charge in (budget.charge_llm, budget.charge_evaluation):
        with pytest.raises(CandidateFailure, match='budget'):
            charge()
    assert budget.counts == WorkCounts(1, 2, 1)


@pytest.mark.parametrize('payload', [
    {'expertise': '', 'messages': ['ok'], 'temperature': None},
    {'expertise': 'x', 'messages': ['ok', ''], 'temperature': None},
    {'expertise': 'x', 'messages': ['ok', '\ud800'], 'temperature': None},
    {'expertise': 'x', 'messages': ['ok'], 'temperature': 10**400},
])
def test_invalid_batch_never_calls_provider(payload):
    from moh.execution.process import Deadline
    from moh.execution.program_callbacks import InvocationDispatcher
    from moh.llm.fake import FakeLLM
    from moh.llm.program_adapter import ProgramLLM

    budget = WorkBudget(20, 20)
    fake = FakeLLM(42)
    dispatcher = InvocationDispatcher('task', ProgramLLM(fake, batch_size=2),
                                      budget, None, max_batch_size=2)
    with pytest.raises(CandidateFailure):
        dispatcher.dispatch('llm_batch', payload, Deadline.after(10))
    assert budget.counts.llm_calls == 0


@pytest.mark.parametrize('error', [OSError('disk'), ValueError('trusted provider state')])
def test_provider_infrastructure_error_propagates(error):
    from moh.execution.process import Deadline
    from moh.execution.program_callbacks import InvocationDispatcher
    from moh.llm.fake import FakeLLM
    from moh.llm.program_adapter import ProgramLLM

    class BrokenFake(FakeLLM):
        def generate_request(self, request):
            raise error

    budget = WorkBudget(2, 2)
    dispatcher = InvocationDispatcher('task', ProgramLLM(BrokenFake(42), batch_size=2),
                                      budget, None, max_batch_size=2)
    with pytest.raises(type(error), match=str(error)):
        dispatcher.dispatch('llm_prompt', {'expertise': 'x', 'message': 'x',
                                          'temperature': None}, Deadline.after(10))
    assert budget.counts.llm_calls == 1


def test_entire_batch_budget_checked_before_provider():
    from moh.execution.process import Deadline
    from moh.execution.program_callbacks import InvocationDispatcher
    from moh.llm.fake import FakeLLM
    from moh.llm.program_adapter import ProgramLLM

    budget = WorkBudget(1, 2)
    dispatcher = InvocationDispatcher('task', ProgramLLM(FakeLLM(42), batch_size=2),
                                      budget, None, max_batch_size=2)
    with pytest.raises(CandidateFailure, match='llm_budget'):
        dispatcher.dispatch('llm_batch', {'expertise': 'x', 'messages': ['x', 'y'],
                                         'temperature': None}, Deadline.after(10))
    assert budget.counts.llm_calls == 0
