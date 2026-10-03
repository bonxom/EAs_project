"""Bundled source loading and worker-level seed contracts."""
import pytest

from moh.core.programs import OptimizerProgram
from moh.execution.optimizer_runner import OptimizerRequest, OptimizerRunner
from moh.execution.process import Deadline, ProcessSupervisor, ProgramLimits
from moh.llm.fake import FakeLLM
from moh.llm.program_adapter import ProgramLLM
from moh.optimizers.seeds import (
    BASIC_SOURCE,
    BEST_PARENT_SOURCE,
    MULTI_TEMPERATURE_SOURCE,
)
from moh.prompts.gls_heuristic import HEURISTIC_FORMAT
from moh.prompts.program_optimizer import OPTIMIZER_FORMAT


def test_seed_sources_are_distinct_text():
    assert len({BASIC_SOURCE, BEST_PARENT_SOURCE, MULTI_TEMPERATURE_SOURCE}) == 3
    assert all('def improve_algorithm' in source
               for source in (BASIC_SOURCE, BEST_PARENT_SOURCE, MULTI_TEMPERATURE_SOURCE))


@pytest.mark.parametrize('source', [BASIC_SOURCE, MULTI_TEMPERATURE_SOURCE, BEST_PARENT_SOURCE])
@pytest.mark.parametrize('function_format', [HEURISTIC_FORMAT, OPTIMIZER_FORMAT])
def test_seed_worker_minimizes_and_preserves_batch_order(source, function_format):
    calls, evaluations = [], []
    fake = FakeLLM(42)
    facade = ProgramLLM(fake, batch_size=2)
    supervisor = ProcessSupervisor()
    limits = ProgramLimits(timeout_seconds=3, batch_size=2)
    deadline = Deadline.after(3)

    def dispatch(operation, payload, deadline):
        if operation.startswith('llm_'):
            calls.append(payload)
            if operation == 'llm_prompt':
                return facade.prompt(payload['expertise'], payload['message'],
                                     payload['temperature'], deadline=deadline)
            return facade.prompt_batch(payload['expertise'], payload['messages'],
                                       payload['temperature'], deadline=deadline)
        assert operation == 'evaluate'
        evaluations.append(payload)
        return float(len(evaluations))

    request = OptimizerRequest(OptimizerProgram('seed', source),
                               {'task': [{'best_sol': 'parent', 'idea': 'parent idea',
                                          'utility': 99.0}]},
                               'task', function_format, 42, 2)
    with supervisor.scope(deadline) as scope:
        result = OptimizerRunner(supervisor, limits).run(request, dispatch,
                                                        deadline=deadline, scope=scope)
    assert result.status == 'success', result.error
    assert result.claimed_utility == 1.0
    assert result.source_code == evaluations[0]['source_code']
    assert result.idea == evaluations[0]['idea']
    expected_function = ('def update_edge_distance' if function_format == HEURISTIC_FORMAT
                         else 'def improve_algorithm')
    assert expected_function in result.source_code
    assert len({entry['source_code'] for entry in evaluations}) == len(evaluations)
    temperatures = [entry['temperature'] for entry in calls if 'messages' in entry]
    assert temperatures == ([0.7, 1.0] if source == MULTI_TEMPERATURE_SOURCE else [1.0])
    assert all(len(entry['messages']) == 2 for entry in calls if 'messages' in entry)
