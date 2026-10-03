import pytest

from moh.llm.base import GenerationError, LLMRequest
from moh.llm.fake import FakeLLM
from moh.optimizers.helpers import extract_code, extract_idea
from moh.prompts.gls_heuristic import HEURISTIC_FORMAT
from moh.prompts.program_optimizer import OPTIMIZER_FORMAT


def test_fake_distinguishes_program_and_heuristic_code():
    fake = FakeLLM(42)
    outer = fake.generate_request(LLMRequest('expert', OPTIMIZER_FORMAT, 0.7))
    inner = fake.generate_request(LLMRequest('expert', HEURISTIC_FORMAT, 0.7))
    assert 'def improve_algorithm' in extract_code(outer)
    assert 'def update_edge_distance' in extract_code(inner)


def test_idea_and_code_lists_preserve_alignment():
    values = ['# {first}\n```python\nx = 1\n```', '# {second}\n```python\nx = 2\n```']
    assert extract_idea(values) == ['first', 'second']
    assert extract_code(values) == ['x = 1', 'x = 2']


@pytest.mark.parametrize('value', ['', '```python\n```', '```python\nx=1\n```\n```python\nx=2\n```'])
def test_reject_empty_or_ambiguous_code(value):
    with pytest.raises(GenerationError):
        extract_code(value)


@pytest.mark.parametrize('language', ['python', 'json'])
@pytest.mark.parametrize('other_language', ['', 'javascript', 'text'])
@pytest.mark.parametrize('other_first', [False, True])
def test_reject_additional_fence_regardless_of_label(language, other_language, other_first):
    supported = f'```{language}\nx=1\n```'
    other = f'```{other_language}\nignored content\n```'
    response = '\n'.join([other, supported] if other_first else [supported, other])
    with pytest.raises(GenerationError, match='missing_or_ambiguous_code'):
        extract_code(response)


@pytest.mark.parametrize('language', ['python', 'json'])
def test_extract_single_supported_fence(language):
    assert extract_code(f'Explanation\n```{language}\nx=1\n```\nAfterword') == 'x=1'


@pytest.mark.parametrize('response', ['x=1', '```\nx=1\n```', '```javascript\nx=1\n```'])
def test_reject_unfenced_or_unsupported_code(response):
    with pytest.raises(GenerationError, match='missing_or_ambiguous_code'):
        extract_code(response)


def test_three_distinct_optimizer_sources():
    fake = FakeLLM(42)
    sources = [extract_code(fake.generate_request(LLMRequest('expert', OPTIMIZER_FORMAT)))
               for _ in range(3)]
    assert len(set(sources)) == 3


@pytest.mark.parametrize('kwargs', [{'temperature': float('nan')}, {'temperature': 3},
                                    {'timeout_seconds': 0}, {'timeout_seconds': True},
                                    {'temperature': 10**400},
                                    {'timeout_seconds': 10**400}])
def test_request_numbers_validated(kwargs):
    with pytest.raises(ValueError):
        LLMRequest('expert', 'message', **kwargs)


def test_batch_order_and_preflight():
    from moh.execution.process import Deadline
    from moh.llm.program_adapter import ProgramLLM
    fake = FakeLLM(1, {'directions': ['one', 'two']})
    facade = ProgramLLM(fake, batch_size=2)
    with pytest.raises(ValueError):
        facade.prompt_batch('expert', ['KIND: directions', ''], deadline=Deadline.after(2))
    assert not fake.cursors
    assert facade.prompt_batch('expert', ['KIND: directions'] * 2,
                               deadline=Deadline.after(2)) == ['one', 'two']


def test_recording_logical_request_once():
    from moh.llm.recording import RecordingLLM
    events = []
    recorder = RecordingLLM(FakeLLM(1), lambda *value: events.append(value), {})
    recorder.generate_request(LLMRequest('role', HEURISTIC_FORMAT, 0.7))
    assert recorder.calls == 1
    assert events[0][1]['expertise'] == 'role'
    assert events[0][1]['temperature'] == 0.7


def test_provider_request_mapping_and_bounded_retry(monkeypatch):
    from types import SimpleNamespace

    import httpx2 as httpx
    import openai

    from moh.llm.openai_client import OpenAILLMClient

    monkeypatch.setenv('OPENAI_API_KEY', 'offline-key')
    values = [SimpleNamespace(output_text='ok', status='completed', model='stub')]
    requests = []

    class Transport:
        responses = None

        def __init__(self):
            self.responses = self

        def create(self, **kwargs):
            requests.append(kwargs)
            value = values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

    client = OpenAILLMClient('stub', 5, lambda _: None, transport=Transport())
    assert client.generate_request(LLMRequest('role', 'message', 0.7, 0.1)) == 'ok'
    assert requests[0]['instructions'] == 'role'
    assert requests[0]['temperature'] == 0.7
    assert 0 < requests[0]['timeout'] <= 0.1
    values.append(openai.APIConnectionError(request=httpx.Request('POST', 'https://invalid')))
    with pytest.raises(GenerationError, match='timeout'):
        client.generate_request(LLMRequest('role', 'message', 0.7, 0.1))
    assert len(requests) == 2


def test_oversized_fake_response():
    from moh.llm.base import MAX_RESPONSE_BYTES
    fake = FakeLLM(1, {'directions': ['x' * (MAX_RESPONSE_BYTES + 1)]})
    with pytest.raises(GenerationError, match='response_limit'):
        fake.generate_request(LLMRequest('role', 'KIND: directions'))


def test_batch_stops_after_deadline():
    from moh.execution.process import CandidateFailure, Deadline
    from moh.llm.program_adapter import ProgramLLM
    from moh.llm.recording import RecordingLLM

    class Client(FakeLLM):
        def generate_request(self, request):
            assert request.timeout_seconds > 0
            result = super().generate_request(request)
            object.__setattr__(deadline, 'expires_at', 0)
            return result

    deadline = Deadline.after(2)
    recorder = RecordingLLM(Client(1), lambda *_: None, {})
    facade = ProgramLLM(recorder, batch_size=2)
    with pytest.raises(CandidateFailure, match='timeout'):
        facade.prompt_batch('expert', ['KIND: directions'] * 2, deadline=deadline)
    assert recorder.calls == 1
