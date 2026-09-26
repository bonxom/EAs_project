from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from moh.llm.base import GenerationError
from moh.llm.openai_client import OpenAILLMClient, validate_environment


class Transport:
    def __init__(self, values):
        self.values = iter(values)
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


def response(text="hello"):
    return SimpleNamespace(
        output_text=text,
        model="test-model",
        status="completed",
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )


def transient():
    return openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid")
    )


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")


@pytest.mark.parametrize("failures", [0, 1, 2, 3])
def test_bounded_retries(failures):
    transport = Transport([transient() for _ in range(failures)] + [response()])
    metadata, delays = [], []
    client = OpenAILLMClient(
        "test-model", 2.0, metadata.append, transport=transport, sleep=delays.append
    )
    if failures == 3:
        with pytest.raises(GenerationError):
            client.generate("prompt")
    else:
        assert client.generate("prompt") == "hello"
        assert metadata[-1].input_tokens == 10
    assert len(transport.requests) == min(failures + 1, 3)
    assert len(metadata) == len(transport.requests)
    assert all(x["timeout"] == 2.0 for x in transport.requests)
    assert len(delays) == min(failures, 2)


def test_permanent_error():
    error = openai.AuthenticationError(
        "offline-test-key",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://example.invalid")
        ),
        body=None,
    )
    transport = Transport([error])
    metadata = []
    client = OpenAILLMClient("test-model", 2.0, metadata.append, transport=transport)
    with pytest.raises(GenerationError) as caught:
        client.generate("prompt")
    assert len(transport.requests) == 1
    assert "offline-test-key" not in str(caught.value) + str(metadata)


@pytest.mark.parametrize("value", [response(""), response(None), object()])
def test_bad_response(value):
    transport = Transport([value])
    client = OpenAILLMClient("test-model", 2.0, lambda _: None, transport=transport)
    with pytest.raises(GenerationError):
        client.generate("prompt")
    assert len(transport.requests) == 1


def test_observer_error_not_retried():
    transport = Transport([response()])

    def observer(_):
        raise OSError("disk failed")

    client = OpenAILLMClient("test-model", 2.0, observer, transport=transport)
    with pytest.raises(OSError):
        client.generate("prompt")
    assert len(transport.requests) == 1


def test_preconditions(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    for key in ["", "  "]:
        monkeypatch.setenv("OPENAI_API_KEY", key)
        with pytest.raises(ValueError):
            validate_environment()
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError):
        validate_environment()
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    for model, timeout in [("", 2.0), ("test", 0), ("test", float("inf"))]:
        with pytest.raises(ValueError):
            OpenAILLMClient(model, timeout, lambda _: None, transport=Transport([]))
