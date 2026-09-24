import pytest

from moh.llm.base import GenerationError
from moh.llm.fake import FakeLLM


def test_independent_streams():
    a, b = FakeLLM(42), FakeLLM(42)
    first = a.generate("KIND: mutate\nfixture")
    assert first == b.generate("KIND: mutate\nfixture")
    a.generate("KIND: reflection\nfixture")
    assert a.generate("KIND: mutate\nfixture") == b.generate("KIND: mutate\nfixture")
    assert FakeLLM(42).generate("KIND: mutate\nfixture") == first
    for _ in range(20):
        assert a.generate("KIND: optimizer_spec\nfixture").startswith("{")


def test_script_exhaustion_and_errors():
    client = FakeLLM(1, {"mutate": ("ok", GenerationError("bad"))})
    assert client.generate("KIND: mutate") == "ok"
    for _ in range(2):
        with pytest.raises(GenerationError):
            client.generate("KIND: mutate")
