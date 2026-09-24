import pytest
from test_selection import members

from moh.llm.base import GenerationError
from moh.llm.fake import FakeLLM
from moh.llm.parsing import strip_code_fence
from moh.llm.recording import RecordingLLM
from moh.prompts.heuristic_generation import heuristic_prompt


@pytest.mark.parametrize(
    "text,want",
    [
        ("```python\nx = 1\n```", "x = 1"),
        ("```\nx = 1\n```", "x = 1"),
        ("Prose\n```python\nx=1\n```", "Prose\n```python\nx=1\n```"),
    ],
)
def test_fences(text, want):
    assert strip_code_fence(text) == want


def test_records_and_prompts():
    events = []
    client = RecordingLLM(
        FakeLLM(42, {"reflection": ("idea",), "mutate": (GenerationError("bad"),)}),
        lambda e, p: events.append((e, p)),
        {"task_id": "t"},
    )
    assert client.generate("KIND: reflection") == "idea"
    with pytest.raises(GenerationError):
        client.generate("KIND: mutate")
    assert client.calls == 2
    assert [x[0] for x in events] == [
        "llm_requested",
        "llm_called",
        "llm_requested",
        "llm_failed",
    ]
    prompt = heuristic_prompt("crossover", members()[:2], "keep close")
    assert prompt.startswith("KIND: crossover\n")
    for value in [
        "keep close",
        '"id": "a"',
        '"id": "b"',
        '"utility": -1.0',
        "select_next_node",
        "source_code",
    ]:
        assert value in prompt


def test_invalid_unicode_source_is_generation_failure(base_spec):
    from moh.core.specs import OptimizerSpec
    from moh.optimizers.inner import generate_child

    with pytest.raises(GenerationError):
        generate_child(
            (), OptimizerSpec(**base_spec), FakeLLM(42, {"mutate": ("\ud800",)}), "h1"
        )
