import json

import pytest

from moh.llm.base import GenerationError
from moh.llm.fake import FakeLLM
from moh.optimizers.meta import MetaOptimizer


@pytest.mark.parametrize(
    "text", ["{}", "NaN", '{"x":1,"x":2}', "```json\n{}\n```", "prose", '{"x":1e999}']
)
def test_bad_json(text):
    with pytest.raises(GenerationError):
        MetaOptimizer().propose((), FakeLLM(42, {"optimizer_spec": (text,)}), "o3")


def test_schema(base_spec):
    for update in [
        {"reflection": False},
        {"population_size": True},
        {"population_size": 11},
        {"use_reflection": "false"},
    ]:
        with pytest.raises(GenerationError):
            MetaOptimizer().propose(
                (),
                FakeLLM(42, {"optimizer_spec": (json.dumps({**base_spec, **update}),)}),
                "o3",
            )
    candidate = MetaOptimizer().propose(
        (), FakeLLM(42, {"optimizer_spec": (json.dumps(base_spec),)}), "o3"
    )
    assert candidate.spec.model_dump() == base_spec
