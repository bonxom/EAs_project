import pytest

from moh.core.models import OptimizerProgram
from moh.llm.base import GenerationError
from moh.llm.fake import FakeLLM
from moh.optimizers.meta import ProgramMetaOptimizer


def test_program_meta_optimizer_propose_with_fakellm():
    llm = FakeLLM(seed=42)
    optimizer = ProgramMetaOptimizer()
    prog1 = optimizer.propose([], llm, "prog_1")
    assert isinstance(prog1, OptimizerProgram)
    assert prog1.id == "prog_1"
    assert "def improve_algorithm(api):" in prog1.source_code

    prog2 = optimizer.propose([], llm, "prog_2")
    assert isinstance(prog2, OptimizerProgram)
    assert prog2.id == "prog_2"


def test_program_meta_optimizer_malformed_llm_response():
    llm = FakeLLM(
        seed=42,
        responses={"optimizer_program": ["invalid python source !!!"]},
    )
    optimizer = ProgramMetaOptimizer()
    with pytest.raises(GenerationError) as exc_info:
        optimizer.propose([], llm, "prog_fail")
    assert str(exc_info.value) == "invalid_optimizer_program"


def test_fakellm_optimizer_spec_remains_unchanged():
    llm = FakeLLM(seed=42)
    spec_response = llm.generate("KIND: optimizer_spec\nsome prompt")
    assert "parent_selection" in spec_response
    assert "generation_operator" in spec_response

    prog_response = llm.generate("KIND: optimizer_program\nsome prompt")
    assert "def improve_algorithm(api):" in prog_response
