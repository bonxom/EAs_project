import pytest

from moh.core.models import OptimizerProgram
from moh.llm.base import GenerationError
from moh.optimizers.programs import parse_optimizer_program


def test_valid_program():
    code = "def improve_algorithm(api):\n    return None"
    program = parse_optimizer_program(code, "p1", "test idea")
    assert isinstance(program, OptimizerProgram)
    assert program.id == "p1"
    assert program.source_code == code
    assert program.idea == "test idea"


def test_valid_program_with_helper():
    code = "def helper(x):\n    return x\n\ndef improve_algorithm(api):\n    return helper(api)"
    program = parse_optimizer_program(code, "p2")
    assert program.id == "p2"
    assert "def helper(x):" in program.source_code


def test_valid_code_fence():
    fenced = "```python\ndef improve_algorithm(api):\n    return None\n```"
    program = parse_optimizer_program(fenced, "p3")
    assert program.source_code == "def improve_algorithm(api):\n    return None"


@pytest.mark.parametrize(
    "invalid_code",
    [
        "",
        "   ",
        "def foo(:",
        "def other_function(api):\n    pass",
        "def improve_algorithm(api):\n    pass\n\ndef improve_algorithm(api):\n    pass",
        "async def improve_algorithm(api):\n    pass",
        "def improve_algorithm():\n    pass",
        "def improve_algorithm(api, extra):\n    pass",
        "def improve_algorithm(*args):\n    pass",
        "def improve_algorithm(**kwargs):\n    pass",
        "def improve_algorithm(wrong_name):\n    pass",
        "def improve_algorithm(api=1):\n    pass",
        "Here is the program:\ndef improve_algorithm(api):\n    pass",
    ],
)
def test_invalid_programs_raise_generation_error(invalid_code):
    with pytest.raises(GenerationError) as exc_info:
        parse_optimizer_program(invalid_code, "fail_id")
    assert str(exc_info.value) == "invalid_optimizer_program"
