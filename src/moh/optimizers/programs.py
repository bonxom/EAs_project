"""Static AST validation for generated optimizer programs."""

import ast

from moh.core.models import OptimizerProgram
from moh.llm.base import GenerationError
from moh.llm.parsing import strip_code_fence


def parse_optimizer_program(
    text: str, candidate_id: str, idea: str | None = None
) -> OptimizerProgram:
    if not isinstance(text, str):
        raise GenerationError("invalid_optimizer_program")
    code = strip_code_fence(text)
    try:
        code.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise GenerationError("invalid_optimizer_program") from exc

    try:
        tree = ast.parse(code, filename="<optimizer_program>")
    except (SyntaxError, ValueError) as exc:
        raise GenerationError("invalid_optimizer_program") from exc

    entrypoints = []
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "improve_algorithm"
        ):
            entrypoints.append(node)

    if len(entrypoints) != 1:
        raise GenerationError("invalid_optimizer_program")

    fn = entrypoints[0]
    if not isinstance(fn, ast.FunctionDef):
        raise GenerationError("invalid_optimizer_program")

    args = fn.args
    if (
        len(args.args) != 1
        or args.args[0].arg != "api"
        or args.vararg is not None
        or args.kwarg is not None
        or len(args.kwonlyargs) != 0
        or len(args.posonlyargs) != 0
        or len(args.defaults) != 0
    ):
        raise GenerationError("invalid_optimizer_program")

    return OptimizerProgram(id=candidate_id, source_code=code, idea=idea)
