import re

_CODE_START = re.compile(r"^[ \t]*(?:import|from|def)\b", re.MULTILINE)
_FENCED_BLOCK = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_algorithm_and_code(response, func_outputs):
    """Extract the algorithm description and runnable code from an LLM response.

    Returns [code_all, algorithm], or None when either part cannot be found.
    """
    algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
    if len(algorithm) == 0:
        if 'python' in response:
            algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
        elif 'import' in response:
            algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
        else:
            algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

    if len(algorithm) == 0:
        return None

    code = _extract_code(response)
    if code is None:
        return None

    # Some models return only the body, without a return statement; in that
    # case append one so the generated function is complete.
    if not re.search(r"^\s*return\b", code, re.MULTILINE):
        indent = _last_line_indent(code)
        code = code + "\n" + indent + "return " + ", ".join(s for s in func_outputs)

    return [code, algorithm[0]]


def _last_line_indent(code):
    for line in reversed(code.splitlines()):
        if line.strip():
            return line[:len(line) - len(line.lstrip())]
    return "    "


def _extract_code(response):
    """Return the first syntactically valid Python code block in ``response``.

    Prefers fenced blocks so trailing prose cannot leak into the code, and
    keeps the model's own return expression instead of overwriting it.
    """
    candidates = _FENCED_BLOCK.findall(response)
    candidates.append(response)

    fallback = None
    for candidate in candidates:
        match = _CODE_START.search(candidate)
        if match is None:
            continue

        block = candidate[match.start():].rstrip()
        if not _is_valid_python(block):
            continue
        if re.search(r"^\s*return\b", block, re.MULTILINE):
            return block
        if fallback is None:
            fallback = block

    return fallback


def _is_valid_python(code):
    try:
        compile(code, "<llm_generated_code>", "exec")
    except SyntaxError:
        return False
    return True
