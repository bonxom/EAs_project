
import ast
import io
import json
import logging
import os
import re
import textwrap
import tokenize
import traceback

logger = logging.getLogger(__name__)

def code_only(code_string):
    """Remove # comments and redundant blank lines."""
    tokens = tokenize.generate_tokens(io.StringIO(code_string).readline)
    code_without_comments = tokenize.untokenize(
        token._replace(string="") if token.type == tokenize.COMMENT else token
        for token in tokens
    )
    cleaned_code = re.sub(r"\n\s*\n", "\n", code_without_comments)
    return cleaned_code.strip()


def clean_code(algorithm_str):
    if isinstance(algorithm_str, str):
        return code_only(algorithm_str)
    elif isinstance(algorithm_str, list):
        return [code_only(s) for s in algorithm_str]


def extract_idea(algorithm_str):
    if isinstance(algorithm_str, str):
        return find_braces(algorithm_str)
    elif isinstance(algorithm_str, list):
        return [extract_idea(s) for s in algorithm_str]


def find_braces(response):
    """Extract content inside first {} pair."""
    match = re.search(r"\{(.*?)\}", response, re.DOTALL)
    if match:
        return match.group(1)
    if "import" in response:
        return response.split("import")[0]
    return None


def extract_code(algorithm_str):
    """Extract fenced code, or a complete raw Python/JSON response."""
    if isinstance(algorithm_str, str):
        code = find_largest_code_block_line_by_line(algorithm_str)
        if code is None:
            logger.warning("No code extracted from LLM response (%d characters)", len(algorithm_str))
        return code
    elif isinstance(algorithm_str, list):
        return [extract_code(s) for s in algorithm_str]


def find_largest_code_block_line_by_line(text):
    """Accept unlabeled/indented fences without treating prose as Python."""
    blocks = []
    fence = None
    current = []
    for line in text.splitlines():
        marker = re.fullmatch(r"\s*(`{3,}|~{3,})([^`~]*)", line)
        if fence is None and marker:
            fence = marker.group(1)
            current = []
        elif fence is not None and line.strip() == fence:
            block = textwrap.dedent("\n".join(current)).strip()
            if block:
                blocks.append(block)
            fence = None
        elif fence is not None:
            current.append(line)
    if blocks:
        return max(blocks, key=len)
    if fence is not None:
        return None
    raw = textwrap.dedent(text).strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        if isinstance(data, (dict, list)):
            return raw
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return None
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body):
        return raw
    return None


def find_txt_block(string):
    """Extract content of ```txt code block."""
    inside_txt_block = False
    current_block = ""
    lines = string.split("\n")

    for line in lines:
        if line == "```txt":
            inside_txt_block = True
            current_block = line + "\n"
        elif line == "```" and inside_txt_block:
            current_block += line + "\n"
            inside_txt_block = False
        elif inside_txt_block:
            current_block += line + "\n"
    return current_block


def match_number(string):
    """Extract first integer from string, default 1."""
    match = re.search(r"\d+", string)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return 1
    return 1


def read_file_as_str(path):
    with open(path, "r") as f:
        return f.read()


def write_str_to_file(s, path, mode="w"):
    if isinstance(s, list):
        s = "\n\n".join(s)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode) as f:
            f.write(s)
    except Exception as e:
        print("Failed to write to file", path, "with exception", e)
        print("Traceback:", traceback.format_exc())
        s = str(s)
        with open(path, mode) as f:
            f.write(s)
