import pytest
from utils.utils import clean_code, extract_code


@pytest.mark.parametrize(
    "response",
    [
        "def improve_algorithm():\n    return 1",
        "```\ndef improve_algorithm():\n    return 1\n```",
        "  ```python\n  def improve_algorithm():\n      return 1\n  ```",
        "Explanation\n```python\ndef improve_algorithm():\n    return 1\n```\nDone.",
    ],
)
def test_extracts_supported_python_formats(response):
    assert extract_code(response) == "def improve_algorithm():\n    return 1"


def test_preserves_json_direction_responses():
    response = '{"insights": ["a", "b"]}'
    assert extract_code(response) == response
    assert extract_code(f"```json\n{response}\n```") == response


def test_ignores_unclosed_blocks_and_prose(caplog):
    assert extract_code("Please try a different heuristic.") is None
    assert extract_code("```python\ndef broken():\n    return 1") is None
    assert "No code extracted" in caplog.text


def test_batch_preserves_alignment():
    assert extract_code(["No code", "def f():\n    return 1"]) == [
        None,
        "def f():\n    return 1",
    ]


def test_recovers_truncated_unfenced_tail():
    truncated = "# idea\ndef f():\n    x = 1\n    y = x +"
    assert extract_code(truncated) == "# idea\ndef f():\n    x = 1"


def test_recovers_when_truncation_cut_multi_line_statement():
    truncated = "# idea\ndef f():\n    x = 1\n    y = x +\n    z = 2"
    assert extract_code(truncated) == "# idea\ndef f():\n    x = 1"


def test_unrecoverable_garbage_stays_none(caplog):
    assert extract_code("def f(:\ndef g(:") is None
    assert "No code extracted" in caplog.text


def test_does_not_trim_valid_code_missing_def_class():
    assert extract_code("value = 1") is None


def test_recovers_orphan_trailing_fence_without_opener():
    response = "# idea\ndef f():\n    return 1\n```"
    assert extract_code(response) == "# idea\ndef f():\n    return 1"


def test_recovers_orphan_trailer_plus_truncated_statement():
    response = "# idea\ndef f():\n    x = 1\n    y = x +\n```"
    assert extract_code(response) == "# idea\ndef f():\n    x = 1"


def test_recovers_unfenced_leading_brace_idea_header():
    response = "{An idea here}\n\ndef f():\n    return 1"
    assert extract_code(response) == "# {An idea here}\n\ndef f():\n    return 1"


def test_recovers_multiline_leading_brace_idea_header():
    response = "{An idea\nspanning lines}\n\ndef f():\n    return 1"
    assert extract_code(response) == "# {An idea\n# spanning lines}\n\ndef f():\n    return 1"


def test_leading_brace_with_no_code_stays_none():
    assert extract_code("{just an idea, no code}") is None


def test_cleaning_preserves_hashes_inside_generated_prompt_strings():
    source = '# idea\ndef f():\n    return "# candidate"  # comment\n'
    cleaned = clean_code(source)
    assert '"# candidate"' in cleaned
    assert "# idea" not in cleaned
    assert "# comment" not in cleaned
    compile(cleaned, "<cleaned-code>", "exec")
