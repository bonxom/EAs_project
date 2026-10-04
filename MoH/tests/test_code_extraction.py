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


def test_cleaning_preserves_hashes_inside_generated_prompt_strings():
    source = '# idea\ndef f():\n    return "# candidate"  # comment\n'
    cleaned = clean_code(source)
    assert '"# candidate"' in cleaned
    assert "# idea" not in cleaned
    assert "# comment" not in cleaned
    compile(cleaned, "<cleaned-code>", "exec")
