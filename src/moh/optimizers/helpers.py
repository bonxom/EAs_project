"""Parse generated responses without executing their contents."""
import re

from moh.llm.base import GenerationError


def _map(response, parser):
    if isinstance(response, list):
        return [parser(value) for value in response]
    return parser(response)


def extract_code(response):
    def parse(value):
        if not isinstance(value, str):
            raise GenerationError('invalid_response')
        # Count every fence delimiter before filtering supported languages.
        # Extra unlabeled/unsupported blocks make the response ambiguous too.
        if value.count('```') != 2:
            raise GenerationError('missing_or_ambiguous_code')
        blocks = re.findall(r'```(?:python|json)\s*\n(.*?)```', value, re.DOTALL)
        if len(blocks) != 1 or not blocks[0].strip():
            raise GenerationError('missing_or_ambiguous_code')
        return blocks[0].strip()
    return _map(response, parse)


def extract_idea(response):
    def parse(value):
        if not isinstance(value, str):
            raise GenerationError('invalid_response')
        ideas = re.findall(r'^\s*#\s*\{([^{}]+)\}', value, re.MULTILINE)
        if len(ideas) != 1 or not ideas[0].strip():
            raise GenerationError('missing_or_ambiguous_idea')
        return ideas[0].strip()
    return _map(response, parse)
