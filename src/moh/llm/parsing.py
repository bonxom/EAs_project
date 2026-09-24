import re


def strip_code_fence(text):
    match = re.fullmatch(r"\s*```(?:python)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
    return match.group(1) if match else text
