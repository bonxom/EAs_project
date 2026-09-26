"""Prompt builder for generating executable optimizer programs."""

import json


def optimizer_program_prompt(records=None, idea=None):
    payload = {
        "records": records or [],
        "idea": idea,
    }
    dumped = json.dumps(payload, sort_keys=True, allow_nan=False)
    return (
        "KIND: optimizer_program\n"
        "Generate a Python program that implements an optimization algorithm.\n"
        "Return only Python code.\n"
        "Define exactly:\n"
        "def improve_algorithm(api):\n"
        "    ...\n"
        "Do not return JSON.\n"
        "Do not return Markdown explanation.\n"
        "Do not include credentials.\n"
        "Do not perform file or network IO.\n"
        f"INPUT_DATA: {dumped}\n"
    )
