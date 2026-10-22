"""Code extraction, sandboxed execution, and timeouts."""

from __future__ import annotations

import contextlib
import json
import re
import signal
import threading
from typing import Any, Dict, List, Optional


def extract_code(text: Optional[str]) -> Optional[str]:
    """Return the largest fenced code block, or the first code-like tail."""
    if not text:
        return None

    blocks: List[str] = []
    current: Optional[List[str]] = None
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current).strip())
                current = None
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        blocks.append("\n".join(current).strip())

    blocks = [b for b in blocks if b]
    if blocks:
        return max(blocks, key=len)
    if "def " in text:
        idx = min(i for i in (text.find("import "), text.find("def ")) if i >= 0)
        return text[idx:].strip()
    return None


def extract_idea(text: Optional[str]) -> str:
    if not text:
        return ""
    match = re.search(r"\{(.*?)\}", text, re.DOTALL)
    return match.group(1).strip() if match else ""


class EvaluationTimeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise EvaluationTimeout(f"evaluation exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def execution_globals() -> Dict[str, Any]:
    import numpy as np
    import random

    return {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "random": random,
        "json": json,
        "extract_code": extract_code,
        "extract_idea": extract_idea,
    }


def exec_code(code: str) -> Dict[str, Any]:
    env = execution_globals()
    exec(compile(code, "<generated>", "exec"), env)
    return env
