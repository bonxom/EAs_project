"""Stable, typed JSON labels hashed to a NumPy-compatible 32-bit seed."""

import hashlib
import json


def derive_seed(root: int, *labels: str | int) -> int:
    if type(root) is not int or root < 0:
        raise ValueError("root seed must be a nonnegative integer")
    if any(type(label) not in (str, int) for label in labels):
        raise ValueError("seed labels must be strings or integers")
    payload = json.dumps([root, *labels], ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")
