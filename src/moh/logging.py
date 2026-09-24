"""Strict, append-only experiment events and retained candidate programs."""

import json
import re
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import BaseModel


def to_json(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return {k: to_json(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def encode_record(payload):
    return json.dumps(
        to_json(payload), allow_nan=False, sort_keys=True, ensure_ascii=True
    )


class RunRecorder:
    @classmethod
    def create(cls, root, resolved_config, *, redactions=()):
        config = json.loads(encode_record(resolved_config))
        path = Path(root) / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex
        )
        path.mkdir(parents=True, exist_ok=False)
        recorder = cls(path, redactions)
        try:
            (path / "heuristics").mkdir()
            (path / "optimizers").mkdir()
            (path / "config.yaml").write_text(
                recorder._redact(yaml.safe_dump(config)), encoding="utf-8"
            )
            try:
                revision = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                    )
                    .decode()
                    .strip()
                )
            except (OSError, subprocess.SubprocessError):
                revision = None
            recorder.metadata = {
                "schema_version": 1,
                "python": sys.version.split()[0],
                "revision": revision,
                "versions": {
                    name: version(name)
                    for name in ("mini-moh", "numpy", "pydantic", "pyyaml", "openai")
                },
            }
            recorder._write_run({"status": "running"})
        except BaseException:
            recorder.close()
            raise
        return recorder

    def __init__(self, path, redactions):
        self.path = path
        self.redactions = tuple(x for x in redactions if x)
        self.sequence = 0
        self.events = (path / "events.jsonl").open("x", encoding="utf-8")

    def _redact(self, text):
        for value in self.redactions:
            text = text.replace(value, "[REDACTED]")
        return text

    def _safe_value(self, value):
        value = to_json(value)
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {k: self._safe_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe_value(v) for v in value]
        return value

    def emit(self, event, payload):
        if {"schema_version", "sequence", "event"} & payload.keys():
            raise ValueError("reserved event envelope key")
        record = {
            "schema_version": 1,
            "sequence": self.sequence,
            "event": event,
            **payload,
        }
        self.events.write(encode_record(self._safe_value(record)) + "\n")
        self.events.flush()
        self.sequence += 1

    def _artifact(self, folder, identifier, suffix, content):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("unsafe artifact ID")
        path = self.path / folder / (identifier + suffix)
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise ValueError("artifact identity collision")
        else:
            path.write_text(content, encoding="utf-8")

    def save_heuristic(self, heuristic):
        self._artifact(
            "heuristics", heuristic.id, ".py", self._redact(heuristic.source_code)
        )

    def save_optimizer(self, candidate):
        self._artifact(
            "optimizers", candidate.id, ".json", encode_record(candidate.spec)
        )

    def _write_run(self, values):
        temp = self.path / "run.json.tmp"
        temp.write_text(
            encode_record(self._safe_value({**self.metadata, **values})),
            encoding="utf-8",
        )
        temp.replace(self.path / "run.json")

    def finish(self, result):
        self._write_run(to_json(result))

    def fail(self, error):
        self._write_run({"status": "error", "error": str(error)[:1024]})

    def close(self):
        self.events.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
