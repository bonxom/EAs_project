"""Per-size run logs."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")[:60]


class RunLogger:
    def __init__(self, output_dir: Path, size: int):
        self.size = size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "llm_outputs").mkdir(exist_ok=True)
        (self.output_dir / "code").mkdir(exist_ok=True)

        self.llm_path = self.output_dir / "llm_calls.jsonl"
        self.tsp_path = self.output_dir / "tsp_results.jsonl"
        self.generation_path = self.output_dir / "generations.jsonl"
        self.text_path = self.output_dir / "run.log"
        for path in (self.llm_path, self.tsp_path, self.generation_path, self.text_path):
            path.touch(exist_ok=True)

        self.llm_count = 0
        self.tsp_count = 0
        self._lock = threading.Lock()

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        with self._lock:
            with open(path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def info(self, message: str) -> None:
        with self._lock:
            with open(self.text_path, "a", encoding="utf-8") as file:
                file.write(f"[{_now()}] {message}\n")

    def log_llm(self, *, loop: str, tag: str, model: str, temperature: float,
                system: str, user: str, response: str,
                error: Optional[str] = None) -> None:
        with self._lock:
            self.llm_count += 1
            seq = self.llm_count

        record = {
            "seq": seq,
            "timestamp": _now(),
            "size": self.size,
            "loop": loop,
            "tag": tag,
            "model": model,
            "temperature": temperature,
            "system": system,
            "user_prompt": user,
            "response": response,
            "error": error,
        }
        self._append_jsonl(self.llm_path, record)

        path = self.output_dir / "llm_outputs" / f"{seq:04d}_{_slug(loop)}_{_slug(tag)}.md"
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"# {loop}/{tag}\n\n")
            file.write(f"Model: `{model}`  \nTemperature: {temperature}  \nSize: {self.size}\n\n")
            if error:
                file.write(f"Error: `{error}`\n\n")
            file.write("## System prompt\n\n```text\n")
            file.write(system.strip() + "\n```\n\n## User prompt\n\n```text\n")
            file.write(user.strip() + "\n```\n\n## Response\n\n")
            file.write(response or "(empty)")
            file.write("\n")

        self.info(f"[{loop}/{tag}] LLM #{seq}")

    def log_tsp(self, *, source: str, objective: float, valid: bool,
                mean_cost: Optional[float], mean_optimal_cost: Optional[float],
                per_instance: List[float], code: str, idea: str = "",
                error: Optional[str] = None) -> None:
        with self._lock:
            self.tsp_count += 1
            seq = self.tsp_count

        self._append_jsonl(self.tsp_path, {
            "seq": seq,
            "timestamp": _now(),
            "size": self.size,
            "source": source,
            "objective": objective,
            "valid": valid,
            "mean_cost": mean_cost,
            "mean_optimal_cost": mean_optimal_cost,
            "per_instance_cost": per_instance,
            "idea": idea,
            "code_sha1": hashlib.sha1(code.encode("utf-8")).hexdigest()[:12],
            "error": error,
        })
        self.info(f"[tsp/{source}] gap={objective:.4f}% valid={valid}")

    def log_generation(self, *, loop: str, generation: int,
                       population: List[Dict[str, Any]]) -> None:
        self._append_jsonl(self.generation_path, {
            "timestamp": _now(),
            "size": self.size,
            "loop": loop,
            "generation": generation,
            "population": [
                {
                    "idea": item.get("idea", ""),
                    "objective": item.get("objective"),
                    "code_sha1": hashlib.sha1(item["code"].encode()).hexdigest()[:12],
                }
                for item in population
            ],
        })
        best = min((item["objective"] for item in population), default=float("inf"))
        self.info(f"[{loop}] generation {generation}: best={best:.4f}")

    def save_summary(self, summary: Dict[str, Any]) -> None:
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    def save_code(self, code: str, name: str) -> Path:
        path = self.output_dir / "code" / f"{_slug(name)}.py"
        path.write_text(code + "\n", encoding="utf-8")
        return path
