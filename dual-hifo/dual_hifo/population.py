"""Bounded population ordered by objective."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class Population:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._items: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[Dict[str, Any]]:
        return list(self._items)

    def add(self, idea: str, code: str, objective: float,
            result: Optional[Dict[str, Any]] = None) -> bool:
        if not code or "def " not in code or "return" not in code:
            return False
        if any(item["code"] == code for item in self._items):
            return False
        self._items.append({
            "idea": idea,
            "code": code,
            "objective": float(objective),
            "best_sol": code,
            "utility": float(objective),
            "result": result or {},
        })
        self._items.sort(key=lambda item: item["objective"])
        del self._items[self.max_size:]
        return True

    def get_best_solution(self) -> Optional[Dict[str, Any]]:
        return self._items[0] if self._items else None

    def get_best_objective(self) -> float:
        return self._items[0]["objective"] if self._items else float("inf")

    def get_random_solution(self, task: Optional[str] = None) -> Dict[str, Any]:
        if not self._items:
            raise ValueError("population is empty")
        weights = [1.0 / (rank + 1 + len(self._items) / 2.0)
                   for rank in range(len(self._items))]
        return random.choices(self._items, weights=weights, k=1)[0]

    def get_solution_by_index(self, task: Optional[str], index: int) -> Dict[str, Any]:
        return self._items[index]

    def get_subtask_size(self, task: Optional[str] = None) -> int:
        return len(self._items)
