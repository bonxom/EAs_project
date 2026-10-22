"""Insight pool and search-phase guidance."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import List, Sequence, Tuple

OUTER_TIPS = [
    "Use elite and diverse members of the inner population to condition the LLM prompt.",
    "Ask for several candidate heuristics and keep the one with the best utility.",
    "Validate generated code and fall back to a parent when the output is invalid.",
]

INNER_TIPS = [
    "Balance immediate edge cost against future connectivity.",
    "Exploit clusters and candidate sets to bias construction.",
    "Diversify when progress stagnates and intensify around good tours otherwise.",
]


class InsightPool:
    def __init__(self, max_size: int = 12, initial_tips: Sequence[str] = INNER_TIPS):
        self.max_size = max_size
        self._tips = deque()
        self._score = {}
        self._used = {}
        for tip in initial_tips:
            self.add(tip)

    def add(self, tip: str) -> bool:
        tip = (tip or "").strip()
        if len(tip) < 12:
            return False
        if any(self._similarity(tip, other) > 0.7 for other in self._tips):
            return False
        if len(self._tips) >= self.max_size:
            self._tips.popleft()
        self._tips.append(tip)
        self._score[tip] = 0.0
        self._used[tip] = 0
        return True

    def get(self, k: int = 3) -> List[str]:
        if not self._tips:
            return []
        ranked = sorted(
            self._tips,
            key=lambda tip: self._score[tip] - 0.1 * math.log(self._used[tip] + 1),
            reverse=True,
        )
        selected = ranked[:k]
        for tip in selected:
            self._used[tip] += 1
        return selected

    def update(self, tip: str, reward: float) -> None:
        if tip not in self._score:
            return
        reward = max(-1.0, min(1.0, reward))
        self._score[tip] = 0.7 * self._score[tip] + 0.3 * reward

    def summary(self) -> dict:
        return {
            "n_tips": len(self._tips),
            "tips": list(self._tips),
        }

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / max(len(wa), len(wb)) if wa and wb else 0.0


class Navigator:
    def __init__(self):
        self.stagnation = 0
        self.improvements = 0
        self.last_best = None
        self.directives = {
            "general": [
                "balance local optimality with global search",
                "improve robustness across instances",
                "reduce unnecessary computation",
            ],
            "exploit": [
                "refine the scoring function",
                "tune critical thresholds",
                "simplify the current best design",
            ],
            "explore": [
                "try a different tour-construction strategy",
                "introduce an adaptive or randomized mechanism",
                "combine ideas from different parents",
            ],
        }

    def get_guidance(self, best_history: Sequence[float],
                     diversity_history: Sequence[float]) -> Tuple[str, str]:
        if len(best_history) < 2:
            return "balanced", random.choice(self.directives["general"])

        improvement = (self.last_best - best_history[-1]) if self.last_best is not None else 0.0
        if improvement <= 1e-4:
            self.stagnation += 1
            self.improvements = 0
        else:
            self.improvements += 1
            self.stagnation = 0
        self.last_best = best_history[-1]

        low_diversity = bool(diversity_history and diversity_history[-1] < 0.3)
        if self.stagnation >= 3 or low_diversity:
            regime = "explore"
        elif self.improvements >= 2:
            regime = "exploit"
        else:
            regime = random.choice(["explore", "exploit", "balanced"])
        pool = self.directives["general"] if regime == "balanced" else self.directives[regime]
        return regime, random.choice(pool)
