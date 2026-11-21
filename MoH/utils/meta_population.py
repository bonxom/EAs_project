"""MetaPopulation and Meta-UIQ for R-MoH Outer-Loop."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from utils.diagnostics import TrajectoryDiagnostics

logger = logging.getLogger(__name__)


@dataclass
class OptimizerIndividual:
    opt_id: str
    code: str
    thought: str
    raw_utility: float
    normalized_utility: float = 0.5
    diagnostics: TrajectoryDiagnostics | None = None
    selection_count: int = 0
    offspring_utilities: list[float] = field(default_factory=list)

    def compute_meta_uiq(
        self,
        total_outer_steps: int,
        alpha: float = 0.6,
        c: float = 0.15,
    ) -> float:
        """Compute Meta-UIQ combining exploitation and UCB exploration."""
        if not self.offspring_utilities:
            valid_rate = self.diagnostics.valid_rate if self.diagnostics else 0.5
            expected_offspring_q = 0.8 * self.normalized_utility + 0.2 * valid_rate
        else:
            expected_offspring_q = sum(self.offspring_utilities) / len(self.offspring_utilities)

        exploitation_term = alpha * self.normalized_utility + (1.0 - alpha) * expected_offspring_q
        exploration_term = c * math.sqrt(
            math.log(max(total_outer_steps, 1) + 1) / (self.selection_count + 1)
        )
        return float(exploitation_term + exploration_term)


class MetaPopulation:
    """Outer-loop population manager using Meta-UIQ and Elitist culling."""

    def __init__(self, size: int = 10, alpha: float = 0.6, c: float = 0.15) -> None:
        self.size = size
        self.alpha = alpha
        self.c = c
        self.individuals: list[OptimizerIndividual] = []

    def __len__(self) -> int:
        return len(self.individuals)

    def get_all(self) -> list[OptimizerIndividual]:
        return list(self.individuals)

    def get_by_id(self, opt_id: str) -> OptimizerIndividual | None:
        for ind in self.individuals:
            if ind.opt_id == opt_id:
                return ind
        return None

    def get_best(self) -> OptimizerIndividual | None:
        if not self.individuals:
            return None
        return min(self.individuals, key=lambda x: x.raw_utility)

    def normalize_utilities(self) -> None:
        """Min-Max normalization: lower raw_utility (gap) maps to higher normalized_utility."""
        if not self.individuals:
            return
        raw_vals = [ind.raw_utility for ind in self.individuals]
        min_raw = min(raw_vals)
        max_raw = max(raw_vals)

        if abs(max_raw - min_raw) < 1e-9:
            for ind in self.individuals:
                ind.normalized_utility = 1.0
        else:
            for ind in self.individuals:
                # Lower raw utility is better => (max - raw) / (max - min) in [0, 1]
                ind.normalized_utility = (max_raw - ind.raw_utility) / (max_raw - min_raw)

    def normalize_single(self, raw_utility: float) -> float:
        if not self.individuals:
            return 0.5
        raw_vals = [ind.raw_utility for ind in self.individuals]
        min_raw = min(raw_vals)
        max_raw = max(raw_vals)
        if abs(max_raw - min_raw) < 1e-9:
            return 1.0 if raw_utility <= min_raw else 0.0
        val = (max_raw - raw_utility) / (max_raw - min_raw)
        return float(np.clip(val, 0.0, 1.0))

    def add_individual(
        self,
        opt_id: str,
        code: str,
        thought: str,
        raw_utility: float,
        diagnostics: TrajectoryDiagnostics | None = None,
        total_outer_steps: int = 1,
    ) -> OptimizerIndividual:
        ind = OptimizerIndividual(
            opt_id=opt_id,
            code=code,
            thought=thought,
            raw_utility=raw_utility,
            diagnostics=diagnostics,
        )
        self.individuals.append(ind)
        self.normalize_utilities()
        if len(self.individuals) > self.size:
            self.cull(total_outer_steps=total_outer_steps)
        return ind

    def select_parent(self, total_outer_steps: int) -> OptimizerIndividual:
        """Select parent probabilistically based on Meta-UIQ."""
        if not self.individuals:
            raise ValueError("MetaPopulation is empty")
        if len(self.individuals) == 1:
            self.individuals[0].selection_count += 1
            return self.individuals[0]

        uiq_scores = [
            ind.compute_meta_uiq(total_outer_steps, self.alpha, self.c)
            for ind in self.individuals
        ]
        # Softmax selection
        exp_scores = np.exp(np.array(uiq_scores) - np.max(uiq_scores))
        probs = exp_scores / np.sum(exp_scores)
        chosen_idx = int(np.random.choice(len(self.individuals), p=probs))
        parent = self.individuals[chosen_idx]
        parent.selection_count += 1
        return parent

    def register_offspring(self, parent_id: str, child_raw_utility: float) -> None:
        parent = self.get_by_id(parent_id)
        if parent:
            norm_val = self.normalize_single(child_raw_utility)
            parent.offspring_utilities.append(norm_val)

    def cull(self, total_outer_steps: int = 1) -> None:
        """Cull individuals exceeding max size, strictly preserving the Elitist individual."""
        if len(self.individuals) <= self.size:
            return

        # Find absolute elitist (minimum raw_utility)
        elitist = min(self.individuals, key=lambda x: x.raw_utility)

        # Candidates for removal: non-elitists
        non_elitists = [ind for ind in self.individuals if ind.opt_id != elitist.opt_id]
        if not non_elitists:
            return

        # Score non-elitists by Meta-UIQ
        scored = [
            (ind, ind.compute_meta_uiq(total_outer_steps, self.alpha, self.c))
            for ind in non_elitists
        ]
        # Worst is candidate with minimum Meta-UIQ
        worst_individual, _ = min(scored, key=lambda x: x[1])

        self.individuals = [ind for ind in self.individuals if ind.opt_id != worst_individual.opt_id]
        self.normalize_utilities()
