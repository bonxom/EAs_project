"""Bounded quality elites plus deterministic behavioral diversity survivors."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

from utils.behavior import (
    BehaviorDescriptor,
    behavior_distance,
    code_digest,
    descriptor_from_dict,
    descriptor_to_dict,
)
from utils.population import Pop
from utils.utils import clean_code

POLICY_VERSION = 'behavior-maxmin-log1p-v1'


def validate_fraction(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError('diversity_fraction must be finite and in [0, 1)')
    return float(value)


class BehaviorPopulation(Pop):
    def __init__(self, identities, size, diversity_fraction=0.3):
        if type(size) is not int or size <= 0:
            raise ValueError('Population size must be a positive integer')
        self.diversity_fraction = validate_fraction(diversity_fraction)
        self.identities = dict(identities)
        if not identities or any(identity.mode != 'val' for identity in identities.values()):
            raise ValueError('Population requires validation identities')
        self.elite_slots = size - math.floor(size * self.diversity_fraction)
        super().__init__(list(identities), size, allow_equal_utility=True)

    def policy_metadata(self):
        return {'version': POLICY_VERSION, 'diversity_fraction': self.diversity_fraction}

    def save_solution(self, task_name, idea, solution, utility):
        raise ValueError('Unmeasured insertion is forbidden; use admit with a measured behavior')

    def _entry(self, task, idea, code, behavior):
        if not isinstance(behavior, BehaviorDescriptor) or behavior.identity != self.identities[task]:
            raise ValueError('Behavior identity mismatch')
        code = clean_code(code)
        compile(code, '<population-candidate>', 'exec')
        if 'def ' not in code or 'return ' not in code:
            raise ValueError('Invalid heuristic source')
        return {'idea': idea, 'best_sol': code, 'utility': behavior.mean_gap_pct,
                'code_sha256': code_digest(code), 'behavior': descriptor_to_dict(behavior),
                'population_policy': self.policy_metadata()}

    @staticmethod
    def _quality(entry):
        return entry['utility'], entry['code_sha256']

    def _select(self, candidates):
        ordered = sorted(candidates, key=self._quality)
        selected, remaining = ordered[:self.elite_slots], ordered[self.elite_slots:]
        descriptors = {e['code_sha256']: descriptor_from_dict(e['behavior']) for e in ordered}
        while remaining and len(selected) < self.size:
            winner = min(remaining, key=lambda e: (
                -min(behavior_distance(descriptors[e['code_sha256']], descriptors[s['code_sha256']])
                     for s in selected), *self._quality(e)))
            selected.append(winner)
            remaining.remove(winner)
        return sorted(selected, key=self._quality)

    def admit(self, task, idea, code, behavior):
        entry = self._entry(task, idea, code, behavior)
        current = self.get_population(task)
        if any(e['code_sha256'] == entry['code_sha256'] for e in current):
            return False
        survivors = self._select([*current, entry])
        self._replace_population(task, survivors)
        return survivors != current

    def diagnostics(self, task):
        entries = self.get_population(task)
        descriptors = [descriptor_from_dict(e['behavior']) for e in entries]
        distances = [behavior_distance(a, b) for a, b in combinations(descriptors, 2)]
        return {'size': len(entries), 'best_mean_gap': entries[0]['utility'] if entries else None,
                'mean_pairwise_distance': math.fsum(distances) / len(distances) if distances else 0.0}

    def _validated_entries(self, task, entries):
        try:
            if not isinstance(entries, list) or len(entries) > self.size:
                raise ValueError('Invalid snapshot capacity')
            validated = []
            for entry in entries:
                measured = self._entry(task, entry['idea'], entry['best_sol'],
                                       descriptor_from_dict(entry['behavior']))
                # JSON roundtrip changes tuples to lists; compare canonical JSON values.
                if json.loads(json.dumps(measured, allow_nan=False)) != entry:
                    raise ValueError('Snapshot metadata mismatch')
                validated.append(measured)
            if len({e['code_sha256'] for e in validated}) != len(validated):
                raise ValueError('Duplicate snapshot candidate')
            return sorted(validated, key=self._quality)
        except (KeyError, TypeError, SyntaxError) as error:
            raise ValueError('Invalid population snapshot') from error

    def load_subtask_from_file(self, task_name, filename):
        entries = self._validated_entries(task_name, json.loads(Path(filename).read_text()))
        self._replace_population(task_name, entries)

    def load_all_data_from_file(self, filename):
        payload = json.loads(Path(filename).read_text())
        if not isinstance(payload, dict) or set(payload) != set(self.identities):
            raise ValueError('Snapshot task mismatch')
        validated = {task: self._validated_entries(task, entries) for task, entries in payload.items()}
        for task, entries in validated.items():
            self._replace_population(task, entries)

    def save_subtask_to_file(self, task_name, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.get_population(task_name), indent=2, allow_nan=False))

    def save_all_data_to_file(self, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({task: self.get_population(task) for task in self.identities},
                                   indent=2, allow_nan=False))
