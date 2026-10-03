"""Pure minimizing population updates and explicit staging for program search."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

from moh.core.programs import GapEvaluation, ScoredProgram


def program_rank(item: ScoredProgram) -> tuple[float, str]:
    return (item.utility, item.id)


def selection_weights(count: int, capacity: int) -> list[float]:
    return [1.0 / (rank + capacity / 2.0) for rank in range(1, count + 1)]


def _validated_members(members: tuple[ScoredProgram, ...]) -> tuple[ScoredProgram, ...]:
    unique: dict[bytes, ScoredProgram] = {}
    identities: dict[str, bytes] = {}
    for item in members:
        if not isinstance(item, ScoredProgram) or item.evaluation.status != "success":
            raise ValueError(
                "population members must be successfully evaluated programs"
            )
        if (
            isinstance(item.evaluation, GapEvaluation)
            and item.evaluation.split != "validation"
        ):
            raise ValueError("held-out scores cannot enter search populations")
        source = item.source_code.encode("utf-8")
        if item.id in identities and identities[item.id] != source:
            raise ValueError("population ID refers to conflicting sources")
        identities[item.id] = source
        unique.setdefault(source, item)
    return tuple(sorted(unique.values(), key=program_rank))


@dataclass(frozen=True)
class ProgramPopulation:
    capacity: int
    members: tuple[ScoredProgram, ...] = ()

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity <= 0:
            raise ValueError("population capacity must be a positive integer")
        object.__setattr__(
            self, "members", _validated_members(tuple(self.members))[: self.capacity]
        )

    def add(self, item: ScoredProgram) -> ProgramPopulation:
        return ProgramPopulation(self.capacity, (*self.members, item))

    def best(self) -> ScoredProgram | None:
        return self.members[0] if self.members else None

    def snapshot(self, task: str) -> dict[str, list[dict]]:
        """Return detached JSON-compatible entries for the optimizer worker."""
        return {
            task: [
                {
                    "id": item.id,
                    "best_sol": item.source_code,
                    "idea": item.idea,
                    "utility": item.utility,
                }
                for item in self.members
            ]
        }

    def select(self, rng: random.Random) -> ScoredProgram:
        if not self.members:
            raise ValueError("cannot select from an empty population")
        return rng.choices(
            self.members,
            weights=selection_weights(len(self.members), self.capacity),
            k=1,
        )[0]


class PopulationTransaction:
    """Stage immutable populations; discarding this object rolls back changes."""

    def __init__(self, populations: Mapping[str, ProgramPopulation]):
        self.staged = dict(populations)

    def add(self, task: str, item: ScoredProgram) -> None:
        self.staged[task] = self.staged[task].add(item)

    def commit(self) -> dict[str, ProgramPopulation]:
        return dict(self.staged)
