import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class Population[T]:
    capacity: int
    members: tuple[T, ...]

    def __init__(self, capacity: int, members: Sequence[T]):
        if type(capacity) is not int or capacity <= 0 or len(members) > capacity:
            raise ValueError("invalid population capacity")
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "members", tuple(members))


def rank_key(status, utility, candidate_id):
    if status == "success":
        if type(utility) not in (int, float) or not math.isfinite(utility):
            raise ValueError("successful rank needs finite utility")
        return (0, -utility, candidate_id)
    if status != "failed" or utility is not None:
        raise ValueError("invalid failed rank")
    return (1, 0.0, candidate_id)
