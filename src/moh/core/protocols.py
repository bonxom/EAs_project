from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Task(Protocol):
    id: str
    instances: tuple[np.ndarray, ...]
    instance_seeds: tuple[int, ...]

    def score_tour(self, instance_index: int, tour: Sequence[int]) -> float: ...
