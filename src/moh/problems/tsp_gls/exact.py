"""Exact Held–Karp optimum for reproducible smoke instances only."""

import numpy as np

from moh.problems.tsp_gls.tour import validate_distances


def held_karp(distances) -> tuple[float, tuple[int, ...]]:
    matrix = np.asarray(distances)
    if matrix.ndim != 2 or matrix.shape[0] > 12:
        raise ValueError("Held–Karp supports at most 12 cities")
    matrix = validate_distances(matrix)
    size = len(matrix)
    costs = {}
    predecessors = {}
    for city in range(1, size):
        costs[1 << (city - 1), city] = float(matrix[0, city])
        predecessors[1 << (city - 1), city] = 0
    full = (1 << (size - 1)) - 1
    for mask in range(1, full + 1):
        for city in range(1, size):
            bit = 1 << (city - 1)
            previous_mask = mask ^ bit
            if not mask & bit or not previous_mask:
                continue
            cost, predecessor = min(
                (
                    costs[previous_mask, previous] + float(matrix[previous, city]),
                    previous,
                )
                for previous in range(1, size)
                if previous_mask & (1 << (previous - 1))
            )
            costs[mask, city] = cost
            predecessors[mask, city] = predecessor
    cost, last = min(
        (costs[full, city] + float(matrix[city, 0]), city) for city in range(1, size)
    )
    reverse = []
    mask = full
    while last:
        reverse.append(last)
        previous = predecessors[mask, last]
        mask ^= 1 << (last - 1)
        last = previous
    return cost, (0, *reversed(reverse), 0)
