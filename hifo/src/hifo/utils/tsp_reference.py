"""Reference cost for TSP-construct instances, used to report % gap.

The evaluation instances are generated with a fixed seed, so the reference is
reproducible. It is a nearest-neighbour construction, not a provably optimal
tour, so the reported gap is a *gap to the NN baseline*: negative values mean
the evolved heuristic beats the baseline.
"""

import numpy as np

_baseline_cache = None


def nearest_neighbor_cost(instance, problem_size=None):
    """Average nearest-neighbour tour cost over one or more instances."""
    instances = instance if isinstance(instance, (list, tuple)) else [instance]
    costs = []
    for single in instances:
        single = np.asarray(single)
        n = problem_size or len(single)
        unvisited = set(range(n))
        current = 0
        unvisited.discard(current)
        total = 0.0
        while unvisited:
            candidate = min(unvisited, key=lambda j: np.linalg.norm(single[current] - single[j]))
            total += np.linalg.norm(single[current] - single[candidate])
            current = candidate
            unvisited.discard(current)
        total += np.linalg.norm(single[current] - single[0])
        costs.append(total)
    return float(np.mean(costs))


def get_baseline(instance_data, problem_size=None):
    """Cached baseline cost for the evaluation instance set."""
    global _baseline_cache
    if _baseline_cache is None:
        instances = [inst for inst, _ in instance_data]
        _baseline_cache = nearest_neighbor_cost(instances, problem_size)
    return _baseline_cache


def gap_percent(cost, baseline):
    """Percentage gap of ``cost`` against ``baseline``; None when unknown."""
    if cost is None or baseline in (None, 0):
        return None
    return (cost - baseline) / baseline * 100.0
