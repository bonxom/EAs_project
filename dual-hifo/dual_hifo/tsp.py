"""Fixed TSP benchmark loader and constructive evaluator."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import exec_code, time_limit

FUNC_NAME = "select_next_node"
FUNC_SIGNATURE = (
    "def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):"
)
FUNCTION_FORMAT = f"""```python
{FUNC_SIGNATURE}
    # Return the integer id of the next node to visit.
    ...
    return next_node
```"""

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "tsp"


@dataclass(frozen=True)
class TSPInstance:
    index: int
    coords: np.ndarray
    dist: np.ndarray
    neighbor_order: np.ndarray
    optimal_cost: float


def tour_cost(coords: np.ndarray, route: Sequence[int]) -> float:
    route = list(route)
    cost = 0.0
    for a, b in zip(route[:-1], route[1:]):
        cost += float(np.linalg.norm(coords[a] - coords[b]))
    cost += float(np.linalg.norm(coords[route[-1]] - coords[route[0]]))
    return cost


def load_instances(*, size: int, n_instances: int,
                   dataset_dir: Path = DATASET_DIR) -> Tuple[List[TSPInstance], Dict[str, Any]]:
    path = dataset_dir / f"TSP{size}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    coords_all = data["coordinate"]
    dist_all = data["distance_matrix"]
    costs = np.asarray(data["cost"], dtype=float)
    n_use = min(int(n_instances), len(costs))
    if n_use <= 0:
        raise ValueError("n_instances must be positive")

    instances: List[TSPInstance] = []
    for idx in range(n_use):
        coords = np.asarray(coords_all[idx], dtype=float)
        dist = np.asarray(dist_all[idx], dtype=float)
        if coords.shape != (size, 2) or dist.shape != (size, size):
            raise ValueError(f"TSP{size} instance {idx} has an unexpected shape")
        instances.append(TSPInstance(
            index=idx,
            coords=coords,
            dist=dist,
            neighbor_order=np.argsort(dist, axis=1),
            optimal_cost=float(costs[idx]),
        ))

    metadata = {
        "file": f"datasets/tsp/TSP{size}.pkl",
        "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
        "n_available": int(len(costs)),
        "n_instances": n_use,
        "selection": "first",
    }
    return instances, metadata


def _greedy_route(instance: TSPInstance, select_next_node: Callable,
                  neighbor_size: Optional[int]) -> Optional[List[int]]:
    n = len(instance.dist)
    route = [0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    limit = n if neighbor_size is None else max(1, min(int(neighbor_size), n))

    for _ in range(1, n - 1):
        current = route[-1]
        candidates = instance.neighbor_order[current][~visited[instance.neighbor_order[current]]]
        if candidates.size == 0:
            return None
        candidates = candidates[:min(limit, candidates.size)]
        try:
            nxt = int(select_next_node(current, 0, candidates.copy(), instance.dist))
        except Exception:
            return None
        if nxt < 0 or nxt >= n or visited[nxt]:
            return None
        route.append(nxt)
        visited[nxt] = True

    remaining = np.flatnonzero(~visited)
    if remaining.size != 1:
        return None
    route.append(int(remaining[0]))
    if len(set(route)) != n:
        return None
    return route


def evaluate_heuristic(code: str, instances: List[TSPInstance], *,
                       neighbor_size: Optional[int] = None,
                       timeout: int = 45) -> Dict[str, Any]:
    invalid = {
        "objective": 1e6,
        "valid": False,
        "mean_cost": None,
        "mean_optimal_cost": None,
        "per_instance_cost": [],
        "per_instance_gap": [],
        "error": None,
    }
    if not code or "def " not in code:
        invalid["error"] = "empty or invalid code"
        return invalid

    try:
        with time_limit(timeout):
            env = exec_code(code)
            select_next_node = env.get(FUNC_NAME)
            if not callable(select_next_node):
                invalid["error"] = f"{FUNC_NAME} not found"
                return invalid

            costs, gaps, opt_costs = [], [], []
            for inst in instances:
                route = _greedy_route(inst, select_next_node, neighbor_size)
                if route is None:
                    invalid["error"] = "invalid node selection"
                    return invalid
                cost = tour_cost(inst.coords, route)
                costs.append(cost)
                gaps.append(max(0.0, (cost / inst.optimal_cost - 1.0) * 100.0))
                opt_costs.append(inst.optimal_cost)
    except Exception as exc:
        invalid["error"] = f"{type(exc).__name__}: {exc}"
        return invalid

    return {
        "objective": float(np.mean(gaps)),
        "valid": True,
        "mean_cost": float(np.mean(costs)),
        "mean_optimal_cost": float(np.mean(opt_costs)),
        "per_instance_cost": [float(c) for c in costs],
        "per_instance_gap": [float(g) for g in gaps],
        "error": None,
    }
