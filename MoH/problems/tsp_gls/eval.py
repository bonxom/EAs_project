"""Subprocess entrypoint for isolated TSP-GLS heuristic evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import func_timeout
import numpy as np

try:
    from .dataset_selection import load_eval_dataset
    from .gls import (
        check_valid_route,
        local_search,
        nearest_neighbor_2End,
        relocate_o2a_all,
        route2tour,
        tour_cost_2End,
        two_opt_o2a_all,
    )
except ImportError:  # Executed as a script by the evaluator parent.
    from dataset_selection import load_eval_dataset
    from gls import (
        check_valid_route,
        local_search,
        nearest_neighbor_2End,
        relocate_o2a_all,
        route2tour,
        tour_cost_2End,
        two_opt_o2a_all,
    )


ITERATION_LIMIT = 1000
PERTURBATION_MOVES = 1
TIME_LIMIT = 20
_UPDATE_EDGE_DISTANCE = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_heuristic(path: str) -> None:
    """Load generated code only inside an instance worker process."""
    global _UPDATE_EDGE_DISTANCE
    heuristic_path = Path(path)
    module_name = f"deployment_heuristic_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, heuristic_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load heuristic from {heuristic_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "update_edge_distance", None)
    if not callable(function):
        raise TypeError("Heuristic must define callable update_edge_distance")
    _UPDATE_EDGE_DISTANCE = function


def solve(params):
    idx, dis_matrix, coord, optimal_cost, opt_tour, problem_size = params
    del idx, coord, opt_tour
    try:
        if _UPDATE_EDGE_DISTANCE is None:
            raise RuntimeError("Heuristic worker was not initialized")
        init_tour = nearest_neighbor_2End(dis_matrix, 0).astype(int)
        init_cost = tour_cost_2End(dis_matrix, init_tour)
        nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:101].astype(int)

        edge_weight = dis_matrix
        cur_route, cur_cost = local_search(
            init_tour,
            init_cost,
            edge_weight,
            nearest_indices,
            first_improvement=False,
        )
        best_route, best_cost = cur_route, cur_cost
        length = len(dis_matrix[0])
        edge_penalty = np.zeros((length, length))
        started = time.time()

        for iteration in range(ITERATION_LIMIT):
            if time.time() >= TIME_LIMIT + started:
                break
            for _move in range(PERTURBATION_MOVES):
                cur_tour = route2tour(cur_route)
                edge_weight_guided = _UPDATE_EDGE_DISTANCE(
                    edge_weight, np.array(cur_tour), edge_penalty
                )
                edge_weight_guided = np.asmatrix(edge_weight_guided)
                edge_weight_gap = edge_weight_guided - edge_weight

                for _top_id in range(5):
                    max_index = np.argmin(-edge_weight_gap, axis=None)
                    row, column = np.unravel_index(max_index, edge_weight_gap.shape)
                    edge_penalty[row, column] += 1
                    edge_penalty[column, row] += 1
                    edge_weight_gap[row, column] = 0
                    edge_weight_gap[column, row] = 0
                    for node_id in (row, column):
                        delta, new_route = two_opt_o2a_all(
                            cur_route, edge_weight_guided, nearest_indices, node_id
                        )
                        if delta < 0:
                            cur_cost = tour_cost_2End(edge_weight, new_route)
                            cur_route = new_route
                        delta, new_route = relocate_o2a_all(
                            cur_route, edge_weight_guided, nearest_indices, node_id
                        )
                        if delta < 0:
                            cur_cost = tour_cost_2End(edge_weight, new_route)
                            cur_route = new_route

            cur_route, cur_cost = local_search(
                cur_route,
                cur_cost,
                edge_weight,
                nearest_indices,
                first_improvement=False,
            )
            cur_cost = tour_cost_2End(edge_weight, cur_route)
            if cur_cost < best_cost:
                best_route, best_cost = cur_route, cur_cost
            if iteration % 50 == 49:
                cur_route, cur_cost = best_route, best_cost

        int_route = [int(node) for node in route2tour(best_route)]
        return validated_result(best_cost, optimal_cost, int_route, problem_size)
    except Exception:  # noqa: BLE001 -- candidate failure becomes a penalty
        return None


def worker_with_timeout(params, timeout):
    try:
        return func_timeout.func_timeout(timeout, solve, args=(params,))
    except func_timeout.FunctionTimedOut:
        print(f"Task {params[0]} timed out.")
        return None


def validated_result(best_cost, optimal_cost, route, problem_size):
    if not check_valid_route(route, problem_size):
        return None
    gap = (best_cost / optimal_cost - 1) * 100
    return float(best_cost), float(max(gap, 0))


def summarize_results(results):
    instances = []
    failures = 0
    for result in results:
        if result is None:
            objective, gap = 1e6, 1e6
            status = "failed"
            failures += 1
        else:
            objective, gap = result
            if not np.isfinite(objective) or not np.isfinite(gap):
                objective, gap = 1e6, 1e6
                status = "failed"
                failures += 1
            else:
                objective, gap = float(objective), float(gap)
                status = "ok"
        instances.append(
            {"objective": objective, "gap_pct": gap, "status": status}
        )
    return {
        "mean_gap_pct": sum(item["gap_pct"] for item in instances) / len(instances),
        "failures": failures,
        "instances": instances,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_size", type=int)
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("mode", choices=("val", "test"))
    parser.add_argument("timeout", type=int, nargs="?", default=90)
    parser.add_argument("seed", type=int, nargs="?", default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--heuristic-path", type=Path)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    started = time.monotonic()
    heuristic_path = args.heuristic_path or Path(__file__).with_name("gpt.py")
    tsp_data, indices, dataset_path = load_eval_dataset(
        args.root_dir,
        args.problem_size,
        args.mode,
        args.seed,
        args.count,
    )
    count = len(indices)
    workers = min(max(args.workers, 1), count, os.cpu_count() or 1, 8)
    print("[*] Running ...")
    print(
        f"[*] mode={args.mode}, data={dataset_path}, seed={args.seed}, "
        f"n_instances={count}"
    )
    print(f"[*] instance_indices={indices}")

    params_list = [
        (
            data_index,
            tsp_data["distance_matrix"][data_index],
            tsp_data["coordinate"][data_index],
            tsp_data["cost"][data_index],
            tsp_data["optimal_tour"][data_index],
            args.problem_size,
        )
        for data_index in indices
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_heuristic,
        initargs=(str(heuristic_path),),
    ) as executor:
        futures = {
            executor.submit(worker_with_timeout, params, args.timeout): params
            for params in params_list
        }
        concurrent.futures.wait(futures)
        results = []
        for future, params in futures.items():
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 -- one candidate instance may fail
                print(f"[*] instance {params[0]} failed: {error}")
                result = None
            results.append(result)

    summary = summarize_results(results)
    for index, instance in zip(indices, summary["instances"], strict=True):
        print(
            f"[*] instance {index}: obj={instance['objective']:.4f}, "
            f"gap={instance['gap_pct']:.4f}"
        )
    elapsed = time.monotonic() - started
    payload = {
        "size": args.problem_size,
        "mode": args.mode,
        "mean_gap_pct": summary["mean_gap_pct"],
        "status": "ok",
        "failures": summary["failures"],
        "indices": indices,
        "code_sha256": _sha256_file(heuristic_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "elapsed_seconds": elapsed,
        "seed": args.seed,
        "count": count,
        "solver_settings": {
            "workers": workers,
            "instance_timeout": args.timeout,
            "time_limit": TIME_LIMIT,
            "iteration_limit": ITERATION_LIMIT,
            "perturbation_moves": PERTURBATION_MOVES,
        },
        "instances": summary["instances"],
    }
    if args.result_path:
        args.result_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"[*] ave_gap on size {args.problem_size}: {summary['mean_gap_pct']}")
    print(summary["mean_gap_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
