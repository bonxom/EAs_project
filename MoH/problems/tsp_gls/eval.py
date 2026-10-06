"""
Task-level heuristic evaluation for TSP-GLS.

← from version_iclr/tasks/gls/secret_utility.py
← restructured to ReEvo subprocess style (reevo/problems/tsp_gls/eval.py)

Changes vs original:
  - No exec(): imports update_edge_distance from gpt.py via normal import
  - No global state: no config, no population, no utility_call_counter
  - Subprocess interface: receives params via sys.argv, prints result to stdout
  - Keeps ProcessPoolExecutor parallelism for instance solving (original pattern)
"""

import concurrent.futures
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import func_timeout
import numpy as np
from dataset_selection import load_eval_dataset
from gls import *
from gpt import update_edge_distance


def solve(params):
    idx, dis_matrix, coord, optimal_cost, opt_tour, problem_size = params
    ite_max = 1000
    perturbation_moves = 1
    time_limit = 20
    try:
        init_tour = nearest_neighbor_2End(dis_matrix, 0).astype(int)
        init_cost = tour_cost_2End(dis_matrix, init_tour)
        nb = 100
        nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:nb+1].astype(int)

        edge_weight = dis_matrix
        cur_route, cur_cost = local_search(init_tour, init_cost, edge_weight, nearest_indices, first_improvement=False)
        best_route, best_cost = cur_route, cur_cost

        length = len(dis_matrix[0])
        n_pert = min(int(length / 10), 20)
        iter_i = 0
        edge_penalty = np.zeros((length, length))
        t = time.time()

        while iter_i < ite_max and time.time() < (time_limit + t):
            for move in range(perturbation_moves):
                cur_tour, best_tour = route2tour(cur_route), route2tour(best_route)

                # ← this calls the function imported from gpt.py
                edge_weight_guided = update_edge_distance(edge_weight, np.array(cur_tour), edge_penalty)
                edge_weight_guided = np.asmatrix(edge_weight_guided)
                edge_weight_gap = edge_weight_guided - edge_weight

                for topid in range(5):
                    max_indices = np.argmin(-edge_weight_gap, axis=None)
                    rows, columns = np.unravel_index(max_indices, edge_weight_gap.shape)

                    edge_penalty[rows, columns] += 1
                    edge_penalty[columns, rows] += 1

                    edge_weight_gap[rows, columns] = 0
                    edge_weight_gap[columns, rows] = 0

                    for id in [rows, columns]:
                        delta, new_route = two_opt_o2a_all(cur_route, edge_weight_guided, nearest_indices, id)
                        if delta < 0:
                            cur_cost = tour_cost_2End(edge_weight, new_route)
                            cur_route = new_route
                        delta, new_route = relocate_o2a_all(cur_route, edge_weight_guided, nearest_indices, id)
                        if delta < 0:
                            cur_cost = tour_cost_2End(edge_weight, new_route)
                            cur_route = new_route

            cur_route, cur_cost = local_search(cur_route, cur_cost, edge_weight, nearest_indices, first_improvement=False)
            cur_cost = tour_cost_2End(edge_weight, cur_route)

            if cur_cost < best_cost:
                best_route, best_cost = cur_route, cur_cost

            iter_i += 1

            if iter_i % 50 == 0:
                cur_route, cur_cost = best_route, best_cost

        int_route = [int(i) for i in route2tour(best_route)]
        gap = (best_cost / optimal_cost - 1) * 100
        if check_valid_route(int_route, problem_size) == False:
            gap = 1e6
        if gap < 0:
            gap = 0
        return best_cost, gap
    except Exception as e:
        dis = 1e6
        gap = (dis - optimal_cost) / optimal_cost
        return dis, gap


# ← from version_iclr/tasks/gls/secret_utility.py: worker_with_timeout() — preserved
def worker_with_timeout(params, timeout):
    try:
        return func_timeout.func_timeout(timeout, solve, args=(params,))
    except func_timeout.FunctionTimedOut:
        print(f"Task {params[0]} timed out.")
        return None



if __name__ == "__main__":
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mode = sys.argv[3]     
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 90
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    tsp_data, random_numbers, file_name = load_eval_dataset(root_dir, problem_size, mode, seed)
    n_tests = len(random_numbers)
    print(f"[*] mode={mode}, data={file_name}, seed={seed}, n_instances={n_tests}")
    print(f"[*] instance_indices={random_numbers}")

    cost_list = tsp_data['cost']
    dist_list = tsp_data['distance_matrix']
    coord_list = tsp_data['coordinate']
    opt_tour_list = tsp_data['optimal_tour']

    params_list = []
    for idx in range(n_tests):
        data_idx = random_numbers[idx]
        params = (data_idx, dist_list[data_idx], coord_list[data_idx],
                  cost_list[data_idx], opt_tour_list[data_idx], problem_size)
        params_list.append(params)

    # ← from version_iclr: ProcessPoolExecutor parallel solving — preserved
    results = [(1e6, 1e6)] * n_tests
    # Test evaluates 128 instances in batches rather than spawning 128 workers.
    with ProcessPoolExecutor(max_workers=min(n_tests, os.cpu_count() or 1, 8)) as executor:
        futures = {executor.submit(worker_with_timeout, params, timeout): params
                   for params in params_list}
        concurrent.futures.wait(futures.keys())
        results = []
        for future, params in futures.items():
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 -- penalize a failed instance and continue
                print(f"[*] instance {params[0]} failed: {error}")
                result = None
            results.append(result if result is not None else (1e6, 1e6))

    gap = [1e6] * n_tests
    obj = [1e6] * n_tests
    for i, res in enumerate(results):
        obj[i], gap[i] = res

    for i in range(n_tests):
        print(f"[*] instance {random_numbers[i]}: obj={obj[i]:.4f}, gap={gap[i]:.4f}")

    ave_gap = sum(gap) / n_tests
    print(f"[*] ave_gap on size {problem_size}: {ave_gap}")

    # ← ReEvo style: last line is the result, parent process reads this
    print(ave_gap)
