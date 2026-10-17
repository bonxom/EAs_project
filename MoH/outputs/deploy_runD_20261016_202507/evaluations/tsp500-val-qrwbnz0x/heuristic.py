#{Applies guided local search edge penalties for size-500 TSP by penalizing highest utility tour edges scaled by local density, tour stagnation, and edge usage.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Track edge usage with mild decay to allow diversification
    edge_n_used *= 0.96
    tour_dist = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Utility formulation prioritizing long, infrequently penalized edges
    utilities = tour_dist / (1.0 + tour_used)

    # Adaptive selection of edges to penalize (top ~10% for n=500)
    k = max(1, int(np.ceil(0.10 * num_nodes)))
    top_k_indices = np.argsort(utilities)[-k:]

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    # Spatial k-nearest neighbor density calculation
    sorted_dists = np.sort(edge_distance, axis=1)
    k_nn = min(10, num_nodes - 1)
    k_nn_avg = np.mean(sorted_dists[:, 1:k_nn + 1], axis=1)
    node_density = 1.0 / (k_nn_avg + 1e-8)
    norm_density = node_density / (np.mean(node_density) + 1e-8)

    # Stagnation factor based on average tour usage
    tour_usage = edge_n_used[u_idx, v_idx]
    avg_tour_usage = np.mean(tour_usage)
    max_used = np.max(edge_n_used) + 1e-12
    stagnation_factor = 1.0 + 1.2 * (avg_tour_usage / max_used)

    # Global scale penalty calibration
    mean_tour_dist = np.mean(tour_dist)
    lambda_factor = 0.25 * mean_tour_dist * stagnation_factor

    # Penalize selected top-utility edges modulated by local cluster density
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        penalty = lambda_factor * (1.0 + 0.5 * np.tanh(density_mod)) / (1.0 + 0.1 * tour_used[idx])
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance