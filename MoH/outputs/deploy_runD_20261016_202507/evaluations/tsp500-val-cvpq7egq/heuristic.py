# {Applies guided local search penalty on high-utility tour edges scaled by local density and stagnation-aware adaptive multiplier, specifically calibrated for 500-city instances.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    edge_n_used *= 0.96

    tour_dist = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Standard guided local search utility: length / (1 + usage)
    utilities = tour_dist / (1.0 + tour_used)

    # Select top edges to penalize (proportional to instance size 500)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argsort(utilities)[-k:]

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    avg_tour_dist = np.mean(tour_dist)
    max_used = np.max(edge_n_used) + 1e-12
    stagnation_ratio = np.mean(tour_used > 1.0)

    # 5-nearest neighbor density calculation
    sorted_dists = np.partition(edge_distance, min(6, num_nodes - 1), axis=1)
    k_nn_avg = np.mean(sorted_dists[:, 1:min(6, num_nodes)], axis=1) + 1e-12
    node_density = 1.0 / k_nn_avg
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Adaptive penalty scaling
    adaptive_lambda = (0.20 + 0.15 * np.tanh(2.0 * stagnation_ratio)) * (avg_tour_dist / num_nodes)

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        d_factor = 0.5 * (norm_density[u] + norm_density[v])
        penalty = adaptive_lambda * (1.0 + 0.5 * d_factor) * (1.0 + edge_n_used[u, v] / max_used)
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance