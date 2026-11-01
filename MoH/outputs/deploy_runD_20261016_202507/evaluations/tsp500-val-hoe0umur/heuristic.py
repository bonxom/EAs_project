#{Combines Guided Local Search edge utility scoring with localized k-nearest-neighbor clustering density and stagnation-driven penalty scaling tailored for large-scale tours.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Trace tour lengths and existing usage counts
    tour_dist = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Calculate guided local search utilities for tour edges
    utilities = tour_dist / (1.0 + tour_used)

    # Penalize a strategic subset of highest-utility (long, under-penalized) edges
    # For n=500, penalizing the top 8% to 12% balances diversification and intensification
    k = max(1, int(np.ceil(0.10 * n)))
    top_k_indices = np.argsort(utilities)[-k:]

    # Increment usage counter for selected edges
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    # Local density estimation using small k-NN neighborhood
    k_nn = min(8, n - 1)
    sorted_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn + 1]
    knn_mean = np.mean(sorted_dists, axis=1) + 1e-12
    node_density = 1.0 / knn_mean
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Search progress and stagnation measurement
    mean_tour_dist = np.mean(tour_dist)
    mean_used = np.mean(edge_n_used[u_idx, v_idx])
    stagnation_factor = 1.0 + 0.5 * np.tanh(mean_used / 3.0)

    # Base penalty scale derived from average tour step size
    lambda_param = 0.30 * mean_tour_dist * stagnation_factor

    # Apply scaled penalties to the selected critical edges
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        # Modulate by local spatial density to break trapped clusters without distorting sparse peripheries
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        penalty = lambda_param * (1.0 + 0.3 * np.tanh(density_mod))

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance