# {Applies guided local search edge penalties using a utility function scaled by average k-nearest-neighbor distances, modulated by local node density and adaptive stagnation to penalize long edges in dense clusters while preserving structural geometry at scale 500.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Track usage decay and edge utilities
    edge_n_used *= 0.96
    tour_dist = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Compute edge utilities for GLS
    utilities = tour_dist / (1.0 + tour_used)

    # For size 500, penalize top edges with highest utility (top ~10-12%)
    k = max(1, int(np.ceil(0.10 * num_nodes)))
    top_indices = np.argsort(utilities)[-k:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    # Spatial density estimation using k-nearest neighbors
    k_nn = min(8, num_nodes - 1)
    sorted_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn + 1]
    local_scale = np.mean(sorted_dists, axis=1) + 1e-12
    density = 1.0 / local_scale
    norm_density = density / (np.mean(density) + 1e-12)

    # Adaptive penalty scaling based on tour length and stagnation
    avg_tour_dist = np.mean(tour_dist)
    tour_overlap = np.mean(edge_n_used[u_idx, v_idx] > 1.0)
    stagnation_factor = 1.0 + 1.2 * tour_overlap

    lambda_base = 0.25 * (avg_tour_dist / num_nodes) * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_len = edge_distance[u, v]
        edge_rel_len = edge_len / (0.5 * (local_scale[u] + local_scale[v]))
        cluster_factor = 0.5 * (norm_density[u] + norm_density[v])

        penalty = lambda_base * (1.0 + 0.3 * cluster_factor + 0.2 * np.tanh(edge_rel_len)) * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance