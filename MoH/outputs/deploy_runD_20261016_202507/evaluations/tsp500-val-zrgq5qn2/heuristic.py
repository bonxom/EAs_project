#{Penalizes long, high-frequency tour edges scaled by local geometric density and stagnation-driven utility while shielding sparse boundary connections, specifically tailored for efficient scaling on 500-node instances.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    mean_dist = np.mean(tour_dists)
    rel_dists = tour_dists / (mean_dist + 1e-12)

    # Guided local search utility: edge length over (1 + usage)
    utilities = rel_dists / (1.0 + 0.5 * tour_used)

    # Select top edges with highest utility to penalize
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Local density estimation via small neighborhood
    k_nn = min(8, num_nodes - 1)
    # Using partition for efficiency on N=500
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(sorted_part, axis=1) + 1e-12
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Measure stagnation from tour reuse
    avg_usage = np.mean(tour_used)
    stagnation = 1.0 + np.tanh(avg_usage * 0.15)

    base_penalty_scale = 0.25 * mean_dist * stagnation

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        penalty = base_penalty_scale * (1.0 + 0.4 * density_mod) * (1.0 + 0.1 * edge_n_used[u, v])

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance