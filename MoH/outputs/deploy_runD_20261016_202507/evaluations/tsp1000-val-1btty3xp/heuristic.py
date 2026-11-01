import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Apply global discount factor to track recency and dampen ancient history
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Vectorized O(N) utility calculation: penalize long, under-penalized edges
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Increment usage counts symmetrically for selected bottleneck edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    updated_tour_used = edge_n_used[u_idx, v_idx]
    mean_tour_dist = np.mean(tour_dists) + 1e-12
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # O(N) local density estimation based on tour neighbors
    w_idx = np.roll(u_idx, 1)
    tour_neighbor_dist = 0.5 * (edge_distance[u_idx, v_idx] + edge_distance[u_idx, w_idx]) + 1e-12
    local_density = mean_tour_dist / tour_neighbor_dist
    norm_density = local_density / (np.max(local_density) + 1e-12)

    base_penalty_scale = 0.30 * mean_tour_dist * stagnation_factor

    # Recreate: Sub-linear saturating response using log1p and sqrt to prevent over-penalization
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        density_mod = 0.5 * (norm_density[u] + norm_density[v])

        # Soft-saturating usage factor avoids runaway penalties on structural bottlenecks
        usage_val = edge_n_used[u, v]
        saturating_usage = np.log1p(usage_val) + 0.5 * np.sqrt(usage_val)

        penalty = base_penalty_scale * (1.0 + 0.5 * rel_len + 0.3 * density_mod) * saturating_usage
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance