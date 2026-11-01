#{Dynamically adapts the penalization quota k between 5% and 18% of tour edges driven by stagnation and overlap feedback while preserving linear-time local density and tour-length-scaled edge penalties.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Geometric decay of historical usage
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Assess stagnation and tour overlap before selection to dynamically size quota k
    overlap_ratio = np.mean(tour_used > 0.8)
    # Dynamic quota interpolation from 5% up to 18% based on overlap stagnation
    quota_ratio = 0.05 + 0.13 / (1.0 + np.exp(-6.0 * (overlap_ratio - 0.35)))
    k_penalize = int(np.clip(np.round(quota_ratio * n), 1, n - 1))

    utilities = tour_dists / (1.0 + tour_used)
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Update usage counts for chosen critical edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    w_idx = np.roll(u_idx, 1)
    tour_neighbor_dist = 0.5 * (edge_distance[u_idx, v_idx] + edge_distance[u_idx, w_idx]) + 1e-12
    local_density = mean_tour_dist / tour_neighbor_dist
    norm_density = local_density / (np.max(local_density) + 1e-12)

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        penalty = base_penalty_scale * (1.0 + 0.5 * rel_len + 0.5 * density_mod) * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance