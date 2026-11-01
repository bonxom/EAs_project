#{A ruin-and-spread edge penalty mechanism that identifies bottleneck edges via utility scoring and propagates decayed fractional penalties to immediately incident tour edges, forcing broader local-search topological rewirings while maintaining O(n) computational complexity.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Historical edge usage geometric decay
    edge_n_used *= 0.95

    # Fast O(n) evaluation of edges on the tour
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    # Select top bottleneck edges via argpartition
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Increment usage on selected edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    updated_tour_used = edge_n_used[u_idx, v_idx]
    mean_tour_dist = np.mean(tour_dists) + 1e-12
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Local density estimation via neighboring edges along the tour
    w_idx = np.roll(u_idx, 1)
    tour_neighbor_dist = 0.5 * (tour_dists + edge_distance[u_idx, w_idx]) + 1e-12
    local_density = mean_tour_dist / tour_neighbor_dist
    norm_density = local_density / (np.max(local_density) + 1e-12)

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Pre-allocate array for direct and propagated edge penalties
    edge_penalties = np.zeros(n, dtype=np.float64)
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        edge_penalties[idx] = base_penalty_scale * (1.0 + 0.5 * rel_len + 0.5 * density_mod) * edge_n_used[u, v]

    # Ruin-and-spread: bleed over penalty to immediate adjacent edges along the tour
    spread_factor = 0.35
    left_penalties = np.roll(edge_penalties, 1) * spread_factor
    right_penalties = np.roll(edge_penalties, -1) * spread_factor
    total_penalties = edge_penalties + left_penalties + right_penalties

    # Apply combined direct and spread penalties to the distance matrix
    active_mask = total_penalties > 0.0
    active_indices = np.flatnonzero(active_mask)
    u_active = u_idx[active_indices]
    v_active = v_idx[active_indices]
    penalties_active = total_penalties[active_indices]

    np.add.at(updated_edge_distance, (u_active, v_active), penalties_active)
    np.add.at(updated_edge_distance, (v_active, u_active), penalties_active)

    return updated_edge_distance