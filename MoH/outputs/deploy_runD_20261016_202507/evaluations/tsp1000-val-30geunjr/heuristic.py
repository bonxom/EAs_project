import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Apply usage decay to prioritize recent history
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_tour_dist = np.mean(tour_dists) + 1e-12

    # Assess stagnation state based on existing edge re-usage
    overlap_ratio = np.mean(tour_used > 0.8)
    avg_tour_usage = np.mean(tour_used)
    max_used = np.max(edge_n_used) + 1e-12
    stagnation_ratio = avg_tour_usage / max_used

    # Dynamic adaptation of penalization budget k between 4% and 18%
    # Intensify when exploring novel routes (low overlap), diversify aggressively when stagnant
    stagnation_severity = 0.6 * overlap_ratio + 0.4 * stagnation_ratio
    k_frac = 0.04 + 0.14 * (stagnation_severity ** 1.5)
    k_penalize = max(1, min(n - 1, int(np.ceil(k_frac * n))))

    # Vectorized utility evaluation
    utilities = tour_dists / (1.0 + tour_used)
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Update frequency memory symmetrically
    selected_u = u_idx[top_indices]
    selected_v = v_idx[top_indices]
    edge_n_used[selected_u, selected_v] += 1.0
    edge_n_used[selected_v, selected_u] += 1.0

    # Local context metrics: spatial density and stagnation amplifier
    w_idx = np.roll(u_idx, 1)
    tour_neighbor_dist = 0.5 * (tour_dists + edge_distance[u_idx, w_idx]) + 1e-12
    local_density = mean_tour_dist / tour_neighbor_dist
    norm_density = local_density / (np.max(local_density) + 1e-12)

    stagnation_factor = 1.0 + 1.8 * overlap_ratio + 0.7 * stagnation_ratio
    base_penalty_scale = 0.22 * mean_tour_dist * stagnation_factor

    # Apply modulated penalties
    rel_lens = tour_dists[top_indices] / mean_tour_dist
    density_mods = 0.5 * (norm_density[selected_u] + norm_density[selected_v])
    usage_weights = edge_n_used[selected_u, selected_v]

    penalties = base_penalty_scale * (1.0 + 0.4 * rel_lens + 0.6 * density_mods) * usage_weights

    np.add.at(updated_edge_distance, (selected_u, selected_v), penalties)
    np.add.at(updated_edge_distance, (selected_v, selected_u), penalties)

    return updated_edge_distance