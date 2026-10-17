#{Replace the monolithic linear penalty with a sub-additive square-root frequency saturation function combined with an exponential threshold on relative edge length to aggressively suppress persistent long-bridge edges while preserving metric fidelity.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Trace-history decay
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Utility calculation using current tour edges
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Increment usage count for selected edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    updated_tour_used = edge_n_used[u_idx, v_idx]
    mean_tour_dist = np.mean(tour_dists) + 1e-12
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Local context density
    w_idx = np.roll(u_idx, 1)
    tour_neighbor_dist = 0.5 * (edge_distance[u_idx, v_idx] + edge_distance[u_idx, w_idx]) + 1e-12
    local_density = mean_tour_dist / tour_neighbor_dist
    norm_density = local_density / (np.max(local_density) + 1e-12)

    # Base penalty scale
    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Non-linear sub-additive penalty dynamics with frequency saturation & exponential length amplification
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        density_mod = 0.5 * (norm_density[u] + norm_density[v])

        # Sub-additive concave frequency response to prevent over-penalization
        usage_val = edge_n_used[u, v]
        freq_response = np.sqrt(usage_val) * (1.0 + 0.5 * np.tanh(usage_val / 4.0))

        # Exponential trigger for edges substantially longer than average
        length_factor = np.exp(np.clip(rel_len - 1.0, -1.0, 3.0))

        # Contextual modulation incorporating density shielding
        modulation = length_factor * (1.0 + 0.4 * density_mod)
        penalty = base_penalty_scale * freq_response * modulation

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance