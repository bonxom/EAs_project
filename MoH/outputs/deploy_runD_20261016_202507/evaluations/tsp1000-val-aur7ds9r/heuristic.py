#{Computes structural geometric strain via incident tour edge asymmetry and triangle tension while modulating penalties with usage dispersion to shield critical bridge edges in O(N) time.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)  # predecessor in tour

    # Discount prior edge memory
    edge_n_used *= 0.95

    # Compute tour edge lengths and utilities
    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    # Vectorized O(N) selection of top candidates
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Update edge usage counts symmetrically
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Quantify incident edge tension & triangle-inequality strain in O(N)
    # Shortcut distance skipping vertex u: d(w, v)
    skip_dists = edge_distance[w_idx, v_idx]
    # Triangle tension: detour ratio relative to direct chord
    # High detour ratio indicates a sharp, redundant zigzag; low ratio indicates a flat, stretched bridge
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    # Node usage dispersion: evaluate how frequently incidents to u have been penalized
    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    usage_dispersion = np.abs(succ_used - pred_used) / (succ_used + pred_used + 1.0)

    # Base penalty scale calibrated to mean tour distance and stagnation
    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Apply modulated penalties
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        # Structural strain modulation: reward breaking redundant sharp turns, protect stretched bridges
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        dispersion_mod = 0.5 * (usage_dispersion[idx] + usage_dispersion[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist

        modulation = 1.0 + 0.4 * rel_len + 0.3 * strain_mod - 0.2 * dispersion_mod
        modulation = max(0.5, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance