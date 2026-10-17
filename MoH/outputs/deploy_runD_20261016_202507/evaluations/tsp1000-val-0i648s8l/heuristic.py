#{Replaces static utility and fixed cutoff with a local-density-weighted utility metric and stagnation-responsive dynamic perturbation quota, concentrating penalties on congested sub-optimal bridges while adapting search breadth to escape deep basins.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    edge_n_used *= 0.95
    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Approximate local density efficiently using incident tour and shortcut edges without full O(N^2) sorting
    skip_dists_fwd = edge_distance[u_idx, np.roll(u_idx, -2)]
    skip_dists_bwd = edge_distance[w_idx, v_idx]
    local_scale_u = 0.5 * (pred_dists + tour_dists)
    local_scale_v = 0.5 * (tour_dists + edge_distance[v_idx, np.roll(u_idx, -2)])
    local_neighborhood_scale = 0.5 * (local_scale_u + local_scale_v)
    density_factor = 1.0 / (local_neighborhood_scale + 1e-12)
    norm_density = density_factor / (np.mean(density_factor) + 1e-12)

    # Reconstruct utility: weight edge distances by local neighborhood density and penalize usage smoothly
    density_weighted_dist = tour_dists * (0.6 + 0.4 * norm_density)
    utilities = density_weighted_dist / (1.0 + 0.8 * tour_used)

    # Dynamic k quota selection driven by stagnation and overlap
    mean_tour_dist = np.mean(tour_dists) + 1e-12
    overlap_ratio = np.mean(tour_used > 1.0)
    stagnation_factor = 1.0 + 1.8 * overlap_ratio

    # Dynamically scale k between ~6% (fine perturbation) and ~18% (basin breakout)
    k_fraction = np.clip(0.06 + 0.08 * (stagnation_factor - 1.0), 0.05, 0.20)
    k_penalize = max(1, int(np.ceil(k_fraction * n)))

    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    chord_base = skip_dists_bwd + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    usage_dispersion = np.abs(succ_used - pred_used) / (succ_used + pred_used + 1.0)

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        dispersion_mod = 0.5 * (usage_dispersion[idx] + usage_dispersion[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist
        dens_mod = 0.5 * (norm_density[idx] + norm_density[(idx + 1) % n])

        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.15 * dens_mod - 0.2 * dispersion_mod
        modulation = max(0.5, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance