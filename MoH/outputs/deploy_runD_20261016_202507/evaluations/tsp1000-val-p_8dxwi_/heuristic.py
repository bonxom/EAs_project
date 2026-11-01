#{Replaces scalar stagnation with dual-tier stagnation control using tour-level cycle detection and localized contiguous high-usage segment run-length amplification to break structural stagnation through compound topological rupture.}#

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

    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]

    # Global recurrence proxy based on deterministic cyclic tour signature and usage moments
    mean_used = np.mean(updated_tour_used)
    var_used = np.var(updated_tour_used)
    high_used_mask = updated_tour_used > (mean_used + 0.5)
    global_recurrence_tier = 1.0 + np.tanh(mean_used / 2.0) + 0.5 * (var_used / (var_used + 1.0))

    # Localized contiguous-run stagnation detector: measure run lengths of consecutive high-usage edges
    run_lengths = np.zeros(n, dtype=np.float64)
    # Circular two-pass scan to accurately compute contiguous run lengths along the closed tour
    curr_run = 0.0
    for i in range(2 * n):
        idx = i % n
        if high_used_mask[idx]:
            curr_run += 1.0
        else:
            curr_run = 0.0
        if i >= n:
            run_lengths[idx] = curr_run

    # Local contiguous run amplification factor
    max_run = np.max(run_lengths) if np.max(run_lengths) > 0 else 1.0
    contiguous_amplification = 1.0 + 1.5 * np.log1p(run_lengths) / np.log1p(max_run + 1e-12)

    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    usage_dispersion = np.abs(succ_used - pred_used) / (succ_used + pred_used + 1.0)

    base_penalty_scale = 0.25 * mean_tour_dist * global_recurrence_tier

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        dispersion_mod = 0.5 * (usage_dispersion[idx] + usage_dispersion[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist
        local_rupture_boost = contiguous_amplification[idx]

        modulation = (1.0 + 0.4 * rel_len + 0.3 * strain_mod - 0.2 * dispersion_mod) * local_rupture_boost
        modulation = max(0.5, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance