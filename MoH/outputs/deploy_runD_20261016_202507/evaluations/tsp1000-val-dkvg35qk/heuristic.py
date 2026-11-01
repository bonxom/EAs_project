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

    # Dual-memory stagnation monitor: recurrence & Gini inequality of active tour edge usage
    overlap_ratio = np.mean(tour_used > 0.95)
    sorted_usage = np.sort(tour_used)
    total_usage = np.sum(sorted_usage)
    if total_usage > 1e-12:
        idx_weights = 2.0 * np.arange(1, n + 1) - n - 1.0
        gini = np.sum(idx_weights * sorted_usage) / (n * total_usage)
        gini = max(0.0, min(1.0, float(gini)))
    else:
        gini = 0.0

    # Basin stagnation index combining recurrence and usage inequality
    stagnation_index = np.clip(0.6 * overlap_ratio + 0.4 * gini, 0.0, 1.0)

    # Dynamically expand candidate selection budget k from 10% to 20%
    k_fraction = 0.10 + 0.10 * stagnation_index
    k_penalize = max(1, int(np.ceil(k_fraction * n)))

    # Guided local search utility scoring
    utilities = tour_dists / (1.0 + tour_used)
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12

    # Structural geometric properties
    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    # Dynamic stagnation factor with non-linear power boost
    base_penalty_scale = 0.25 * mean_tour_dist * (1.0 + 2.0 * stagnation_index)
    power_boost_exp = 1.0 + 0.5 * stagnation_index

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist

        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.20 * entropy_mod - 0.25 * irreplaceability_mod
        modulation = max(0.4, modulation)

        eff_usage = edge_n_used[u, v] ** power_boost_exp
        penalty = base_penalty_scale * modulation * eff_usage

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance