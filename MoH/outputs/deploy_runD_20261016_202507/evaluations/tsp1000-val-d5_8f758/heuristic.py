#{Replaces incident-edge usage dispersion with a node-level usage entropy and bottleneck replacement ratio that shields irreplaceable bridge edges while amplifying penalties on congested, high-degree alternative nodes.}#

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

    # Guided Local Search utility calculation
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Geometry: Skip shortcut / triangle tension
    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    # Local bottleneck centrality and replacement ratio:
    # Evaluate how viable the direct local alternative (chord) is compared to current tour edges
    # High replacement cost indicates a bridge-like, structurally constrained edge that should be protected
    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    # Incident edge usage concentration (entropy proxy on incident tour edges)
    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    # Normalized incident entropy in [0, 1] where 1 means equally used, 0 means heavily skewed
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])

        rel_len = tour_dists[idx] / mean_tour_dist

        # Shield irreplaceable edges (high irreplaceability_mod) while penalizing redundant high-entropy configurations
        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.20 * entropy_mod - 0.25 * irreplaceability_mod
        modulation = max(0.4, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance