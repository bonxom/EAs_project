#{Replaces local two-edge predecessor-successor entropy with a comprehensive Node-Degree Crystallization Index computed over all incident edge usages in memory, triggering a non-linear penalty boost on topologically crystallized trap nodes to compel novel routing pathways.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency memory
    edge_n_used *= 0.95

    # Guided local search utility-based selection
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
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Identification of unique endpoints involved in penalization
    penalized_u = u_idx[top_indices]
    penalized_v = v_idx[top_indices]
    unique_candidates = np.unique(np.concatenate([penalized_u, penalized_v]))

    # Geometric alternative baselines
    sub_dists = edge_distance[unique_candidates]
    k_nn = min(4, n - 1)
    part_k = np.partition(sub_dists, k_nn, axis=1)[:, 1:k_nn]
    node_alt_baseline = np.mean(part_k, axis=1) + 1e-12
    alt_map = np.zeros(n)
    alt_map[unique_candidates] = node_alt_baseline

    # Zigzag tension metric
    incident_sum = pred_dists + tour_dists
    skip_dists = edge_distance[w_idx, v_idx]
    zigzag_tension = incident_sum / (skip_dists + 1e-12)
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    # Node-Degree Crystallization Index computation across entire incident memory
    sub_usage = edge_n_used[unique_candidates]
    row_sums = np.sum(sub_usage, axis=1, keepdims=True)
    active_mask = row_sums.squeeze(1) > 1e-8

    crystallization_map = np.zeros(n)
    if np.any(active_mask):
        active_candidates = unique_candidates[active_mask]
        norm_usage = sub_usage[active_mask] / (row_sums[active_mask] + 1e-12)
        # Shannon entropy of edge frequency distribution per node
        pos_probs = np.where(norm_usage > 1e-12, norm_usage, 1.0)
        entropy = -np.sum(norm_usage * np.log2(pos_probs), axis=1)
        # Count of distinct edges traversed through node
        effective_degree = np.sum(sub_usage[active_mask] > 1e-4, axis=1)
        max_possible_entropy = np.log2(np.maximum(effective_degree, 2))
        normalized_entropy = entropy / (max_possible_entropy + 1e-12)
        # Crystallization is high when usage is concentrated on very few edges (low entropy)
        crystallization_val = np.clip(1.0 - normalized_entropy, 0.0, 1.0)
        crystallization_map[active_candidates] = crystallization_val

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        e_dist = tour_dists[idx]

        geom_alt = 0.5 * (alt_map[u] + alt_map[v])
        opp_ratio = e_dist / geom_alt
        opp_mod = np.clip(opp_ratio - 1.0, -0.5, 1.5)

        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])

        # Exponential surge based on endpoint crystallization
        cryst_score = 0.5 * (crystallization_map[u] + crystallization_map[v])
        cryst_surge = np.exp(1.5 * cryst_score) - 1.0

        rel_len = e_dist / mean_tour_dist
        modulation = 1.0 + 0.30 * rel_len + 0.20 * strain_mod + 0.35 * cryst_surge + 0.25 * opp_mod
        modulation = max(0.35, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance