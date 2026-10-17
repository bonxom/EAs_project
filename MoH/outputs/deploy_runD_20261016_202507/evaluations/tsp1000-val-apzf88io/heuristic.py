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
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio
    unique_candidates = np.unique(np.concatenate([u_idx[top_indices], v_idx[top_indices]]))
    sub_dists = edge_distance[unique_candidates]
    k_nn = min(4, n - 1)
    part_k = np.partition(sub_dists, k_nn, axis=1)[:, 1:k_nn]
    node_alt_baseline = np.mean(part_k, axis=1) + 1e-12
    alt_map = np.zeros(n)
    alt_map[unique_candidates] = node_alt_baseline
    incident_sum = pred_dists + tour_dists
    skip_dists = edge_distance[w_idx, v_idx]
    zigzag_tension = incident_sum / (skip_dists + 1e-12)
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)
    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))
    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        e_dist = tour_dists[idx]
        geom_alt = 0.5 * (alt_map[u] + alt_map[v])
        opp_ratio = e_dist / geom_alt
        opp_mod = np.clip(opp_ratio - 1.0, -0.5, 1.5)
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = e_dist / mean_tour_dist
        modulation = 1.0 + 0.30 * rel_len + 0.20 * strain_mod + 0.20 * entropy_mod + 0.25 * opp_mod
        modulation = max(0.35, modulation)
        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty
    return updated_edge_distance