import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)      # predecessor of u
    z_idx = np.roll(u_idx, -2)     # successor of v

    # Decayed frequency memory
    edge_n_used *= 0.95

    # Edge distances and utilities
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    # Top edge selection via utility
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    # Stagnation factor based on tour usage overlap
    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Geometry-aware alternative baseline for involved nodes
    unique_candidates = np.unique(np.concatenate([u_idx[top_indices], v_idx[top_indices]]))
    sub_dists = edge_distance[unique_candidates]
    k_nn = min(4, n - 1)
    part_k = np.partition(sub_dists, k_nn, axis=1)[:, 1:k_nn]
    node_alt_baseline = np.mean(part_k, axis=1) + 1e-12
    alt_map = np.zeros(n)
    alt_map[unique_candidates] = node_alt_baseline

    # 4-node 2-opt alternative cost surrogate in the local 4-hop chain (w -> u -> v -> z)
    # Compares current chain edges with potential 2-opt swaps reconnecting around (u, v)
    current_cost_pred = edge_distance[w_idx, u_idx] + tour_dists
    current_cost_succ = tour_dists + edge_distance[v_idx, z_idx]
    alt_cost_pred = edge_distance[w_idx, v_idx] + edge_distance[u_idx, z_idx]
    alt_cost_succ = edge_distance[w_idx, z_idx] + tour_dists

    delta_pred = alt_cost_pred - current_cost_pred
    delta_succ = alt_cost_succ - current_cost_succ
    min_swap_delta = np.minimum(delta_pred, delta_succ)

    # Normalized 2-opt shortcut susceptibility: lower delta indicates an accessible alternate path
    # Convert delta to a susceptibility metric (clipped positive responsiveness)
    surrogate_rel = min_swap_delta / mean_tour_dist
    shortcut_susceptibility = np.clip(1.0 - surrogate_rel, 0.0, 2.5)

    # Incident edge entropy
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

        shortcut_mod = shortcut_susceptibility[idx]
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = e_dist / mean_tour_dist

        modulation = 1.0 + 0.30 * rel_len + 0.25 * shortcut_mod + 0.15 * entropy_mod + 0.25 * opp_mod
        modulation = max(0.35, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance