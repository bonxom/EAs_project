import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency memory
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Guided local search utility scoring
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.12 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Increment usage count for penalized edges symmetrically
    penalized_u = u_idx[top_indices]
    penalized_v = v_idx[top_indices]
    for u, v in zip(penalized_u, penalized_v):
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Geometric tension & irreplaceability
    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    # Compute full incident neighborhood entropy from edge_n_used for penalized endpoints
    active_nodes = np.unique(np.concatenate([penalized_u, penalized_v]))
    active_usage = edge_n_used[active_nodes, :]
    node_tot = np.sum(active_usage, axis=1, keepdims=True) + 1e-12
    norm_p = active_usage / node_tot

    log_p = np.zeros_like(norm_p)
    nonzero_mask = norm_p > 1e-12
    log_p[nonzero_mask] = np.log2(norm_p[nonzero_mask])

    raw_node_entropy = -np.sum(norm_p * log_p, axis=1)
    effective_deg = np.sum(active_usage > 1e-4, axis=1)
    max_h = np.log2(np.maximum(effective_deg, 2.0))
    norm_h = raw_node_entropy / max_h

    entropy_lookup = np.zeros(edge_distance.shape[0])
    entropy_lookup[active_nodes] = norm_h

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        endpoint_entropy = 0.5 * (entropy_lookup[u] + entropy_lookup[v])
        rel_len = tour_dists[idx] / mean_tour_dist

        # High entropy promotes exploration escape; low entropy shields bottleneck corridors
        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.30 * (endpoint_entropy - 0.5) - 0.25 * irreplaceability_mod
        modulation = max(0.3, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance