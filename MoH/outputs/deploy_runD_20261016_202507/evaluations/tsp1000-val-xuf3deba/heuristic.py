#{Replaces the static top-10% cutoff with a stagnation-adaptive quota scaling from 8% to 18% combined with a non-adjacent greedy selection to prevent contiguous edge cascades while preserving structural tour stability.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency tracking
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # GLS edge utility evaluation
    utilities = tour_dists / (1.0 + tour_used)

    # Dynamic penalization quota based on stagnation severity
    overlap_ratio = np.mean(tour_used > 1.0)
    adaptive_ratio = 0.08 + 0.10 * np.clip(overlap_ratio * 2.0, 0.0, 1.0)
    target_k = max(2, int(np.ceil(adaptive_ratio * n)))

    # Adaptive selection with non-adjacent filtering to avoid contiguous cascades
    sorted_candidate_indices = np.argsort(-utilities)
    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []

    for idx in sorted_candidate_indices:
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        # Filter out contiguous edges to avoid local path destruction
        if not selected_mask[prev_idx] and not selected_mask[next_idx]:
            selected_mask[idx] = True
            selected_indices.append(idx)
            if len(selected_indices) >= target_k:
                break

    # If target quota was not reached due to anti-adjacency constraints, relax sequentially
    if len(selected_indices) < target_k:
        for idx in sorted_candidate_indices:
            if not selected_mask[idx]:
                selected_mask[idx] = True
                selected_indices.append(idx)
                if len(selected_indices) >= target_k:
                    break

    selected_indices = np.array(selected_indices, dtype=int)

    # Update edge usage counters
    for idx in selected_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    updated_overlap = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * updated_overlap

    # Structural tension and irreplaceability modulation
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

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Apply symmetric modulated penalty to selected edges
    for idx in selected_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist

        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.20 * entropy_mod - 0.25 * irreplaceability_mod
        modulation = max(0.4, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance