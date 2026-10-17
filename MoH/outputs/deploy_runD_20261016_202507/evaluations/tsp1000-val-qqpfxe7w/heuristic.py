import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed edge frequency tracking
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Guided Local Search utility-based candidate selection targeting top 10%
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

    # Reconstruct topological resilience using multi-span alternative 2-opt reconnection gap
    # Evaluate 2-opt exchange reconnection costs at offsets +2, +3, -2, -3 along the tour
    offsets = [2, 3, -2, -3]
    candidate_alt_costs = []
    for off in offsets:
        target_u = np.roll(u_idx, -off)
        target_v = np.roll(u_idx, -(off + 1))
        # Reconnection cost of replacing (u, v) and (target_u, target_v) with cross edges (u, target_u) and (v, target_v)
        alt_cost = edge_distance[u_idx, target_u] + edge_distance[v_idx, target_v] - edge_distance[target_u, target_v]
        candidate_alt_costs.append(alt_cost)

    stacked_alts = np.vstack(candidate_alt_costs)
    min_alt_reconnect = np.min(stacked_alts, axis=0)
    # Detour gap: difference between best nearby alternative reconnection and current edge cost
    detour_gap = np.maximum(0.0, min_alt_reconnect - tour_dists)
    norm_detour_gap = detour_gap / (np.mean(detour_gap) + 1e-12)

    # Local strain tension and frequency entropy balance
    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    zigzag_tension = (pred_dists + tour_dists) / chord_base
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
        next_idx = (idx + 1) % n
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[next_idx])
        detour_mod = 0.5 * (norm_detour_gap[idx] + norm_detour_gap[next_idx])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[next_idx])
        rel_len = tour_dists[idx] / mean_tour_dist

        # High detour gap implies critical link that is hard to replace without significant penalty;
        # links with accessible detours (low detour gap) receive stronger perturbation to escape local optima
        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.20 * entropy_mod - 0.20 * detour_mod
        modulation = max(0.4, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance