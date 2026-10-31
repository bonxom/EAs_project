# {Combines bottleneck-bridge metrics with detour slackness to dynamically modulate non-linear increment accumulation and spatial propagation, aggressively penalizing redundant detour edges while conserving structural bridge connections.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    if n < 3:
        return updated_edge_distance

    tour = np.asarray(local_opt_tour, dtype=int)
    u_idx = tour
    v_idx = np.roll(tour, -1)
    prev_idx = np.roll(tour, 1)
    next2_idx = np.roll(tour, -2)

    costs = edge_distance[u_idx, v_idx].astype(float)
    penalties = edge_n_used[u_idx, v_idx].astype(float)
    total_nodes = edge_distance.shape[0]

    # Evaluate bridge relief via 2-opt alternative links
    c_prev_v = edge_distance[prev_idx, v_idx].astype(float)
    c_u_next2 = edge_distance[u_idx, next2_idx].astype(float)
    bridge_relief = np.minimum(c_prev_v, c_u_next2) / (costs + 1e-6)

    k_nn = min(5, total_nodes - 1)
    if k_nn > 1:
        knn_u_args = np.argpartition(edge_distance[u_idx], range(1, k_nn + 1), axis=1)[:, 1:k_nn + 1]
        knn_v_args = np.argpartition(edge_distance[v_idx], range(1, k_nn + 1), axis=1)[:, 1:k_nn + 1]

        is_u_in_knn_v = np.any(knn_v_args == u_idx[:, None], axis=1)
        is_v_in_knn_u = np.any(knn_u_args == v_idx[:, None], axis=1)
        mutual_knn = is_u_in_knn_v & is_v_in_knn_u

        shared_neighbors = np.sum(knn_u_args[:, :, None] == knn_v_args[:, None, :], axis=(1, 2))
        cluster_bridge_score = 1.0 / (1.0 + shared_neighbors)

        part_u = np.partition(edge_distance[u_idx], (1, k_nn), axis=1)
        part_v = np.partition(edge_distance[v_idx], (1, k_nn), axis=1)
        d1_u, dk_u = part_u[:, 1], part_u[:, k_nn]
        d1_v, dk_v = part_v[:, 1], part_v[:, k_nn]

        delta_u = np.maximum(dk_u - d1_u, 1e-6)
        delta_v = np.maximum(dk_v - d1_v, 1e-6)
        slack_u = (costs - d1_u) / delta_u
        slack_v = (costs - d1_v) / delta_v
        detour_slackness = np.maximum(slack_u, slack_v)

        effective_bridge_metric = bridge_relief * (0.7 + 0.6 * cluster_bridge_score) * np.where(mutual_knn, 0.8, 1.0)
        bottleneck_discount = 1.0 / (1.0 + np.exp(np.clip(effective_bridge_metric - 2.0, -4.0, 4.0)))
        slack_factor = np.clip(1.0 + detour_slackness * (0.35 + 0.65 * bottleneck_discount), 0.25, 4.0)
        gap_alt = np.maximum(dk_u, dk_v) + 1e-6
    else:
        detour_slackness = np.zeros(n, dtype=float)
        bottleneck_discount = 1.0 / (1.0 + np.exp(np.clip(bridge_relief - 1.8, -4.0, 4.0)))
        slack_factor = np.ones(n, dtype=float)
        gap_alt = np.ones(n, dtype=float)

    adj_weights = (np.roll(costs, 1) + 2.0 * costs + np.roll(costs, -1)) / 4.0
    utilities = (costs * adj_weights * slack_factor * (0.55 + 0.45 * bottleneck_discount)) / ((1.0 + penalties) * gap_alt)
    max_util = np.max(utilities)

    mean_penalty = np.mean(penalties)
    stagnation = 1.0 - np.exp(-mean_penalty / (np.sqrt(n) + 1e-6))
    mean_cost = np.mean(costs)
    avg_tour_cost = mean_cost if mean_cost > 0 else 1.0
    std_cost = np.std(costs)
    cv_cost = std_cost / avg_tour_cost

    sensitivity = 0.12 + 0.12 * np.clip(cv_cost, 0.0, 1.0) * stagnation + 0.08 * stagnation
    threshold = max_util * (1.0 - np.clip(sensitivity, 0.08, 0.35))
    selected_indices = np.where(utilities >= (threshold - 1e-8))[0]

    if stagnation > 0.65:
        radius = 2
    elif stagnation > 0.25:
        radius = 1
    else:
        radius = 0

    std_denom = std_cost + 1e-6
    edge_increments = {}

    def accumulate_increment(edge_key, inc_val, local_discount):
        old_inc = edge_increments.get(edge_key, 0.0)
        if stagnation < 0.25:
            edge_increments[edge_key] = max(old_inc, inc_val)
        else:
            # Conservative sub-linear saturation on bridges; aggressive accumulation on detours
            eff_beta = (0.5 * (1.0 - 0.5 * stagnation)) * (1.5 - 0.7 * local_discount)
            edge_increments[edge_key] = old_inc + (inc_val / (1.0 + eff_beta * old_inc))

    for idx in selected_indices:
        b_disc = bottleneck_discount[idx]
        inc_curr = (costs[idx] / avg_tour_cost) * (0.65 + 0.35 * b_disc)
        e_curr = (min(u_idx[idx], v_idx[idx]), max(u_idx[idx], v_idx[idx]))
        accumulate_increment(e_curr, inc_curr, b_disc)

        if radius > 0:
            cost_z = (costs[idx] - mean_cost) / std_denom
            slack_boost = np.clip(detour_slackness[idx] * 0.10, 0.0, 0.20)
            base_decay = 0.25 + slack_boost + 0.25 / (1.0 + np.exp(-cost_z))
            for offset in range(1, radius + 1):
                spatial_discount = (base_decay * stagnation) ** offset
                for sgn in (-1, 1):
                    h_idx = (idx + sgn * offset) % n
                    e_h = (min(u_idx[h_idx], v_idx[h_idx]), max(u_idx[h_idx], v_idx[h_idx]))
                    h_disc = bottleneck_discount[h_idx]
                    inc_adj = spatial_discount * (costs[h_idx] / avg_tour_cost) * (0.65 + 0.35 * h_disc)
                    accumulate_increment(e_h, inc_adj, h_disc)

    for (u, v), inc in edge_increments.items():
        updated_edge_distance[u, v] += inc
        updated_edge_distance[v, u] += inc
        edge_n_used[u, v] += 1
        edge_n_used[v, u] += 1

    return updated_edge_distance