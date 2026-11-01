#{Replaces Pareto rank selection with an adaptive incident-edge regret utility that measures the opportunity cost of replacing tour edges against 2nd and 3rd nearest candidate neighbors, modulated by historical visit frequency, local density, and stagnation amplification.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decayed frequency memory
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Neighborhood geometry and k-NN partitioning
    k_nn = min(8, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, :k_nn]
    sorted_part = np.sort(part_dists, axis=1)

    # 1st, 2nd, and 3rd nearest candidate distances (excluding self at index 0)
    best_cands = sorted_part[:, 1]
    second_cands = sorted_part[:, 2] if sorted_part.shape[1] > 2 else sorted_part[:, 1]
    third_cands = sorted_part[:, 3] if sorted_part.shape[1] > 3 else second_cands

    local_scale = np.mean(sorted_part[:, 1:k_nn], axis=1) + 1e-12
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    # Regret computation: opportunity cost of current edge relative to 2nd & 3rd alternate incident candidates
    alt_cost_u = 0.6 * second_cands[u_idx] + 0.4 * third_cands[u_idx]
    alt_cost_v = 0.6 * second_cands[v_idx] + 0.4 * third_cands[v_idx]
    edge_regret = np.maximum(0.0, tour_dists - 0.5 * (alt_cost_u + alt_cost_v))

    # Adaptive regret-based utility: high regret scaled inversely with usage frequency to prioritize viable escapes
    norm_regret = edge_regret / edge_scale
    regret_utility = (norm_regret + (tour_dists / (mean_dist + 1e-12))) / (1.0 + 0.45 * tour_used)

    # Density and stagnation tracking
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    total_tour_used = np.sum(tour_used) + 1e-12
    prob_dist = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    norm_entropy = shannon_entropy / (np.log(num_nodes) + 1e-12)

    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    stagnation_cycle = cycling_ratio + 1.5 * high_usage_ratio + 0.5 * entropy_stagnation

    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))

    # Select top-k edges with highest regret-based escape potential
    top_k_indices = np.argpartition(regret_utility, -k)[-k:]

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Identify penalized contiguous segment lengths
    is_penalized = np.zeros(num_nodes, dtype=bool)
    is_penalized[top_k_indices] = True
    segment_lengths = np.ones(num_nodes, dtype=float)
    if np.any(is_penalized):
        if np.all(is_penalized):
            segment_lengths[:] = num_nodes
        else:
            first_unpen = np.where(~is_penalized)[0][0]
            rolled_pen = np.roll(is_penalized, -first_unpen)
            diffs = np.diff(rolled_pen.astype(np.int8))
            starts = np.where(diffs == 1)[0] + 1
            ends = np.where(diffs == -1)[0] + 1
            rolled_lengths = np.zeros(num_nodes, dtype=float)
            for s, e in zip(starts, ends):
                length = e - s
                rolled_lengths[s:e] = length
            segment_lengths = np.roll(rolled_lengths, first_unpen)

    bridge_vulnerability = tour_dists / edge_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    # Apply modulated penalties
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.35 * np.log1p(np.maximum(0.0, seg_len - 1.0))
        bridge_factor = 1.0 + 0.30 * np.tanh(norm_bridge[idx] - 1.0)
        usage_factor = 1.0 + 0.20 * np.sqrt(current_usage)
        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * segment_multiplier * bridge_factor
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance