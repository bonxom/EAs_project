#{Reconstructs regret utility via directional replacement opportunity costs from mutual nearest neighbor neighborhoods weighted by spatial centrality gradients, modulating decayed edge penalties to break stagnation in local optima.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    p_idx = np.roll(u_idx, 1)

    # Decayed frequency tracking
    edge_n_used *= 0.96
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Node-level spatial centrality gradient
    node_mean_dists = np.mean(edge_distance, axis=1)
    global_mean = np.mean(node_mean_dists)
    node_centrality = global_mean / (node_mean_dists + 1e-12)
    norm_centrality = node_centrality / (np.max(node_centrality) + 1e-12)

    # Dynamic k-NN alternative pool excluding active tour connections
    k_nn = min(14, num_nodes - 1)
    part_indices = np.argpartition(edge_distance, k_nn, axis=1)[:, :k_nn]

    # Compute directed asymmetric replacement opportunity costs for each edge (u -> v)
    u_part = part_indices[u_idx, :]
    v_part = part_indices[v_idx, :]

    # Exclude trivial connections (self, predecessor, successor)
    u_dists = edge_distance[u_idx[:, None], u_part]
    v_dists = edge_distance[v_idx[:, None], v_part]

    mask_u = (u_part != u_idx[:, None]) & (u_part != v_idx[:, None]) & (u_part != p_idx[:, None])
    mask_v = (v_part != v_idx[:, None]) & (v_part != u_idx[:, None])

    # Filtered k-nearest alternatives
    u_dists_clean = np.where(mask_u, u_dists, np.inf)
    v_dists_clean = np.where(mask_v, v_dists, np.inf)

    u_sorted_alts = np.sort(u_dists_clean, axis=1)[:, :4]
    v_sorted_alts = np.sort(v_dists_clean, axis=1)[:, :4]

    alt_cost_u = np.mean(u_sorted_alts, axis=1)
    alt_cost_v = np.mean(v_sorted_alts, axis=1)

    # Spatial centrality gradient between u and v
    centrality_gradient = np.abs(norm_centrality[u_idx] - norm_centrality[v_idx])
    grad_weight_u = norm_centrality[u_idx] / (norm_centrality[u_idx] + norm_centrality[v_idx] + 1e-12)
    grad_weight_v = 1.0 - grad_weight_u

    # Directional alternative baseline
    directional_baseline = grad_weight_u * alt_cost_u + grad_weight_v * alt_cost_v

    # Directional regret calculation
    local_scale = 0.5 * (alt_cost_u + alt_cost_v) + 1e-12
    directional_regret = np.maximum(0.0, tour_dists - directional_baseline) / local_scale

    # Rank penalty factor against k-NN distribution
    rank_u = np.sum(u_dists < tour_dists[:, None], axis=1)
    rank_v = np.sum(v_dists < tour_dists[:, None], axis=1)
    rank_factor = 1.0 + np.tanh(0.25 * (rank_u + rank_v))

    # Regret utility combined with frequency suppression
    regret_utility = ((directional_regret * rank_factor) + (tour_dists / (mean_dist + 1e-12)) * (1.0 + centrality_gradient)) / (1.0 + 0.45 * tour_used)

    # Stagnation and burst tracking
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
    top_k_indices = np.argpartition(regret_utility, -k)[-k:]

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Compute contiguous segment lengths for penalized edges
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

    bridge_vulnerability = tour_dists / local_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    # Apply symmetric penalization
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        cent_mod = 0.5 * (norm_centrality[u] + norm_centrality[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]

        segment_multiplier = 1.0 + 0.35 * np.log1p(max(0.0, seg_len - 1.0))
        bridge_factor = 1.0 + 0.30 * np.tanh(norm_bridge[idx] - 1.0)
        usage_factor = 1.0 + 0.20 * np.sqrt(current_usage)

        penalty = base_penalty_scale * (1.0 + 0.35 * cent_mod) * usage_factor * segment_multiplier * bridge_factor
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance