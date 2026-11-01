import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decayed frequency tracking
    edge_n_used *= 0.96
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # RUIN & RECREATE: Dynamic neighborhood scope and angular/quadrant dispersion
    # Partition a moderately sized candidate neighborhood per node
    k_max = min(16, num_nodes - 1)
    part_idx = np.argpartition(edge_distance, k_max, axis=1)[:, :k_max]
    row_indices = np.arange(num_nodes)[:, None]
    candidate_dists = edge_distance[row_indices, part_idx]
    sort_orders = np.argsort(candidate_dists, axis=1)
    sorted_part_idx = np.take_along_axis(part_idx, sort_orders, axis=1)
    sorted_dists = np.take_along_axis(candidate_dists, sort_orders, axis=1)

    # Calculate local dispersion (coefficient of variation of 2nd to 8th NN distances)
    core_dists = sorted_dists[:, 1:min(8, k_max)]
    core_mean = np.mean(core_dists, axis=1) + 1e-12
    core_std = np.std(core_dists, axis=1)
    cv = core_std / core_mean

    # Dynamic k scope: clustered areas have tighter scopes, dispersed areas wider scopes
    k_scope = np.clip(np.round(4.0 + 6.0 * (1.0 / (1.0 + cv))).astype(int), 3, k_max - 1)

    # Directional/angular balance proxy: approximate relative coordinate projections from pairwise distances
    # For each node's top neighbors, compute directional diversity around the tour edge
    alt_regrets = np.zeros(num_nodes)
    for i in range(num_nodes):
        u, v = u_idx[i], v_idx[i]
        d_uv = tour_dists[i]

        # Consider alternatives for u within dynamic scope
        u_k = k_scope[u]
        nbrs_u = sorted_part_idx[u, 1:u_k]
        nbrs_u = nbrs_u[nbrs_u != v]

        if len(nbrs_u) > 0:
            d_uw = edge_distance[u, nbrs_u]
            d_vw = edge_distance[v, nbrs_u]
            # Law of cosines approximation for angle (v-u-w)
            cos_theta = (d_uv**2 + d_uw**2 - d_vw**2) / (2.0 * d_uv * d_uw + 1e-12)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Quadrant/lateral diversity: prioritize alternatives that deviate from current collinear/congested corridor
            angular_penalty_weight = 1.0 + 0.5 * (1.0 - np.abs(cos_theta))
            effective_alt_dists = d_uw / angular_penalty_weight
            best_alt = np.min(effective_alt_dists)
        else:
            best_alt = core_mean[u]

        alt_regrets[i] = best_alt

    local_scale = core_mean
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    # Rank utility calculation
    rank_u = np.sum(sorted_dists[u_idx, :] < tour_dists[:, None], axis=1)
    rank_v = np.sum(sorted_dists[v_idx, :] < tour_dists[:, None], axis=1)
    mean_rank = 0.5 * (rank_u + rank_v)
    rank_penalty_factor = 1.0 + np.tanh(mean_rank / 2.0)

    # Direction-informed regret
    edge_regret = np.maximum(0.0, tour_dists - alt_regrets)
    norm_regret = edge_regret / edge_scale

    regret_utility = ((norm_regret * rank_penalty_factor) + (tour_dists / (mean_dist + 1e-12))) / (1.0 + 0.45 * tour_used)

    # Stagnation and entropy diagnostics
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

    # Adaptive top-k selection
    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))
    top_k_indices = np.argpartition(regret_utility, -k)[-k:]

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Consecutive penalized segment detection
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

    # Apply penalties
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