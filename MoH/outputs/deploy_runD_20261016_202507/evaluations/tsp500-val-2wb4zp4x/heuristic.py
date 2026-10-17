#{Preserves decayed frequency and k-NN regret ranking while replacing isolated edge penalties with bilateral spatial spillover dispersion that radiates a fraction of penalty mass into incident nearest-neighbor edges of congested incident endpoints via a localized Gaussian kernel.}

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

    # Neighborhood baseline and regret utility evaluation
    k_nn = min(12, num_nodes - 1)
    part_indices = np.argpartition(edge_distance, k_nn, axis=1)[:, :k_nn]
    rows = np.arange(num_nodes)[:, None]
    part_dists = edge_distance[rows, part_indices]
    sorted_part = np.sort(part_dists, axis=1)

    k_alt = min(5, k_nn)
    alt_pool = sorted_part[:, 1:k_alt]
    harmonic_alt = (k_alt - 1) / np.sum(1.0 / (alt_pool + 1e-12), axis=1)
    local_scale = np.mean(sorted_part[:, 1:min(8, k_nn)], axis=1) + 1e-12
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    rank_u = np.sum(part_dists[u_idx, :] < tour_dists[:, None], axis=1)
    rank_v = np.sum(part_dists[v_idx, :] < tour_dists[:, None], axis=1)
    mean_rank = 0.5 * (rank_u + rank_v)
    rank_penalty_factor = 1.0 + np.tanh(mean_rank / 2.0)

    baseline_alt = 0.5 * (harmonic_alt[u_idx] + harmonic_alt[v_idx])
    edge_regret = np.maximum(0.0, tour_dists - baseline_alt)
    norm_regret = edge_regret / edge_scale
    regret_utility = ((norm_regret * rank_penalty_factor) + (tour_dists / (mean_dist + 1e-12))) / (1.0 + 0.45 * tour_used)

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
    top_k_indices = np.argpartition(regret_utility, -k)[-k:]

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

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
                rolled_lengths[s:e] = e - s
            segment_lengths = np.roll(rolled_lengths, first_unpen)

    bridge_vulnerability = tour_dists / edge_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    k_spill = min(4, num_nodes - 1)

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

        # Primary direct penalty on the chosen edge
        direct_penalty = 0.70 * penalty
        updated_edge_distance[u, v] += direct_penalty
        updated_edge_distance[v, u] += direct_penalty

        # Bilateral spatial spillover dispersion onto incident edges
        spillover_budget = 0.15 * penalty
        for node in (u, v):
            nbrs = part_indices[node, :k_spill]
            valid_nbrs = nbrs[nbrs != (v if node == u else u)]
            if len(valid_nbrs) > 0:
                nbr_dists = edge_distance[node, valid_nbrs]
                sigma = local_scale[node] + 1e-12
                gauss_kernel = np.exp(-0.5 * (nbr_dists / sigma) ** 2)
                kernel_sum = np.sum(gauss_kernel) + 1e-12
                spill_penalties = spillover_budget * (gauss_kernel / kernel_sum)

                updated_edge_distance[node, valid_nbrs] += spill_penalties
                updated_edge_distance[valid_nbrs, node] += spill_penalties

    return updated_edge_distance