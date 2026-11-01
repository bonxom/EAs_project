import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # 1. Decay historical edge usage
    edge_n_used *= 0.96
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # 2. Neighborhood geometry and alternative baseline
    k_nn = min(12, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, :k_nn]
    sorted_part = np.sort(part_dists, axis=1)
    k_alt = min(5, k_nn)
    alt_pool = sorted_part[:, 1:k_alt]
    harmonic_alt = (k_alt - 1.0) / np.sum(1.0 / (alt_pool + 1e-12), axis=1)
    local_scale = np.mean(sorted_part[:, 1:min(8, k_nn)], axis=1) + 1e-12
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    # 3. Dynamic rank and regret-augmented utility
    rank_u = np.sum(part_dists[u_idx, :] < tour_dists[:, None], axis=1)
    rank_v = np.sum(part_dists[v_idx, :] < tour_dists[:, None], axis=1)
    mean_rank = 0.5 * (rank_u + rank_v)
    rank_penalty_factor = 1.0 + np.tanh(mean_rank / 2.0)
    baseline_alt = 0.5 * (harmonic_alt[u_idx] + harmonic_alt[v_idx])
    edge_regret = np.maximum(0.0, tour_dists - baseline_alt)
    norm_regret = edge_regret / edge_scale
    regret_utility = ((norm_regret * rank_penalty_factor) + (tour_dists / (mean_dist + 1e-12))) / (1.0 + 0.45 * tour_used)

    # 4. Dual-scale spatial stagnation detection:
    # Tour-level overlap persistence
    overlap_ratio = np.mean(tour_used > 1.2)
    repetition_intensity = np.mean(tour_used) / (np.max(tour_used) + 1e-12)
    persistence_metric = 0.6 * overlap_ratio + 0.4 * repetition_intensity

    # Local spatial divergence of usage (measuring spatial clustering of search effort)
    node_usage_profile = np.sum(edge_n_used, axis=1)
    k_clust = min(10, num_nodes - 1)
    nn_indices = np.argpartition(edge_distance, k_clust, axis=1)[:, :k_clust]
    local_neighbor_usage = np.mean(node_usage_profile[nn_indices], axis=1)
    global_mean_usage = np.mean(node_usage_profile) + 1e-12

    # High variance in local neighbor usage relative to global mean indicates localized spatial clustering
    spatial_dispersion = np.std(local_neighbor_usage) / global_mean_usage
    spatial_stagnation = np.clip(spatial_dispersion, 0.0, 3.0)

    # Combined dual-scale stagnation detector
    stagnation_factor = persistence_metric + 0.75 * spatial_stagnation

    # 5. Adaptive selection fraction (k) and penalty intensity
    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_factor), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))
    top_k_indices = np.argpartition(regret_utility, -k)[-k:]

    burst_multiplier = 1.0 + 2.0 * np.tanh(stagnation_factor)
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # 6. Structural contiguous segment modulation
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

    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)
    bridge_vulnerability = tour_dists / edge_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    # 7. Apply symmetric penalties
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