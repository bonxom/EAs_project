import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Historical edge usage geometric decay
    edge_n_used *= 0.96
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Opportunity cost / regret utility calculation via k-nearest neighbors
    k_nn = min(8, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, :k_nn]
    sorted_part = np.sort(part_dists, axis=1)
    second_cands = sorted_part[:, 2] if sorted_part.shape[1] > 2 else sorted_part[:, 1]
    third_cands = sorted_part[:, 3] if sorted_part.shape[1] > 3 else second_cands

    local_scale = np.mean(sorted_part[:, 1:k_nn], axis=1) + 1e-12
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12
    alt_cost_u = 0.6 * second_cands[u_idx] + 0.4 * third_cands[u_idx]
    alt_cost_v = 0.6 * second_cands[v_idx] + 0.4 * third_cands[v_idx]
    edge_regret = np.maximum(0.0, tour_dists - 0.5 * (alt_cost_u + alt_cost_v))
    norm_regret = edge_regret / edge_scale

    # RUIN & RECREATE: Localized multi-scale stagnation tracking per node neighborhood
    # Compute local incident usage statistics around each node to identify local cycling loops
    node_incident_used = np.sum(edge_n_used, axis=1)

    # Running short-term variance and mean usage in local topological neighborhood along the tour
    # Use a rolling 5-node window along the tour to measure local cycling intensity and entrapment
    w_size = 5
    pad_used = np.pad(tour_used, (w_size // 2, w_size // 2), mode='wrap')
    # Efficient rolling sum and rolling sum of squares using cumulative sums
    cumsum = np.cumsum(np.insert(pad_used, 0, 0))
    cumsum_sq = np.cumsum(np.insert(pad_used ** 2, 0, 0))
    local_mean_used = (cumsum[w_size:] - cumsum[:-w_size]) / w_size
    local_var_used = np.maximum(0.0, (cumsum_sq[w_size:] - cumsum[:-w_size]) / w_size - local_mean_used ** 2)

    # Local stagnation score combining neighborhood usage intensity and variance
    # High mean and low variance indicate uniform repetitive cycling in that sub-region
    # High mean with moderate variance indicates high-frequency churning edges
    local_stagnation = (local_mean_used / (1.0 + np.sqrt(local_var_used))) + 1.2 * (local_mean_used > 1.5).astype(float)
    norm_local_stag = local_stagnation / (np.mean(local_stagnation) + 1e-12)

    # Regret utility modulated by historical usage and regional entrapment
    regret_utility = (norm_regret + (tour_dists / (mean_dist + 1e-12))) / (1.0 + 0.40 * tour_used)
    utility_guided = regret_utility * (1.0 + 0.35 * np.tanh(norm_local_stag))

    # Adaptive k selection based on average local entrapment
    mean_stag = np.mean(local_stagnation)
    k_frac = np.clip(0.10 + 0.08 * np.tanh(mean_stag), 0.08, 0.22)
    k = max(2, int(np.ceil(k_frac * num_nodes)))
    top_k_indices = np.argpartition(utility_guided, -k)[-k:]

    # Structural segment grouping of penalized edges
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

    # Node density and spatial factors
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)
    bridge_vulnerability = tour_dists / edge_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    base_penalty_scale = 0.28 * mean_dist

    # Apply targeted burst penalties specifically on trapped sub-regions
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.30 * np.log1p(np.maximum(0.0, seg_len - 1.0))
        bridge_factor = 1.0 + 0.25 * np.tanh(norm_bridge[idx] - 1.0)
        usage_factor = 1.0 + 0.18 * np.sqrt(current_usage)

        # Localized burst multiplier: amplify heavily only if the edge resides in a stagnated neighborhood
        local_burst = 1.0 + 0.85 * np.tanh(norm_local_stag[idx])

        penalty = base_penalty_scale * local_burst * (1.0 + 0.30 * density_mod) * usage_factor * segment_multiplier * bridge_factor
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance