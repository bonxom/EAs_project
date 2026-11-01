#{Replaces 1D contiguous segment counting with vectorized morphological tour-dilation and mutual nearest-neighbor graph coherence to adaptively modulate regret penalties on structural bottlenecks and dense clusters.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decayed edge frequency tracking
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Neighborhood and local density estimation via k-NN
    k_nn = min(12, num_nodes - 1)
    part_indices = np.argpartition(edge_distance, k_nn, axis=1)[:, :k_nn]
    part_dists = np.take_along_axis(edge_distance, part_indices, axis=1)
    sorted_order = np.argsort(part_dists, axis=1)
    sorted_part = np.take_along_axis(part_dists, sorted_order, axis=1)
    sorted_nn_idx = np.take_along_axis(part_indices, sorted_order, axis=1)

    k_alt = min(5, k_nn)
    alt_pool = sorted_part[:, 1:k_alt]
    harmonic_alt = (k_alt - 1) / np.sum(1.0 / (alt_pool + 1e-12), axis=1)
    local_scale = np.mean(sorted_part[:, 1:min(8, k_nn)], axis=1) + 1e-12
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    # Rank & regret utility computation
    rank_u = np.sum(part_dists[u_idx, :] < tour_dists[:, None], axis=1)
    rank_v = np.sum(part_dists[v_idx, :] < tour_dists[:, None], axis=1)
    mean_rank = 0.5 * (rank_u + rank_v)
    rank_penalty_factor = 1.0 + np.tanh(mean_rank / 2.0)

    baseline_alt = 0.5 * (harmonic_alt[u_idx] + harmonic_alt[v_idx])
    edge_regret = np.maximum(0.0, tour_dists - baseline_alt)
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

    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))
    top_k_indices = np.argpartition(regret_utility, -k)[-k:]

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Vectorized morphological path-dilation along the tour
    is_penalized = np.zeros(num_nodes, dtype=float)
    is_penalized[top_k_indices] = 1.0
    dilated_context = (
        0.50 * is_penalized +
        0.25 * (np.roll(is_penalized, 1) + np.roll(is_penalized, -1)) +
        0.15 * (np.roll(is_penalized, 2) + np.roll(is_penalized, -2)) +
        0.10 * (np.roll(is_penalized, 3) + np.roll(is_penalized, -3))
    )

    # Mutual nearest-neighbor graph alignment & bridge vulnerability
    k_mutual = min(6, k_nn)
    u_neighbors = sorted_nn_idx[u_idx, 1:k_mutual]
    v_neighbors = sorted_nn_idx[v_idx, 1:k_mutual]
    u_in_v = np.any(v_neighbors == u_idx[:, None], axis=1)
    v_in_u = np.any(u_neighbors == v_idx[:, None], axis=1)
    mutual_alignment = (u_in_v.astype(float) + v_in_u.astype(float)) * 0.5

    bridge_vulnerability = (tour_dists / edge_scale) / (mutual_alignment + 0.5)
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]

        morphological_boost = 1.0 + 0.40 * dilated_context[idx]
        bridge_factor = 1.0 + 0.30 * np.tanh(norm_bridge[idx] - 1.0)
        usage_factor = 1.0 + 0.20 * np.sqrt(current_usage)

        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * morphological_boost * bridge_factor
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance