#{Replaces 1D contiguous segment tracking with vectorized 2-opt reconnecting bypass impact forecasting, scaling edge penalties by local alternative reconnection slack and topological vulnerability to accelerate escape from persistent local minima.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decay history tracking
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # k-NN local distance normalization
    k_nn = min(8, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(part_dists, axis=1) + 1e-12
    min_nbr_u = part_dists[:, 0]

    edge_min_base = 0.5 * (min_nbr_u[u_idx] + min_nbr_u[v_idx])
    slackness = (tour_dists - edge_min_base) / (0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12)
    slackness = np.maximum(0.0, slackness)

    rel_dists = tour_dists / (mean_dist + 1e-12)
    efficiency_utility = rel_dists / (1.0 + 0.5 * tour_used)

    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Stagnation estimation via usage entropy and cycling indicators
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_dist = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    max_entropy = np.log(num_nodes) + 1e-12
    norm_entropy = shannon_entropy / max_entropy

    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    stagnation_cycle = cycling_ratio + 1.5 * high_usage_ratio + 0.5 * entropy_stagnation

    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))

    rank_eff = np.argsort(np.argsort(efficiency_utility))
    rank_slack = np.argsort(np.argsort(slackness))
    pareto_score = 0.55 * rank_eff + 0.45 * rank_slack
    top_k_indices = np.argpartition(pareto_score, -k)[-k:]

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # 2-opt reconnecting bypass impact forecasting:
    # Estimate the bypass slack for each candidate edge (u, v) by testing reconnection with neighbor edges
    prev_u = np.roll(u_idx, 1)
    next_v = np.roll(v_idx, -1)

    cost_curr_prev = tour_dists + edge_distance[prev_u, u_idx]
    cost_swap_prev = edge_distance[prev_u, v_idx] + edge_distance[u_idx, next_v]
    gain_prev = cost_curr_prev - cost_swap_prev

    cost_curr_next = tour_dists + edge_distance[v_idx, next_v]
    cost_swap_next = edge_distance[u_idx, next_v] + edge_distance[prev_u, v_idx]
    gain_next = cost_curr_next - cost_swap_next

    recon_slack = np.maximum(gain_prev, gain_next)
    norm_recon_slack = (recon_slack - np.mean(recon_slack)) / (np.std(recon_slack) + 1e-12)
    bypass_multiplier = 1.0 + 0.35 * np.tanh(norm_recon_slack)

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)
        bypass_mod = bypass_multiplier[idx]

        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * bypass_mod
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance