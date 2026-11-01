#{This algorithm replaces static Borda ranking with an adaptive Chebyshev compromise metric combining edge efficiency, two-hop shortcut detour ratio, and spatial congestion to identify and penalize topological bottleneck edges in Guided Local Search.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    prev_idx = np.roll(u_idx, 1)
    next_idx = np.roll(v_idx, -1)

    # Usage decay
    edge_n_used *= 0.96
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Local neighborhood scale
    k_nn = min(8, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(part_dists, axis=1) + 1e-12
    min_nbr_u = part_dists[:, 0]
    edge_min_base = 0.5 * (min_nbr_u[u_idx] + min_nbr_u[v_idx])

    # Criterion 1: Normalized inefficiency utility discounted by usage
    rel_dists = tour_dists / (mean_dist + 1e-12)
    efficiency_utility = rel_dists / (1.0 + 0.5 * tour_used)
    c1 = (efficiency_utility - np.min(efficiency_utility)) / (np.ptp(efficiency_utility) + 1e-12)

    # Criterion 2: Local detour ratio via 2-hop shortcut potential across incident edges
    # Measures how sub-optimal the edge (u, v) is compared to bypassing via adjacent tour paths
    alt_prev = edge_distance[prev_idx, v_idx] + edge_distance[prev_idx, u_idx]
    alt_next = edge_distance[u_idx, next_idx] + edge_distance[v_idx, next_idx]
    best_alt = np.minimum(alt_prev, alt_next)
    detour_slack = tour_dists / (best_alt * 0.5 + 1e-12)
    c2 = (detour_slack - np.min(detour_slack)) / (np.ptp(detour_slack) + 1e-12)

    # Criterion 3: Spatial congestion (density & slackness against local k-NN scale)
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)
    pair_density = 0.5 * (norm_density[u_idx] + norm_density[v_idx])
    slackness = np.maximum(0.0, (tour_dists - edge_min_base) / (0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12))
    congestion_score = pair_density * slackness
    c3 = (congestion_score - np.min(congestion_score)) / (np.ptp(congestion_score) + 1e-12)

    # Stagnation estimation
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_dist = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    max_entropy = np.log(num_nodes) + 1e-12
    norm_entropy = shannon_entropy / max_entropy
    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    stagnation_cycle = cycling_ratio + 1.5 * high_usage_ratio + 0.5 * entropy_stagnation

    # Adaptive Chebyshev Compromise Metric: dynamically shift weight towards detour & congestion under stagnation
    alpha = np.clip(0.40 - 0.15 * np.tanh(stagnation_cycle), 0.20, 0.50)
    beta = np.clip(0.35 + 0.10 * np.tanh(stagnation_cycle), 0.25, 0.45)
    gamma = 1.0 - alpha - beta

    weighted_crit = np.vstack([alpha * c1, beta * c2, gamma * c3])
    chebyshev_metric = np.max(weighted_crit, axis=0) + 0.08 * (alpha * c1 + beta * c2 + gamma * c3)

    # Selection of top-k candidate edges
    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))
    top_k_indices = np.argpartition(chebyshev_metric, -k)[-k:]

    # Segment clustering to penalize contiguous chains of bad edges
    is_penalized = np.zeros(num_nodes, dtype=bool)
    is_penalized[top_k_indices] = True
    segment_lengths = np.zeros(num_nodes, dtype=float)
    visited = np.zeros(num_nodes, dtype=bool)
    for i in range(num_nodes):
        if is_penalized[i] and not visited[i]:
            start = i
            while is_penalized[(start - 1) % num_nodes] and not visited[(start - 1) % num_nodes]:
                start = (start - 1) % num_nodes
                if start == i:
                    break
            curr = start
            chain = []
            while is_penalized[curr] and not visited[curr]:
                visited[curr] = True
                chain.append(curr)
                curr = (curr + 1) % num_nodes
                if curr == start:
                    break
            chain_len = len(chain)
            for idx in chain:
                segment_lengths[idx] = chain_len

    # Base penalty scale modulated by stagnation and burst multiplier
    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Symmetric edge distance update
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.4 * (seg_len - 1.0)
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)
        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * segment_multiplier
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance