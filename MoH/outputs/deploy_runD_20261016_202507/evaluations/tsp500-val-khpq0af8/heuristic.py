#{A spatially non-uniform recency-frequency GLS edge penalty scheme that applies heterogeneous memory decay accelerating forgetfulness in dense clusters while preserving peripheral edges, and accumulates increments scaled by spatial eccentricity to eliminate cyclic traps.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Precompute k-NN scale and spatial topology metrics
    k_nn = min(8, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(part_dists, axis=1) + 1e-12
    min_nbr = part_dists[:, 0]

    node_density = 1.0 / local_scale
    mean_density = np.mean(node_density) + 1e-12
    norm_density = node_density / mean_density

    # Node spatial eccentricity relative to the median distance profile
    node_mean_dist = np.mean(edge_distance, axis=1)
    global_mean_dist = np.mean(node_mean_dist) + 1e-12
    eccentricity = node_mean_dist / global_mean_dist
    norm_eccentricity = eccentricity / (np.max(eccentricity) + 1e-12)

    # Spatially non-uniform memory decay: dense regions decay faster (e.g. ~0.93), peripheral slower (~0.985)
    # Form an outer-mean representation of node density to compute decay factor per node pair
    decay_vector = np.clip(0.985 - 0.05 * (norm_density / (norm_density + 1.0)), 0.92, 0.99)
    # Apply broadcasted symmetric decay matrix to edge_n_used
    decay_matrix = np.minimum.outer(decay_vector, decay_vector)
    edge_n_used *= decay_matrix

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Slackness and efficiency utility calculation
    edge_min_base = 0.5 * (min_nbr[u_idx] + min_nbr[v_idx])
    edge_local_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12
    slackness = np.maximum(0.0, (tour_dists - edge_min_base) / edge_local_scale)

    rel_dists = tour_dists / (mean_dist + 1e-12)
    efficiency_utility = rel_dists / (1.0 + 0.5 * tour_used)

    # Stagnation and entropy monitoring
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

    rank_eff = np.argsort(np.argsort(efficiency_utility))
    rank_slack = np.argsort(np.argsort(slackness))
    pareto_score = 0.55 * rank_eff + 0.45 * rank_slack
    top_k_indices = np.argpartition(pareto_score, -k)[-k:]

    # Identify contiguous penalized segment lengths
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

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Recency-Frequency-Weighted increment & penalization update
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_eccentricity = 0.5 * (norm_eccentricity[u] + norm_eccentricity[v])
        # Higher increment on peripheral/eccentric edges to firmly resist ping-pong cycling
        increment = 0.8 + 0.8 * edge_eccentricity
        edge_n_used[u, v] += increment
        edge_n_used[v, u] += increment

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.4 * (seg_len - 1.0)
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)

        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * segment_multiplier
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance