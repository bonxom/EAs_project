import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Exponential decay of historical edge frequency
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Static k-NN local distance normalization and density estimation
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

    # Continuous generalized Jaccard overlap between current tour and historical frequency matrix
    # Current tour is an indicator set of weight 1.0 per undirected edge
    tour_overlap = np.sum(np.minimum(1.0, tour_used))
    total_history_weight = 0.5 * np.sum(edge_n_used)
    jaccard_similarity = tour_overlap / (total_history_weight + num_nodes - tour_overlap + 1e-12)

    # Entrapment severity from Jaccard overlap saturation
    entrapment_score = np.clip((jaccard_similarity - 0.25) / 0.50, 0.0, 1.5)
    stagnation_factor = np.tanh(entrapment_score * 2.0)

    # Adaptive penalty pool size
    k_frac = np.clip(0.10 + 0.14 * stagnation_factor, 0.08, 0.28)
    k = max(2, int(np.ceil(k_frac * num_nodes)))

    # Dual-criterion Pareto ranking
    rank_eff = np.argsort(np.argsort(efficiency_utility))
    rank_slack = np.argsort(np.argsort(slackness))
    pareto_score = 0.55 * rank_eff + 0.45 * rank_slack
    top_k_indices = np.argpartition(pareto_score, -k)[-k:]

    # Dynamic penalty scaling
    burst_multiplier = 1.0 + 2.5 * stagnation_factor
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Contiguous tour segment clustering calculation
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

    # Standard targeted GLS penalty modulation
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

    # Catastrophic Diversification: Coordinated perturbation on persistent cycle entrapment
    if entrapment_score > 0.8:
        mean_usage = np.mean(edge_n_used[edge_n_used > 0]) if np.any(edge_n_used > 0) else 0.0
        severe_threshold = mean_usage * 1.75
        overused_u, overused_v = np.where((edge_n_used > severe_threshold) & (np.triu(np.ones_like(edge_n_used), k=1) > 0))
        if len(overused_u) > 0:
            diversification_penalty = 0.35 * mean_dist * (entrapment_score - 0.7)
            np.add.at(updated_edge_distance, (overused_u, overused_v), diversification_penalty)
            np.add.at(updated_edge_distance, (overused_v, overused_u), diversification_penalty)
            edge_n_used[overused_u, overused_v] *= 0.85
            edge_n_used[overused_v, overused_u] *= 0.85

    return updated_edge_distance