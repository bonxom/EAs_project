import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Continuous memory decay
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Adaptive local k-NN baseline scale and slackness
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

    # Global stagnation and entropy analysis
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_dist = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    max_entropy = np.log(num_nodes) + 1e-12
    norm_entropy = shannon_entropy / max_entropy

    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    stagnation_cycle = cycling_ratio + 1.5 * high_usage_ratio + 0.5 * entropy_stagnation

    # Dynamic top-k selection
    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))

    rank_eff = np.argsort(np.argsort(efficiency_utility))
    rank_slack = np.argsort(np.argsort(slackness))
    pareto_score = 0.55 * rank_eff + 0.45 * rank_slack
    top_k_indices = np.argpartition(pareto_score, -k)[-k:]

    # Topological Repulsive Potential Field (Dual-Phase Thermal Perturbation)
    thermal_phase = np.clip((stagnation_cycle - 0.35) / 1.2, 0.0, 1.0)
    node_stagnation_intensity = np.zeros(num_nodes)
    np.add.at(node_stagnation_intensity, u_idx, tour_used)
    np.add.at(node_stagnation_intensity, v_idx, tour_used)
    node_stagnation_intensity = node_stagnation_intensity / (np.max(node_stagnation_intensity) + 1e-12)

    # Identify stagnant topological core
    stagnant_core = node_stagnation_intensity > np.quantile(node_stagnation_intensity, 0.75)

    # Consecutive penalized segment clustering
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
            chain_len = float(len(chain))
            for idx in chain:
                segment_lengths[idx] = chain_len

    base_penalty_scale = 0.25 * mean_dist

    # Apply standard penalized edge updates with localized field modulation
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.4 * (seg_len - 1.0)
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)

        # Dual-phase field enhancement
        core_repulsion = 1.0 + 1.8 * thermal_phase * (node_stagnation_intensity[u] + node_stagnation_intensity[v])
        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * segment_multiplier * core_repulsion

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    # Phase-2: When thermal stagnation threshold is breached, inject localized repulsive potential
    if thermal_phase > 0.05:
        core_nodes = np.where(stagnant_core)[0]
        if len(core_nodes) > 1:
            core_k_nn = min(4, num_nodes - 1)
            core_neighbors = np.argpartition(edge_distance[core_nodes], core_k_nn, axis=1)[:, :core_k_nn]
            field_strength = 0.12 * mean_dist * thermal_phase
            for i, src in enumerate(core_nodes):
                nbrs = core_neighbors[i]
                valid_nbrs = nbrs[nbrs != src]
                updated_edge_distance[src, valid_nbrs] += field_strength * node_stagnation_intensity[src]
                updated_edge_distance[valid_nbrs, src] += field_strength * node_stagnation_intensity[src]

    return updated_edge_distance