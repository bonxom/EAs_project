import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decayed frequency memory
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Local neighborhood geometry via k-NN partitioning
    k_nn = min(12, num_nodes - 1)
    knn_indices = np.argpartition(edge_distance, k_nn, axis=1)[:, 1:k_nn]

    # Extract distances of k-NN for local scale normalization
    knn_rows = np.arange(num_nodes)[:, None]
    knn_dists = edge_distance[knn_rows, knn_indices]
    local_scale = np.mean(knn_dists, axis=1) + 1e-12
    min_nbr_u = np.min(knn_dists, axis=1)

    edge_min_base = 0.5 * (min_nbr_u[u_idx] + min_nbr_u[v_idx])
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    # Relative slackness and efficiency utility
    slackness = np.maximum(0.0, (tour_dists - edge_min_base) / edge_scale)
    rel_dists = tour_dists / (mean_dist + 1e-12)
    efficiency_utility = rel_dists / (1.0 + 0.5 * tour_used)

    # Density computation
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Stagnation & Shannon entropy analysis
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_dist = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    norm_entropy = shannon_entropy / (np.log(num_nodes) + 1e-12)
    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    stagnation_cycle = cycling_ratio + 1.5 * high_usage_ratio + 0.5 * entropy_stagnation

    # Dynamic Pareto top-k selection
    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))

    rank_eff = np.argsort(np.argsort(efficiency_utility))
    rank_slack = np.argsort(np.argsort(slackness))
    pareto_score = 0.55 * rank_eff + 0.45 * rank_slack
    top_k_indices = np.argpartition(pareto_score, -k)[-k:]

    # Ruin & Recreate: 2D topological graph-cut / bottleneck estimator via Jaccard overlap
    u_top = u_idx[top_k_indices]
    v_top = v_idx[top_k_indices]

    # Vectorized mutual k-NN intersection computation for selected edges
    u_neighbors = knn_indices[u_top]  # Shape: (k, k_nn - 1)
    v_neighbors = knn_indices[v_top]  # Shape: (k, k_nn - 1)

    # Count overlapping neighbor IDs between u and v
    matches = (u_neighbors[:, :, None] == v_neighbors[:, None, :])
    intersection = np.sum(matches, axis=(1, 2))
    num_knn_items = k_nn - 1
    union = 2 * num_knn_items - intersection
    jaccard_overlap = intersection / (union + 1e-12)

    # Cross-cluster bottleneck multiplier: low Jaccard implies bridging edge
    # Strongly penalize inefficient cross-cluster edges while protecting dense natural intra-clusters
    bottleneck_multiplier = 1.0 + 0.8 * (1.0 - jaccard_overlap)

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    bridge_vulnerability = tour_dists / edge_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    for i, idx in enumerate(top_k_indices):
        u, v = u_top[i], v_top[i]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]

        topo_factor = bottleneck_multiplier[i]
        bridge_factor = 1.0 + 0.30 * np.tanh(norm_bridge[idx] - 1.0)
        usage_factor = 1.0 + 0.20 * np.sqrt(current_usage)

        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * topo_factor * bridge_factor
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance