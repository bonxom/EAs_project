#{Replace naive contiguous chain clustering with a bottleneck bridge detector that scales penalties based on the ratio of edge length to endpoint k-nearest-neighbor distances, severely penalizing inter-cluster transit edges while preserving dense intra-cluster connectivity.}

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

    # Utility calculation for candidate edge filtering
    mean_dist = np.mean(tour_dists)
    rel_dists = tour_dists / (mean_dist + 1e-12)
    utilities = rel_dists / (1.0 + 0.5 * tour_used)

    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Endpoint k-nearest neighbor scale and local density
    k_nn = min(8, num_nodes - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    knn_scales = np.mean(sorted_part, axis=1) + 1e-12
    node_density = 1.0 / knn_scales
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Global stagnation metrics
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_distribution = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_distribution * np.log(prob_distribution + 1e-12))
    norm_entropy = shannon_entropy / (np.log(num_nodes) + 1e-12)

    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Bottleneck bridge detector: ratio of edge length to average endpoint k-NN distances
    # Edges spanning across clusters have lengths substantially exceeding endpoint k-NN scales
    avg_endpoint_knn = 0.5 * (knn_scales[u_idx] + knn_scales[v_idx])
    bridge_ratio = tour_dists / avg_endpoint_knn

    # Contrast between endpoint densities reflects transition from dense to sparse or cross-cluster
    density_contrast = np.abs(norm_density[u_idx] - norm_density[v_idx]) / (norm_density[u_idx] + norm_density[v_idx] + 1e-12)

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        current_usage = edge_n_used[u, v]
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)

        # Inter-cluster bridge multiplier: strongly penalize stretched bridge edges
        b_ratio = bridge_ratio[idx]
        bridge_multiplier = 1.0 + 0.5 * np.log1p(np.maximum(0.0, b_ratio - 1.0)) + 0.3 * density_contrast[idx]

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * bridge_multiplier

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance