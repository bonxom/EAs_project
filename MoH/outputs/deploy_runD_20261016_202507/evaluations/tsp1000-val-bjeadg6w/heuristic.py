#{Replaces the 2-edge incident tour entropy with a k-nearest neighbor usage dispersion entropy evaluated over the top Euclidean neighbors of candidate endpoints, modulating edge penalties to prioritize escape from flexible dense clusters while protecting geometric bottlenecks.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    # Compute k-nearest neighbor usage dispersion entropy for unique candidate endpoints
    unique_candidate_nodes = np.unique(np.concatenate((u_idx[top_indices], v_idx[top_indices])))
    k_nn = min(10, n - 1)
    # Get indices of k nearest neighbors (excluding self at distance 0)
    knn_indices = np.argpartition(edge_distance[unique_candidate_nodes], kth=k_nn, axis=1)[:, 1:k_nn + 1]

    node_knn_usage = edge_n_used[unique_candidate_nodes[:, None], knn_indices]
    row_sums = np.sum(node_knn_usage, axis=1, keepdims=True)
    zero_mask = (row_sums.squeeze(-1) == 0)

    probs = node_knn_usage / (row_sums + 1e-12)
    log_probs = np.log2(np.maximum(probs, 1e-12))
    knn_entropy_vals = -np.sum(probs * log_probs, axis=1) / np.log2(k_nn)
    knn_entropy_vals[zero_mask] = 1.0

    node_dispersion = np.zeros(n)
    node_dispersion[unique_candidate_nodes] = knn_entropy_vals

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        entropy_dispersion_mod = 0.5 * (node_dispersion[u] + node_dispersion[v])
        rel_len = tour_dists[idx] / mean_tour_dist

        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.20 * entropy_dispersion_mod - 0.25 * irreplaceability_mod
        modulation = max(0.4, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance