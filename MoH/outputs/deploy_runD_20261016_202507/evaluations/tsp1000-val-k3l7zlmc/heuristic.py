#{Replaces fragile two-neighbor tour entropy with a k-nearest-neighbor usage dispersion entropy across incident geometric candidate edges, scaling penalization to target true topological entrapment over transient congestion.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency tracking
    edge_n_used *= 0.95

    # Tour metrics and GLS edge utility evaluation
    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    # Selective subset penalization: top 10%
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

    # Geometry-aware structural strain & irreplaceability
    skip_dists = edge_distance[w_idx, v_idx]
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / (skip_dists + 1e-12)
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    # K-nearest-neighbor usage dispersion entropy
    k_knn = min(8, n - 1)
    knn_idx = np.argpartition(edge_distance, k_knn, axis=1)[:, :k_knn]
    knn_usage = np.take_along_axis(edge_n_used, knn_idx, axis=1)
    knn_sum = np.sum(knn_usage, axis=1, keepdims=True) + 1e-12
    knn_probs = knn_usage / knn_sum

    # Normalized Shannon entropy across the k nearest neighbors
    node_dispersion_entropy = -np.sum(knn_probs * np.log2(np.maximum(knn_probs, 1e-12)), axis=1) / np.log2(k_knn)

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        # Low dispersion entropy signifies topological entrapment on specific preferred edges
        rigidity_mod = 1.0 - 0.5 * (node_dispersion_entropy[u] + node_dispersion_entropy[v])
        rel_len = tour_dists[idx] / mean_tour_dist

        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.25 * rigidity_mod - 0.25 * irreplaceability_mod
        modulation = max(0.4, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance