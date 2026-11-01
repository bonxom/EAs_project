#{Ruin the triangle skip bypass metric and recreate it as a k-NN alternative availability index based on the 3rd-to-5th nearest neighbors of endpoints to shield structurally essential cluster bridges while penalizing redundant edges in dense neighborhoods.}#

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

    # Calculate k-NN alternative availability index for endpoints of penalized edges
    # Using 3rd to 5th nearest neighbors (indices 3:6 when sorted, excluding self at 0)
    penalized_u = u_idx[top_indices]
    penalized_v = v_idx[top_indices]
    unique_nodes = np.unique(np.concatenate([penalized_u, penalized_v]))

    # Partial sort to find distances to 3rd-5th nearest neighbors efficiently
    knn_k = min(6, n - 1)
    knn_dists = np.partition(edge_distance[unique_nodes], knn_k, axis=1)[:, 3:knn_k]
    node_knn_mean = np.mean(knn_dists, axis=1) + 1e-12
    knn_lookup = dict(zip(unique_nodes, node_knn_mean))

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        d_edge = tour_dists[idx]

        # Availability ratio: compares edge length to neighborhood scale around endpoints
        avg_nbr_dist = 0.5 * (knn_lookup[u] + knn_lookup[v])
        # Higher availability ratio means the edge is significantly longer than local alternatives (penalize more);
        # low availability ratio implies edge is short relative to neighborhood or is a bridge link
        availability_ratio = d_edge / avg_nbr_dist
        norm_availability = np.clip(availability_ratio, 0.2, 4.0)

        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = d_edge / mean_tour_dist

        modulation = 0.5 + 0.35 * rel_len + 0.30 * norm_availability + 0.20 * entropy_mod
        modulation = max(0.4, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance