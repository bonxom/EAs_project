# {'idea': 'Identify persistent topological crossings and near-miss 2-opt strain alongside Voronoi cluster zig-zags, dynamically unblocking historically penalized crossing edges in Guided Local Search utility to aggressively penalize structural bottlenecks.'}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Create an explicit copy of edge distance matrix
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    if n < 4:
        return updated_edge_distance

    u_nodes = local_opt_tour
    v_nodes = np.roll(local_opt_tour, -1)
    edge_lens = edge_distance[u_nodes, v_nodes]
    tour_len = float(np.sum(edge_lens))
    avg_len = tour_len / float(n)
    safe_avg_len = max(avg_len, 1e-6)

    # 1. Vectorized 2-opt topological crossing and strain evaluation
    l_mat = edge_lens[:, None] + edge_lens[None, :]
    d_uu = edge_distance[np.ix_(u_nodes, u_nodes)]
    d_vv = edge_distance[np.ix_(v_nodes, v_nodes)]
    gain_2opt = l_mat - d_uu - d_vv

    # Mask self and adjacent tour edges
    valid_mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid_mask, False)
    diag_indices = np.arange(n - 1)
    valid_mask[diag_indices, diag_indices + 1] = False
    valid_mask[diag_indices + 1, diag_indices] = False
    valid_mask[0, n - 1] = False
    valid_mask[n - 1, 0] = False

    masked_gain = np.where(valid_mask, gain_2opt, -1e9)
    max_gain = np.max(masked_gain, axis=1)
    pos_gain = np.where(valid_mask & (gain_2opt > 0.0), gain_2opt, 0.0)
    sum_pos_gain = np.sum(pos_gain, axis=1)

    norm_gain = np.clip(gain_2opt / safe_avg_len, -8.0, 0.0)
    strain_matrix = np.exp(norm_gain) * valid_mask
    strain_score = np.sum(strain_matrix, axis=1) / float(max(1, n - 3))

    # 2. Spatial Voronoi clustering using farthest-point medoids
    num_clusters = min(8, max(2, int(np.sqrt(n))))
    total_node_dist = np.sum(edge_distance, axis=1)
    center_0 = int(np.argmin(total_node_dist))
    centers = [center_0]
    min_dist_to_centers = edge_distance[center_0].copy()
    for _ in range(1, num_clusters):
        next_center = int(np.argmax(min_dist_to_centers))
        centers.append(next_center)
        min_dist_to_centers = np.minimum(
            min_dist_to_centers, edge_distance[next_center]
        )

    dist_to_centers = edge_distance[:, centers]
    cluster_ids = np.argmin(dist_to_centers, axis=1)

    # Refine medoids to geometric intra-cluster centroids
    for k in range(num_clusters):
        members = np.where(cluster_ids == k)[0]
        if len(members) > 0:
            sub_dists = edge_distance[np.ix_(members, members)]
            best_member = members[int(np.argmin(np.sum(sub_dists, axis=1)))]
            centers[k] = best_member

    dist_to_centers = edge_distance[:, centers]
    cluster_ids = np.argmin(dist_to_centers, axis=1)

    cluster_avg_dist = np.zeros(num_clusters)
    global_mean_dist = float(np.mean(edge_distance))
    for k in range(num_clusters):
        members = np.where(cluster_ids == k)[0]
        if len(members) > 1:
            sub_dists = edge_distance[np.ix_(members, members)]
            cluster_avg_dist[k] = np.sum(sub_dists) / float(
                len(members) * (len(members) - 1)
            )
        else:
            cluster_avg_dist[k] = global_mean_dist

    # Detect inter-cluster zig-zag transitions
    tour_clusters = cluster_ids[local_opt_tour]
    runs = [tour_clusters[0]]
    for c in tour_clusters[1:]:
        if c != runs[-1]:
            runs.append(c)
    if len(runs) > 1 and runs[0] == runs[-1]:
        runs.pop()

    num_segments = np.zeros(num_clusters, dtype=int)
    for c in runs:
        num_segments[c] += 1

    # 3. Local neighborhood scale estimation (k-NN mean distance)
    sorted_dists = np.sort(edge_distance, axis=1)
    k_val = min(4, n - 1)
    knn_mean = np.mean(sorted_dists[:, 1:k_val + 1], axis=1)

    # 4. Compute augmented utility components
    edge_counts = edge_n_used[u_nodes, v_nodes].astype(float)
    # Unblock historically used edges if they are persistently crossing
    unblock_factor = 1.0 + 1.5 * np.minimum(
        np.maximum(0.0, max_gain) / safe_avg_len, 2.0
    )
    effective_counts = edge_counts / unblock_factor
    base_utilities = edge_lens / (1.0 + effective_counts)

    crossing_multipliers = np.ones(n, dtype=float)
    cluster_multipliers = np.ones(n, dtype=float)

    for i in range(n):
        mg = max_gain[i]
        if mg > 0.0:
            mg_norm = mg / safe_avg_len
            sg_norm = sum_pos_gain[i] / safe_avg_len
            crossing_multipliers[i] = (
                1.0 + 1.2 * min(mg_norm, 3.0) + 0.4 * min(sg_norm, 4.0)
            )
        else:
            crossing_multipliers[i] = 1.0 + 0.35 * min(strain_score[i], 2.0)

        c_u = cluster_ids[u_nodes[i]]
        c_v = cluster_ids[v_nodes[i]]
        if c_u != c_v:
            excess_crossings = max(0, num_segments[c_u] - 1) + max(
                0, num_segments[c_v] - 1
            )
            avg_local_dist = 0.5 * (cluster_avg_dist[c_u] + cluster_avg_dist[c_v])
            dist_ratio = edge_lens[i] / max(avg_local_dist, 1e-6)
            cluster_multipliers[i] = (
                1.0
                + 0.8
                + 0.35 * min(float(excess_crossings), 4.0)
                + 0.25 * min(dist_ratio, 4.0)
            )
        else:
            intra_ratio = edge_lens[i] / max(cluster_avg_dist[c_u], 1e-6)
            if intra_ratio > 1.3:
                cluster_multipliers[i] = 1.0 + 0.25 * min(intra_ratio - 1.0, 3.0)

    local_densities = 0.5 * (knn_mean[u_nodes] + knn_mean[v_nodes])
    knn_ratios = edge_lens / np.maximum(local_densities, 1e-6)
    knn_multipliers = 1.0 + 0.15 * np.maximum(0.0, np.minimum(knn_ratios - 1.5, 3.0))

    augmented_utilities = (
        base_utilities
        * crossing_multipliers
        * cluster_multipliers
        * knn_multipliers
    )

    # 5. Apply guided penalties to highest utility edges
    max_utility = float(np.max(augmented_utilities))
    selected_indices = np.where(augmented_utilities >= 0.99 * max_utility)[0]
    if len(selected_indices) == 0:
        selected_indices = np.array([int(np.argmax(augmented_utilities))])

    lambda_param = 0.3 * safe_avg_len

    for idx in selected_indices:
        u_idx = u_nodes[idx]
        v_idx = v_nodes[idx]
        pen_weight = 1.0
        if cluster_ids[u_idx] != cluster_ids[v_idx]:
            pen_weight *= 1.3
        if max_gain[idx] > 0.0:
            pen_weight *= 1.0 + 0.4 * min(max_gain[idx] / safe_avg_len, 2.0)

        penalty = lambda_param * pen_weight
        updated_edge_distance[u_idx, v_idx] += penalty
        updated_edge_distance[v_idx, u_idx] += penalty

    return updated_edge_distance
