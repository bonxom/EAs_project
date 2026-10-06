# {'idea': 'A proximity-graph topology and bottleneck-aware guided local search strategy that identifies high-betweenness bridges via Gabriel/RNG empty-region violations and k-NN ranks, synergized with k-medoids spatial clustering fragmentation, multi-scale geometric slack, and non-adjacent candidate selection to escape local optima.'}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Create an independent copy of edge distance matrix
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    if n < 4:
        return updated_edge_distance

    # Extract directed / undirected tour topology
    u_nodes = np.asarray(local_opt_tour, dtype=int)
    v_nodes = np.roll(u_nodes, -1)
    u_prev = np.roll(u_nodes, 1)
    v_next = np.roll(u_nodes, -2)

    edge_lens = edge_distance[u_nodes, v_nodes]
    edge_counts = np.maximum(edge_n_used[u_nodes, v_nodes], edge_n_used[v_nodes, u_nodes])
    tour_len = float(np.sum(edge_lens))
    avg_edge_len = tour_len / float(n)
    lambda_param = 0.35 * avg_edge_len

    # Proximity Graph Topology: Gabriel and RNG empty-region intrusion analysis
    dist_matrix_u = edge_distance[u_nodes, :]
    dist_matrix_v = edge_distance[v_nodes, :]
    lens_col = edge_lens[:, None]

    # k-NN rank of endpoints
    rank_u = np.sum(dist_matrix_u < lens_col, axis=1)
    rank_v = np.sum(dist_matrix_v < lens_col, axis=1)
    mean_rank = 0.5 * (rank_u + rank_v)
    rank_score = np.clip(mean_rank / 6.0, 0.0, 4.0)

    # Gabriel circle violation: nodes w inside diametral sphere
    d2_u = dist_matrix_u ** 2
    d2_v = dist_matrix_v ** 2
    gabriel_viol = np.sum((d2_u + d2_v) < (lens_col ** 2), axis=1)

    # Elliptical intrusion: witness nodes cutting through the edge corridor
    d_sum = dist_matrix_u + dist_matrix_v
    tight_intrusions = np.maximum(0, np.sum(d_sum < (1.12 * lens_col), axis=1) - 2)

    # Combined graph topological bridge penalty score
    topo_score = (
        0.35 * rank_score
        + 0.25 * np.clip(gabriel_viol / 3.0, 0.0, 3.0)
        + 0.40 * np.clip(tight_intrusions / 2.0, 0.0, 3.0)
    )

    # Multi-scale k-NN local stretch factor
    k_nn = min(8, n - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1 : k_nn + 1]
    knn_mean = np.mean(sorted_part, axis=1)
    local_scale = 0.5 * (knn_mean[u_nodes] + knn_mean[v_nodes]) + 1e-6
    stretch = edge_lens / local_scale
    knn_mult = 0.38 * np.clip(stretch - 1.0, 0.0, 3.5)

    # Geometric local slack and 2-opt / spike swap potentials
    d_prev_u = edge_distance[u_prev, u_nodes]
    d_u_v = edge_lens
    d_prev_v = edge_distance[u_prev, v_nodes]
    d_v_next = edge_distance[v_nodes, v_next]
    d_u_next = edge_distance[u_nodes, v_next]

    slack_u = np.maximum(0.0, d_prev_u + d_u_v - d_prev_v) / (d_prev_u + d_u_v + 1e-6)
    slack_v = np.maximum(0.0, d_u_v + d_v_next - d_u_next) / (d_u_v + d_v_next + 1e-6)
    spike_score = 0.5 * (slack_u + slack_v)

    swap_cost_diff = (d_prev_v + d_u_next) - (d_prev_u + d_v_next)
    swap_potential = np.clip(1.0 - np.maximum(0.0, swap_cost_diff) / (d_u_v + 1e-6), 0.0, 1.0)
    opt2_slack = np.clip((d_prev_u + d_v_next - (d_prev_v + d_u_next)) / (d_u_v + 1e-6), 0.0, 1.5)

    # Spatial clustering via refined k-medoids++
    num_clusters = min(n - 1, max(3, int(np.round(np.sqrt(n * 1.30)))))
    total_node_dist = np.sum(edge_distance, axis=1)
    center_0 = int(np.argmin(total_node_dist))
    centers = [center_0]
    min_dist_to_centers = edge_distance[center_0].copy()
    for _ in range(1, num_clusters):
        next_center = int(np.argmax(min_dist_to_centers))
        centers.append(next_center)
        min_dist_to_centers = np.minimum(min_dist_to_centers, edge_distance[next_center])

    dist_to_centers = edge_distance[:, centers]
    cluster_ids = np.argmin(dist_to_centers, axis=1)
    for k in range(num_clusters):
        members = np.where(cluster_ids == k)[0]
        if len(members) > 0:
            sub_dists = edge_distance[np.ix_(members, members)]
            best_member = members[int(np.argmin(np.sum(sub_dists, axis=1)))]
            centers[k] = best_member

    dist_to_centers = edge_distance[:, centers]
    cluster_ids = np.argmin(dist_to_centers, axis=1)
    cluster_sizes = np.bincount(cluster_ids, minlength=num_clusters)

    global_mean_dist = np.mean(edge_distance)
    cluster_avg_dist = np.zeros(num_clusters)
    for k in range(num_clusters):
        members = np.where(cluster_ids == k)[0]
        if len(members) > 1:
            sub_dists = edge_distance[np.ix_(members, members)]
            cluster_avg_dist[k] = np.sum(sub_dists) / (len(members) * (len(members) - 1))
        else:
            cluster_avg_dist[k] = global_mean_dist

    # Tour cluster traversal and fragmentation analysis
    tour_clusters = cluster_ids[u_nodes]
    diffs = np.where(tour_clusters != np.roll(tour_clusters, 1))[0]
    seg_lengths = np.zeros(n, dtype=int)
    num_segments = np.zeros(num_clusters, dtype=int)

    if len(diffs) == 0:
        seg_lengths[:] = n
        num_segments[tour_clusters[0]] = 1
    else:
        num_diffs = len(diffs)
        for i in range(num_diffs):
            start = diffs[i]
            end = diffs[(i + 1) % num_diffs]
            length = (end - start) % n
            if length == 0:
                length = n
            if start < end:
                seg_lengths[start:end] = length
            else:
                seg_lengths[start:] = length
                seg_lengths[:end] = length
            num_segments[tour_clusters[start]] += 1

    c_prev = cluster_ids[u_prev]
    c_u = cluster_ids[u_nodes]
    c_v = cluster_ids[v_nodes]
    c_next = cluster_ids[v_next]

    is_ping_pong = ((c_prev == c_v) & (c_u != c_v)) | ((c_u == c_next) & (c_v != c_u))
    is_cut = (c_u != c_v)

    seg_len_u = seg_lengths
    seg_len_v = np.roll(seg_lengths, -1)
    size_u = np.maximum(1, cluster_sizes[c_u])
    size_v = np.maximum(1, cluster_sizes[c_v])
    frag_u = 1.0 - (seg_len_u / size_u.astype(float))
    frag_v = 1.0 - (seg_len_v / size_v.astype(float))
    frag_score = 0.5 * (frag_u + frag_v)

    excess_crossings = np.maximum(0, num_segments[c_u] - 1) + np.maximum(0, num_segments[c_v] - 1)
    avg_cluster_d = 0.5 * (cluster_avg_dist[c_u] + cluster_avg_dist[c_v])
    dist_ratio = edge_lens / np.maximum(avg_cluster_d, 1e-6)

    cut_multipliers = np.zeros(n, dtype=float)
    cut_multipliers[is_cut] = (
        0.82
        + 0.42 * np.minimum(excess_crossings[is_cut].astype(float), 4.0)
        + 0.48 * frag_score[is_cut]
        + 0.24 * np.minimum(dist_ratio[is_cut], 4.0)
    )

    intra_mask = ~is_cut
    is_frag_cluster = (num_segments[c_u] > 1) & intra_mask
    cut_multipliers[is_frag_cluster] += (
        0.36 * frag_u[is_frag_cluster]
        + 0.18 * np.minimum(dist_ratio[is_frag_cluster], 3.0)
    )
    cut_multipliers[is_ping_pong] += 0.50

    # Base GLS utilities with combined graph topological multiplier
    base_utilities = edge_lens / (1.0 + (edge_counts ** 0.88))
    total_multipliers = (
        1.0
        + cut_multipliers
        + knn_mult
        + 0.42 * topo_score
        + 0.28 * spike_score
        + 0.20 * swap_potential
        + 0.16 * opt2_slack
    )
    augmented_utilities = base_utilities * total_multipliers

    max_utility = float(np.max(augmented_utilities))
    if max_utility <= 0:
        return updated_edge_distance

    # Diversity-aware candidate edge selection
    threshold = 0.86 * max_utility
    candidate_indices = np.where(augmented_utilities >= threshold)[0]
    cand_utils = augmented_utilities[candidate_indices]
    sorted_order = np.argsort(-cand_utils)
    sorted_candidates = candidate_indices[sorted_order]

    max_penalize = max(2, min(n // 12, 7))
    selected_indices = []
    used_vertices = set()

    for idx in sorted_candidates:
        u_idx = int(u_nodes[idx])
        v_idx = int(v_nodes[idx])
        if (u_idx not in used_vertices and v_idx not in used_vertices) or len(selected_indices) < 2 or augmented_utilities[idx] >= 0.96 * max_utility:
            selected_indices.append(idx)
            used_vertices.add(u_idx)
            used_vertices.add(v_idx)
            if len(selected_indices) >= max_penalize:
                break

    if len(selected_indices) == 0 and len(sorted_candidates) > 0:
        selected_indices = sorted_candidates[:max_penalize]

    # Apply targeted edge penalties
    for idx in selected_indices:
        u = u_nodes[idx]
        v = v_nodes[idx]
        rel_factor = augmented_utilities[idx] / max_utility
        cut_boost = 1.65 if is_cut[idx] else 1.0
        excess_boost = 1.0 + 0.15 * min(float(excess_crossings[idx]), 3.0)
        stagnation_boost = 1.0 + 0.12 * min(float(edge_counts[idx]), 4.0)
        ping_pong_boost = 1.28 if is_ping_pong[idx] else 1.0
        topo_boost = 1.0 + 0.18 * min(float(topo_score[idx]), 3.0)

        penalty = (
            lambda_param
            * cut_boost
            * excess_boost
            * stagnation_boost
            * ping_pong_boost
            * topo_boost
            * rel_factor
        )
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    np.fill_diagonal(updated_edge_distance, 0.0)
    return updated_edge_distance
