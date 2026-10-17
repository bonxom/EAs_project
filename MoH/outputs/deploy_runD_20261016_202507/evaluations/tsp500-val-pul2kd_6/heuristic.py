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
    rel_dists = tour_dists / (mean_dist + 1e-12)

    # Normalized edge-length utility filtering
    utilities = rel_dists / (1.0 + 0.5 * tour_used)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Local density estimation via k-nearest neighbors
    k_nn = min(8, num_nodes - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(sorted_part, axis=1) + 1e-12
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Deterministic spatial cluster partitioning via greedy farthest-point medoids
    n_clusters = max(4, min(16, num_nodes // 30))
    medoids = [0]
    min_dists_to_medoids = np.copy(edge_distance[0])
    for _ in range(1, n_clusters):
        next_medoid = int(np.argmax(min_dists_to_medoids))
        medoids.append(next_medoid)
        min_dists_to_medoids = np.minimum(min_dists_to_medoids, edge_distance[next_medoid])

    # Assign each node to its nearest medoid cluster
    medoid_matrix = edge_distance[:, medoids]
    node_cluster = np.argmin(medoid_matrix, axis=1)

    # Compute localized edge persistence / stagnation per spatial cluster
    cluster_burst = np.ones(n_clusters, dtype=float)
    for c in range(n_clusters):
        # Identify edges whose endpoints both belong to cluster c or touch cluster c
        in_cluster = (node_cluster[u_idx] == c) | (node_cluster[v_idx] == c)
        if np.any(in_cluster):
            c_used = tour_used[in_cluster]
            c_cycling = np.mean(c_used > 1.5)
            c_high_used = np.mean(c_used > 3.0)
            c_avg_used = np.mean(c_used)
            # Higher burst multiplier exclusively applied to frozen/stagnant subregions
            cluster_burst[c] = 1.0 + 2.0 * c_cycling + 3.0 * c_high_used + 0.5 * c_avg_used

    # Edge segment continuity detection
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

    base_scale = 0.25 * mean_dist
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        c_u, c_v = node_cluster[u], node_cluster[v]
        localized_burst = 0.5 * (cluster_burst[c_u] + cluster_burst[c_v])

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.4 * (seg_len - 1.0)
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)

        penalty = base_scale * localized_burst * (1.0 + 0.35 * density_mod) * usage_factor * segment_multiplier
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance