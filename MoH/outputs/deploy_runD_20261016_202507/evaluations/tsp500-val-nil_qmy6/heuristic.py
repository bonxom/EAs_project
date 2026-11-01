import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # 1. Memory decay
    edge_n_used *= 0.96

    # 2. Extract tour distances and usage frequencies
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)
    rel_dists = tour_dists / (mean_dist + 1e-12)

    # 3. Utility calculation for penalized edges
    utilities = rel_dists / (1.0 + 0.5 * tour_used)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # 4. Local scale and density estimation via k-nearest neighbors
    k_nn = min(8, num_nodes - 1)
    k_nn_indices = np.argpartition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    knn_dists = np.take_along_axis(edge_distance, k_nn_indices, axis=1)
    local_scale = np.mean(knn_dists, axis=1) + 1e-12
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # 5. Stagnation and burst dynamic scaling
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_dist = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    max_entropy = np.log(num_nodes) + 1e-12
    norm_entropy = shannon_entropy / max_entropy

    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # 6. Contiguous segment clustering of penalized edges
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

    # 7. Apply selective guided edge penalties
    penalized_nodes = set()
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0
        penalized_nodes.add(u)
        penalized_nodes.add(v)

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.4 * (seg_len - 1.0)
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)
        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * segment_multiplier

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    # 8. Bipolar reward discount: carve descent valleys on promising local k-NN shortcuts
    reward_scale = 0.08 * mean_dist * (1.0 + 0.5 * np.tanh(burst_multiplier - 1.0))
    penalized_node_arr = np.fromiter(penalized_nodes, dtype=int)
    if len(penalized_node_arr) > 0:
        candidate_neighbors = k_nn_indices[penalized_node_arr, :min(4, k_nn_indices.shape[1])]
        for i, u in enumerate(penalized_node_arr):
            for v in candidate_neighbors[i]:
                if u == v or is_penalized[u] and (v == v_idx[u] or v == u_idx[(u - 1) % num_nodes]):
                    continue
                prior_usage = edge_n_used[u, v]
                if prior_usage < 1.0:
                    discount = reward_scale * (1.0 - prior_usage)
                    updated_edge_distance[u, v] = max(1e-6, updated_edge_distance[u, v] - discount)
                    updated_edge_distance[v, u] = max(1e-6, updated_edge_distance[v, u] - discount)

    return updated_edge_distance