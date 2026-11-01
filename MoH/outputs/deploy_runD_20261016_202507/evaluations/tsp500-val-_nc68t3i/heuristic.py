#{Replace global Shannon entropy with rolling spatial-windowed entropy and localized recurrence tracking along the tour to detect regional cycling and inject targeted hyper-penalties on trapped sub-paths while preserving global search dynamics.}#

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

    # Utilities favor long edges with relatively low past penalties
    utilities = rel_dists / (1.0 + 0.5 * tour_used)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Local density estimation via k-nearest neighbors
    k_nn = min(8, num_nodes - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(sorted_part, axis=1) + 1e-12
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Rolling windowed entropy and recurrence tracking to identify localized basin entrapment
    w_size = max(5, min(25, num_nodes // 10))
    # Pad tour_used circularly for uniform rolling window calculations
    extended_used = np.concatenate([tour_used, tour_used[:w_size - 1]])

    # Efficient rolling sum and rolling entropy using sliding window
    shape = (num_nodes, w_size)
    strides = (extended_used.strides[0], extended_used.strides[0])
    windows = np.lib.stride_tricks.as_strided(extended_used, shape=shape, strides=strides)

    w_sum = np.sum(windows, axis=1, keepdims=True) + 1e-12
    w_prob = windows / w_sum
    w_entropy = -np.sum(w_prob * np.log(w_prob + 1e-12), axis=1)
    max_w_entropy = np.log(w_size) + 1e-12
    w_norm_entropy = w_entropy / max_w_entropy

    # Localized rigidity and recurrent lock-in factors
    w_mean_usage = np.mean(windows, axis=1)
    w_cycling = np.mean(windows > 1.5, axis=1)
    local_rigidity = (1.0 - w_norm_entropy) * (1.0 + 1.5 * w_cycling) * (1.0 + 0.3 * np.log1p(w_mean_usage))

    # Global background scaling
    overall_cycling = np.mean(tour_used > 1.5)
    global_multiplier = 1.0 + 1.2 * overall_cycling
    base_penalty_scale = 0.25 * mean_dist * global_multiplier

    # Identify contiguous penalized chains to amplify co-dependent edge breaks
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

    # Apply modulated penalties
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.35 * (seg_len - 1.0)
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)

        # Local entrapment hyper-penalty amplification
        local_burst = 1.0 + 0.8 * local_rigidity[idx]

        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * segment_multiplier * local_burst
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance