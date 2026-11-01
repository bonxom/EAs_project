#{Replaces static symmetric k-NN density with an anisotropic directional gradient and peripheral boundary shielding based on tour incident geometries to selectively penalize congested core-traversing edges while preserving fragile outer-hull links.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    prev_idx = np.roll(u_idx, 1)
    next_idx = np.roll(u_idx, -2)

    # Memory decay
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Utility score for edge candidate filtering
    rel_dists = tour_dists / (mean_dist + 1e-12)
    utilities = rel_dists / (1.0 + 0.5 * tour_used)

    # Select top ~12% edges with highest utility
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Ruin & Recreate: Directional density gradients and peripheral boundary shielding
    # Assess directional asymmetry and node centrality
    node_mean_dist = np.mean(edge_distance, axis=1)
    global_mean_dist = np.mean(node_mean_dist)
    centrality = global_mean_dist / (node_mean_dist + 1e-12)
    norm_centrality = centrality / (np.max(centrality) + 1e-12)

    # Incident edge lengths to detect peripheral geometry constraints (outer hull edges)
    incident_u = edge_distance[prev_idx, u_idx]
    incident_v = edge_distance[v_idx, next_idx]
    avg_incident = 0.5 * (incident_u + incident_v)

    # Directional gradient: ratio of edge length to adjacent boundary context
    # High ratio indicates bridge/crossing through interior, low ratio indicates tight boundary
    directional_ratio = tour_dists / (avg_incident + 1e-12)

    # Peripheral shield: nodes with low centrality and high relative incident lengths are peripheral
    peri_u = 1.0 - norm_centrality[u_idx]
    peri_v = 1.0 - norm_centrality[v_idx]
    peripheral_score = 0.5 * (peri_u + peri_v)
    boundary_shield = 1.0 / (1.0 + 1.2 * peripheral_score)

    # Core traversing enhancement: edges connecting central nodes in dense regions
    core_factor = 0.5 * (norm_centrality[u_idx] + norm_centrality[v_idx])
    directional_core_gradient = np.clip(directional_ratio * (0.6 + 0.8 * core_factor), 0.5, 2.5)

    # Stagnation and burst detection
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_distribution = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_distribution * np.log(prob_distribution + 1e-12))
    max_entropy = np.log(num_nodes) + 1e-12
    norm_entropy = shannon_entropy / max_entropy

    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Consecutive edge chain grouping
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

        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.4 * (seg_len - 1.0)
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)

        aniso_mod = directional_core_gradient[idx] * boundary_shield[idx]
        penalty = base_penalty_scale * aniso_mod * usage_factor * segment_multiplier

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance