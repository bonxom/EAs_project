#{Adaptive guided local search that penalizes high-utility tour edges scaled by stagnation burst dynamics, endpoint-relative edge stretch against local 5-NN neighborhoods, and global spatial eccentricity to dismantle dense cluster shortcuts while shielding boundary links.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decaying frequency memory
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)
    rel_dists = tour_dists / (mean_dist + 1e-12)

    # Utility calculation: edge length vs historical usage
    utilities = rel_dists / (1.0 + 0.5 * tour_used)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Dual-scale geometric analysis: Local 5-NN scale
    k_nn = min(6, num_nodes - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(sorted_part, axis=1) + 1e-12

    # Global spatial eccentricity: relative centrality of nodes
    global_node_mean = np.mean(edge_distance, axis=1)
    global_mean = np.mean(global_node_mean) + 1e-12
    # Nodes with small mean distance to all nodes are central (high centrality)
    centrality = global_mean / (global_node_mean + 1e-12)
    norm_centrality = centrality / (np.mean(centrality) + 1e-12)

    # Stagnation & burst multipliers
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

    # Apply penalties to selected top-k edges
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        # Edge-relative stretch: tour edge length relative to endpoints' local 5-NN scale
        endpoint_scale = 0.5 * (local_scale[u] + local_scale[v])
        stretch = tour_dists[idx] / endpoint_scale
        stretch_factor = np.clip(stretch, 0.5, 3.0)

        # Eccentricity / centrality modifier: penalize cluster interiors, shield boundaries
        centrality_mod = 0.5 * (norm_centrality[u] + norm_centrality[v])

        current_usage = edge_n_used[u, v]
        cycle_escalation = 1.0 + 0.15 * current_usage + 0.05 * (current_usage ** 1.5)

        # Combined penalty modulation
        penalty = base_penalty_scale * (0.6 * stretch_factor + 0.4 * centrality_mod) * cycle_escalation

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance