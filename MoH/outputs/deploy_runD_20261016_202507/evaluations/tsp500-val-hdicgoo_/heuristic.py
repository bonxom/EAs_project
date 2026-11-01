#{Dynamically detects topological bottlenecks and cluster-internal redundancy using the harmonic mean of 3-nearest-neighbor distances to aggressively penalize redundant intra-cluster edges while shielding essential inter-cluster bridges.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Core decayed frequency tracking
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Utility calculation balancing edge distance against usage history
    rel_dists = tour_dists / (mean_dist + 1e-12)
    utilities = rel_dists / (1.0 + 0.5 * tour_used)

    # Filter top-k candidate edges to penalize (~12%)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Compute 3-NN local scale efficiently without full matrix sorting
    k_nn = min(4, num_nodes)
    part_dists = np.partition(edge_distance, k_nn - 1, axis=1)[:, 1:k_nn]
    local_scales = np.mean(part_dists, axis=1) + 1e-12

    # Dynamic bottleneck vs intra-cluster hull detection via harmonic mean scale
    avg_usage = np.mean(tour_used)
    stagnation = 1.0 + np.tanh(avg_usage * 0.15)
    base_penalty_scale = 0.25 * mean_dist * stagnation

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        d_uv = tour_dists[idx]

        # Harmonic mean of local neighborhood scales around u and v
        scale_u = local_scales[u]
        scale_v = local_scales[v]
        harmonic_scale = 2.0 * scale_u * scale_v / (scale_u + scale_v + 1e-12)

        # Relative bridge ratio: > 1 indicates a long link / bridge between clusters
        bridge_ratio = d_uv / harmonic_scale

        # Shield inter-cluster bridges (low penalty) and heavily penalize dense internal edges (high penalty)
        topology_mod = 1.0 / (1.0 + np.exp(2.5 * (bridge_ratio - 1.2)))
        redundancy_multiplier = 0.4 + 1.6 * topology_mod

        # Register edge usage and apply modulated penalty symmetrically
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        penalty = base_penalty_scale * redundancy_multiplier * (1.0 + 0.12 * edge_n_used[u, v])
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance