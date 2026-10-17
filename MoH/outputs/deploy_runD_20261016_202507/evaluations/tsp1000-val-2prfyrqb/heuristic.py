#{Replaces myopic single-step chord checks with a k-nearest alternative detouring metric that assesses edge redundancy against each endpoint's best local neighbors, penalizing redundant edges with viable short detours while shielding structural bridge edges.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency tracking
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # GLS edge utility evaluation prioritizing long, low-frequency edges
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

    # Calculate k-nearest alternative detouring penalty for nodes involved in top_indices
    # Identify unique nodes in the penalized subset to minimize partition overhead on size 1000
    penalized_nodes = np.unique(np.concatenate((u_idx[top_indices], v_idx[top_indices])))

    # Extract small k-nearest neighbor distances for alternative connections (e.g., top 4 nearest neighbors)
    k_nn = min(5, n - 1)
    # np.partition across the rows of penalized nodes to extract non-self, alternative local edge costs
    sub_dists = edge_distance[penalized_nodes, :]
    part_dists = np.partition(sub_dists, k_nn, axis=1)[:, :k_nn + 1]
    sorted_part = np.sort(part_dists, axis=1)
    # The 0th entry is the diagonal (0.0 distance to self), entries 1:k_nn are the closest neighbors
    alt_nn_dist = np.mean(sorted_part[:, 1:], axis=1)
    node_to_alt = dict(zip(penalized_nodes, alt_nn_dist))

    # Incident edge entropy
    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        d_uv = tour_dists[idx]

        # Alternative detour availability: contrast edge length with the average k-nearest alternatives
        best_alt_u = node_to_alt[u]
        best_alt_v = node_to_alt[v]
        mean_alt = 0.5 * (best_alt_u + best_alt_v)

        # Detour redundancy: if edge is much longer than local alternatives, it is redundant and easily bypassed.
        # If edge is shorter or close to local alternatives, it is a structural bridge/bottleneck and penalized less.
        detour_redundancy = (d_uv - mean_alt) / (d_uv + mean_alt + 1e-12)

        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = d_uv / mean_tour_dist

        # Combined modulation: amplify penalties on redundant edges with abundant local alternatives
        modulation = 1.0 + 0.35 * rel_len + 0.30 * detour_redundancy + 0.20 * entropy_mod
        modulation = max(0.4, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance