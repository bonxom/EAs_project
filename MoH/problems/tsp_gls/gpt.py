import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Initialize updated distance matrix from a copy of the base distance matrix
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    if n <= 1:
        return updated_edge_distance

    # Step 1: Multi-Scale Topological Edge Density Initialization
    # Compute localized k-nearest neighbor density metric for each node
    k_nn = min(8, max(1, n - 1))
    sorted_k_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1 : k_nn + 1]
    node_density = np.mean(sorted_k_dists, axis=1) + 1e-9
    avg_density = np.mean(node_density)

    # Step 2: Extract current tour edge characteristics
    u_idx = local_opt_tour
    v_idx = np.roll(local_opt_tour, -1)
    tour_edge_distances = edge_distance[u_idx, v_idx]
    tour_length = np.sum(tour_edge_distances)
    avg_edge_distance = tour_length / n

    tour_edge_penalties = np.maximum(
        edge_n_used[u_idx, v_idx], edge_n_used[v_idx, u_idx]
    )

    # Topological relative distance to detect bottleneck/inter-cluster bridge edges
    topo_dist = (2.0 * tour_edge_distances) / (node_density[u_idx] + node_density[v_idx])
    blended_cost = 0.65 * tour_edge_distances + 0.35 * avg_density * topo_dist

    # Step 3: Multi-Feature Utility Scoring
    utilities = blended_cost / (1.0 + 0.8 * tour_edge_penalties)
    max_utility = np.max(utilities)

    # Step 4: Deterministic Rank-Proportional Pareto Top-K Perturbation
    sorted_order = np.argsort(-utilities)
    k_perturb = min(max(2, n // 20), n)
    top_indices = sorted_order[:k_perturb]

    delta_penalty = np.zeros_like(edge_distance, dtype=float)
    if max_utility > 1e-9:
        for rank, idx in enumerate(top_indices):
            util_val = utilities[idx]
            if util_val >= 0.70 * max_utility:
                rank_weight = np.exp(-rank / 1.5) * (util_val / max_utility)
                u = u_idx[idx]
                v = v_idx[idx]
                delta_penalty[u, v] += rank_weight
                delta_penalty[v, u] += rank_weight

    # Step 5: Multi-Tier Memory Dynamics with Non-Linear Tabu Saturation
    symmetrical_used = np.maximum(edge_n_used, edge_n_used.T)
    evaporation_rate = 0.08
    decayed_memory = symmetrical_used * (1.0 - evaporation_rate)
    
    # Dual-tier non-linear saturation for chronic bottleneck penalization
    tabu_saturated_memory = (decayed_memory ** 0.88) + 0.35 * (decayed_memory ** 2) / (decayed_memory + 3.0)
    effective_penalties = tabu_saturated_memory + delta_penalty

    # Step 6: Scale and update edge distance matrix
    alpha = 0.28
    lambda_factor = alpha * avg_edge_distance
    updated_edge_distance += lambda_factor * effective_penalties
    np.fill_diagonal(updated_edge_distance, 0.0)

    return updated_edge_distance
