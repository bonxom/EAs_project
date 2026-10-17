# {Combines utility-guided edge selection with local density estimation and stagnation-aware penalty scaling tailored for large 1000-node TSP instances to escape deep local optima efficiently without high computational overhead.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Mild decay on historical usage to keep penalties adaptive over iterations
    edge_n_used *= 0.95

    tour_dist = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Guided Local Search (GLS) utility: penalize long edges that have been used less frequently
    utilities = tour_dist / (1.0 + tour_used)

    # Select top fraction of edges in tour with highest utility for penalization
    # For n=1000, 10% (100 edges) provides a strong, targeted perturbation
    k = max(1, int(np.ceil(0.10 * n)))
    top_k_indices = np.argsort(utilities)[-k:]

    sel_u = u_idx[top_k_indices]
    sel_v = v_idx[top_k_indices]

    # Update usage counts for chosen edges
    np.add.at(edge_n_used, (sel_u, sel_v), 1.0)
    np.add.at(edge_n_used, (sel_v, sel_u), 1.0)

    # Fast local scale estimation via sampled or nearest neighbor distances along tour
    avg_tour_dist = np.mean(tour_dist) + 1e-12
    rel_dist = tour_dist[top_k_indices] / avg_tour_dist

    # Stagnation factor based on average usage of selected edges
    cur_used = edge_n_used[sel_u, sel_v]
    stagnation = 1.0 + 0.5 * np.log1p(np.mean(cur_used))

    # Calculate penalty magnitude scaled by instance characteristics and edge usage
    base_lambda = 0.25 * avg_tour_dist * stagnation
    penalties = base_lambda * (1.0 + rel_dist) / (1.0 + 0.1 * cur_used)

    # Apply symmetric penalties to updated distance matrix
    np.add.at(updated_edge_distance, (sel_u, sel_v), penalties)
    np.add.at(updated_edge_distance, (sel_v, sel_u), penalties)

    return updated_edge_distance