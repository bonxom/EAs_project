#{Scales guided local search penalties for TSP size 1000 by penalizing the longest and most rigid tour edges, modulated by local k-nearest neighbor density and edge persistence while shielding short essential connections.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour, dtype=np.intp)
    v_idx = np.roll(u_idx, -1)

    # Trace tour edge distances and current usage counts
    tour_dist = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Calculate local density using approximate nearest neighbors for size 1000 efficiency
    # Using partition instead of full sort to maintain high performance for N=1000
    k_nn = 8
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    node_density = 1.0 / (np.mean(part_dists, axis=1) + 1e-12)
    norm_density = node_density / np.mean(node_density)

    # Guided local search utility: prioritize penalizing long edges with low-to-moderate prior penalties
    utilities = tour_dist / (1.0 + tour_used)

    # Target top 12% highest utility edges to break local minima cleanly at N=1000
    k_penalize = max(1, int(0.12 * num_nodes))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Update frequency memory for penalized edges
    u_sel = u_idx[top_indices]
    v_sel = v_idx[top_indices]
    edge_n_used[u_sel, v_sel] += 1.0
    edge_n_used[v_sel, u_sel] += 1.0

    # Base penalty scale derived from average tour edge distance
    mean_tour_dist = np.mean(tour_dist)

    # Edge-specific modulation factors
    # 1. Density modulation: congested areas get higher penalties to force detour across clusters
    edge_density_factor = 0.5 * (norm_density[u_sel] + norm_density[v_sel])

    # 2. Relative length factor: longer edges in the tour incur larger escape penalties
    rel_len = tour_dist[top_indices] / (mean_tour_dist + 1e-12)

    # 3. Usage persistence: progressively increase penalty if edge persists across iterations
    pers_factor = np.log1p(edge_n_used[u_sel, v_sel])

    # Compute composite penalty
    lambda_param = 0.35 * mean_tour_dist
    penalty = lambda_param * rel_len * edge_density_factor * (1.0 + 0.25 * pers_factor)

    # Apply symmetric updates to edge distances
    np.add.at(updated_edge_distance, (u_sel, v_sel), penalty)
    np.add.at(updated_edge_distance, (v_sel, u_sel), penalty)

    return updated_edge_distance