#{Vectorizes edge penalty updates by combining geometric strain and local k-NN density into a smooth sigmoid-modulated multiplicative temperature scale, applying symmetric updates via np.add.at without iterative loops.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency memory
    edge_n_used *= 0.95

    # Tour edge metrics
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Guided local search utility scoring and top-k selection
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.12 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Increment frequency usage on candidate edges
    top_u = u_idx[top_indices]
    top_v = v_idx[top_indices]
    edge_n_used[top_u, top_v] += 1.0
    edge_n_used[top_v, top_u] += 1.0

    # Geometric strain calculation
    mean_tour_dist = np.mean(tour_dists) + 1e-12
    skip_dists = edge_distance[w_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    geometric_strain = (pred_dists + tour_dists) / (skip_dists + 1e-12)
    norm_strain = geometric_strain / (np.mean(geometric_strain) + 1e-12)

    # Local density estimation using small k-NN neighborhood
    k_nn = min(8, max(2, n - 1))
    knn_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn + 1]
    knn_mean = np.mean(knn_part, axis=1) + 1e-12
    node_density = 1.0 / knn_mean
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Adaptive stagnation and search temperature
    updated_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Multiplicative synergy modulation via sigmoid activation
    top_strain = 0.5 * (norm_strain[top_indices] + norm_strain[(top_indices + 1) % n])
    top_density = 0.5 * (norm_density[top_u] + norm_density[top_v])
    rel_len = tour_dists[top_indices] / mean_tour_dist

    synergy_score = 1.2 * rel_len + 0.8 * top_strain + 0.5 * top_density - 2.0
    sigmoid_mod = 0.5 + 1.5 / (1.0 + np.exp(-synergy_score))

    base_penalty = 0.25 * mean_tour_dist * stagnation_factor
    penalties = base_penalty * sigmoid_mod * edge_n_used[top_u, top_v]

    # Vectorized symmetric penalty application
    np.add.at(updated_edge_distance, (top_u, top_v), penalties)
    np.add.at(updated_edge_distance, (top_v, top_u), penalties)

    return updated_edge_distance