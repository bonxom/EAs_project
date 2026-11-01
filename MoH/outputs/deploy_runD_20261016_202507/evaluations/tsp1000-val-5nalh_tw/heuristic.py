#{Ruin-and-recreate edge penalty modulation by replacing noisy sub-sampled k-NN partitions with a stable geometric-mean neighborhood dispersion bottleneck ratio and tension modulation to selectively penalize non-essential bridging edges while preserving core guided local search memory.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency memory
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Guided local search utility-based selection
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

    # Stable k-NN neighborhood dispersion baseline for endpoints
    k_nn = min(5, n - 1)
    cand_nodes = np.unique(np.concatenate([u_idx[top_indices], v_idx[top_indices]]))
    cand_matrix = edge_distance[cand_nodes]
    cand_part = np.partition(cand_matrix, k_nn, axis=1)[:, 1:k_nn + 1]
    cand_knn_radii = np.mean(cand_part, axis=1) + 1e-12

    knn_map = np.zeros(n, dtype=float)
    knn_map[cand_nodes] = cand_knn_radii

    # Zigzag tension and structural entropy indicators
    skip_dists = edge_distance[w_idx, v_idx]
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / (skip_dists + 1e-12)
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Apply modulated penalties
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        e_dist = tour_dists[idx]

        # Geometric bottleneck ratio: edge length vs mutual neighborhood scale
        geom_dispersion = np.sqrt(knn_map[u] * knn_map[v])
        bottleneck_ratio = e_dist / geom_dispersion
        bottleneck_mod = np.clip(bottleneck_ratio - 1.0, -0.5, 2.0)

        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = e_dist / mean_tour_dist

        modulation = 1.0 + 0.25 * rel_len + 0.20 * strain_mod + 0.15 * entropy_mod + 0.30 * bottleneck_mod
        modulation = max(0.35, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance