#{Ruin linear usage multiplication in favor of a sub-linear diminishing-returns saturation curve modulated by local k-NN density normalization and geometric tension to prevent runaway edge distortion while escaping local optima.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency memory
    edge_n_used *= 0.95

    # Tour metrics and candidate utility ranking
    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    # Local spatial scale via nearest neighbors (efficient top-k partition across matrix)
    k_nn = min(8, max(2, n - 1))
    partitioned_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(partitioned_dists, axis=1) + 1e-12

    # Global and stagnation characteristics
    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.2 * overlap_ratio

    # Geometric tension & irreplaceability modulation
    skip_dists = edge_distance[w_idx, v_idx]
    zigzag_tension = (pred_dists + tour_dists) / (skip_dists + 1e-12)
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)
    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        # Diminishing returns saturation curve replaces unbounded linear scaling
        usage = edge_n_used[u, v]
        usage_saturation = np.log1p(usage)

        # Local density scale normalization
        edge_local_scale = 0.5 * (local_scale[u] + local_scale[v])
        base_scale = 0.40 * edge_local_scale * stagnation_factor

        # Geometric & structural modulation
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irrep_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist

        modulation = 1.0 + 0.30 * rel_len + 0.20 * strain_mod - 0.25 * irrep_mod
        modulation = max(0.35, modulation)

        penalty = base_scale * usage_saturation * modulation
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance