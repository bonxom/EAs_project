# {An adaptive guided local search heuristic that replaces simple incident neighbor averaging with a 2-hop localized clustering and spatial isolation metric based on incident tour edge-length variance and spatial centrality ratios to aggressively penalize edges in congested clusters while shielding peripheral geometric transitions.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Trace history decay
    edge_n_used *= 0.95

    # Utility calculation and top-edge selection
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Update usage counts for chosen edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    updated_tour_used = edge_n_used[u_idx, v_idx]
    mean_tour_dist = np.mean(tour_dists) + 1e-12
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Incident-degree congestion and 2-hop spatial isolation metric
    prev_idx = np.roll(u_idx, 1)
    next_idx = np.roll(u_idx, -1)
    prev2_idx = np.roll(u_idx, 2)
    next2_idx = np.roll(u_idx, -2)

    # 2-hop edge lengths incident to each node along the tour
    d_prev = edge_distance[u_idx, prev_idx]
    d_next = edge_distance[u_idx, next_idx]
    d_prev2 = edge_distance[prev_idx, prev2_idx]
    d_next2 = edge_distance[next_idx, next2_idx]

    # Local incident scale and variance (dispersion captures uniform dense clusters vs isolated transition bridges)
    local_mean = 0.25 * (d_prev + d_next + d_prev2 + d_next2) + 1e-12
    var_hop = 0.25 * ((d_prev - local_mean)**2 + (d_next - local_mean)**2 + 
                      (d_prev2 - local_mean)**2 + (d_next2 - local_mean)**2)
    cv_hop = np.sqrt(var_hop) / local_mean

    # Spatial congestion: high density corresponds to small local edge lengths relative to global tour mean
    density_ratio = mean_tour_dist / local_mean
    # Congestion factor increases in tight, uniform clusters and drops for peripheral/boundary nodes with high variance
    cluster_congestion = density_ratio / (1.0 + cv_hop)

    # Normalize congestion metric across tour
    norm_congestion = cluster_congestion / (np.max(cluster_congestion) + 1e-12)

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Apply modulated penalties
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        congestion_mod = 0.5 * (norm_congestion[u] + norm_congestion[v])
        modulation = 1.0 + 0.4 * rel_len + 0.6 * congestion_mod
        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance