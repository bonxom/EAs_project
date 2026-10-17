#{Replaces the crude tour-adjacent 1D density proxy with a geometric triangle-inequality slack metric combined with historical vertex incident penalty load to capture local clustering rigidity in O(n) time, selectively penalizing congested interior cluster edges while protecting peripheral bridges.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    prev_idx = np.roll(u_idx, 1)

    # Maintain historical edge usage decay
    edge_n_used *= 0.95

    # Linear-time tour edge retrieval and utility calculation
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    # Select top-k candidate edges for perturbation
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Register usage updates for targeted edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Spatial-topological metric: compute triangle-inequality slack of consecutive tour triplets
    # Dense cluster nodes have small chord distances (prev->next), while peripheral bridge nodes have large chords
    chord_dists = edge_distance[prev_idx, v_idx]
    sum_arms = edge_distance[prev_idx, u_idx] + tour_dists
    # Triangle ratio: near 1.0 for collinear/stretched peripheral transitions, lower for compact cluster folds
    triplet_compactness = 1.0 - (chord_dists / (sum_arms + 1e-12))
    norm_compactness = (triplet_compactness - np.min(triplet_compactness)) / (
        np.ptp(triplet_compactness) + 1e-12
    )

    # Vertex incident penalty load: aggregate historical usage along tour boundaries
    incident_load = updated_tour_used + np.roll(updated_tour_used, 1)
    norm_load = incident_load / (np.max(incident_load) + 1e-12)

    # Local clustering rigidity: high in congested cluster interiors, low along peripheral bridges
    rigidity = 0.5 * norm_compactness + 0.5 * norm_load

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Apply modulated penalties strictly in O(k) time
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        rigidity_mod = 0.5 * (rigidity[idx] + rigidity[(idx + 1) % n])
        penalty = base_penalty_scale * (1.0 + 0.4 * rel_len + 0.6 * rigidity_mod) * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance