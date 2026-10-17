#{Adaptive ruin-and-diversify Guided Local Search dynamically expands candidate selection cardinality with search stagnation and uses a deterministic stride-interleaved utility pool to penalize diverse high-utility edges without cycling identical subsets.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Trace history decay
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    overlap_ratio = np.mean(tour_used > 1.0)
    stagnation_factor = 1.0 + 2.0 * overlap_ratio

    # Dynamically modulate k_penalize: expand ruin cardinality from ~8% up to 20% under stagnation
    k_fraction = np.clip(0.08 + 0.08 * overlap_ratio, 0.08, 0.20)
    k_penalize = max(2, int(np.ceil(k_fraction * n)))

    # Over-select candidate pool of size 2 * k_penalize
    pool_size = min(n, 2 * k_penalize)
    top_pool_indices = np.argpartition(utilities, -pool_size)[-pool_size:]

    # Deterministic rank-biased dispersion: sort pool descending by utility
    pool_sorted_indices = top_pool_indices[np.argsort(-utilities[top_pool_indices])]

    # Interleaved deterministic selection to break cyclic lock-in
    first_tier_k = k_penalize // 2
    second_tier_k = k_penalize - first_tier_k
    top_indices = np.concatenate((
        pool_sorted_indices[:first_tier_k],
        pool_sorted_indices[first_tier_k : first_tier_k + 2 * second_tier_k : 2]
    ))

    # Update usage counts
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    # Local geometry density estimation
    w_idx = np.roll(u_idx, 1)
    tour_neighbor_dist = 0.5 * (edge_distance[u_idx, v_idx] + edge_distance[u_idx, w_idx]) + 1e-12
    local_density = mean_tour_dist / tour_neighbor_dist
    norm_density = local_density / (np.max(local_density) + 1e-12)

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Apply modulated edge penalties symmetrically
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        penalty = base_penalty_scale * (1.0 + 0.5 * rel_len + 0.5 * density_mod) * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance