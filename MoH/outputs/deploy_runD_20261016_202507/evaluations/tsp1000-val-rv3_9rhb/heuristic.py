#{Ruin the myopic 2-hop bridge metric and recreate an adaptive penalty landscape governed by symmetric k-nearest-neighbor topological rank isolation, shielding geometric skeleton edges while exponentially penalizing sub-optimal high-rank connections.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decayed frequency memory
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Guided Local Search utility calculation
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.12 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Update frequency counters for top selected edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio
    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Fast k-NN rank determination for penalized edges only
    # Determine the topological rank of edge (u, v) in the local neighborhoods of u and v
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        d_uv = tour_dists[idx]

        # Rank of v with respect to u, and u with respect to v (1-based: 1 means nearest neighbor)
        rank_u = np.count_nonzero(edge_distance[u] < d_uv)
        rank_v = np.count_nonzero(edge_distance[v] < d_uv)
        sym_rank = 0.5 * (rank_u + rank_v)

        # Skeleton shield vs. high-rank escalation
        # If mutual close neighbors (rank <= 2), heavily suppress penalty to preserve structural backbone.
        # If rank > 4, escalate penalty exponentially to break unnatural long chords.
        if sym_rank <= 1.5:
            rank_mod = 0.35 + 0.25 * sym_rank
        elif sym_rank <= 4.0:
            rank_mod = 0.75 + 0.20 * (sym_rank - 1.5)
        else:
            rank_mod = 1.25 + 0.40 * np.log1p(sym_rank - 4.0)

        rel_len = d_uv / mean_tour_dist
        penalty = base_penalty_scale * rank_mod * (0.8 + 0.4 * rel_len) * edge_n_used[u, v]

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance