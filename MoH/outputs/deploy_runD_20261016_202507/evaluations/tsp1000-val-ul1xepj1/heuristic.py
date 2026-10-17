#{Replaces global scalar stagnation with segmental persistence and Shannon entropy-based entrapment tracking to selectively apply targeted disruption kicks to stagnant sub-chains while preserving flexible transitions.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decay edge trace history
    edge_n_used *= 0.95

    # Evaluate candidate tour edge utilities
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    # Fast O(n) utility-guided candidate selection
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Increment usage trace for penalized edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12

    # Measure node usage entropy and rigidity (lower entropy indicates entrapment in fixed paths)
    penalized_nodes = np.unique(np.concatenate((u_idx[top_indices], v_idx[top_indices])))
    sub_usage = edge_n_used[penalized_nodes, :]
    row_sums = np.sum(sub_usage, axis=1, keepdims=True) + 1e-12
    probs = sub_usage / row_sums
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    max_entropy = np.log(n) + 1e-12
    node_rigidity = np.zeros(n)
    node_rigidity[penalized_nodes] = np.clip(1.0 - (entropy / max_entropy), 0.0, 1.0)

    # Segmental persistence: consecutive edges with high recurrence in local tour
    updated_tour_used = edge_n_used[u_idx, v_idx]
    prev_tour_used = edge_n_used[w_idx, u_idx]
    next_v = np.roll(v_idx, -1)
    next_tour_used = edge_n_used[v_idx, next_v]
    chain_persistence = (prev_tour_used + 2.0 * updated_tour_used + next_tour_used) / 4.0
    chain_persistence_norm = chain_persistence / (np.mean(chain_persistence) + 1e-12)

    # Local geometric density factor
    tour_neighbor_dist = 0.5 * (edge_distance[u_idx, v_idx] + edge_distance[u_idx, w_idx]) + 1e-12
    local_density = mean_tour_dist / tour_neighbor_dist
    norm_density = local_density / (np.max(local_density) + 1e-12)

    # Apply adaptive localized kick penalties
    base_penalty_scale = 0.25 * mean_tour_dist
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        rigidity_mod = 0.5 * (node_rigidity[u] + node_rigidity[v])
        chain_kick = 1.0 + 0.8 * np.tanh(chain_persistence_norm[idx]) + 0.6 * rigidity_mod
        penalty = base_penalty_scale * chain_kick * (1.0 + 0.5 * rel_len + 0.3 * density_mod) * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance