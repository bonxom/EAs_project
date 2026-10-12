#{This algorithm incorporates a recency-decay frequency memory mechanism that compounds penalties exponentially for chronically revisited bottleneck edges while protecting neglected edges through guided local search utility scoring and propagation.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    if n < 3:
        return updated_edge_distance

    tour = np.asarray(local_opt_tour, dtype=int)
    u_idx = tour
    v_idx = np.roll(tour, -1)

    costs = edge_distance[u_idx, v_idx].astype(float)
    penalties = edge_n_used[u_idx, v_idx].astype(float)

    # Calculate baseline GLS utilities over contextually smoothed edge costs
    adj_weights = (np.roll(costs, 1) + 2.0 * costs + np.roll(costs, -1)) / 4.0
    utilities = adj_weights / (1.0 + penalties)

    max_util = np.max(utilities)
    mean_penalty = np.mean(penalties)
    stagnation = 1.0 - np.exp(-mean_penalty / (np.sqrt(n) + 1e-6))
    threshold = max_util * (1.0 - 0.15 * stagnation)

    selected_indices = np.where(utilities >= (threshold - 1e-8))[0]
    avg_tour_cost = np.mean(costs) if np.mean(costs) > 0 else 1.0

    # Frequency-to-cost ratio metrics to detect chronic trapping
    safe_costs = np.where(costs > 1e-6, costs, 1e-6)
    penalty_cost_ratios = penalties / safe_costs
    adaptive_tabu_ratio = np.mean(penalty_cost_ratios) + np.std(penalty_cost_ratios)

    edge_increments = {}
    for idx in selected_indices:
        e_curr = (min(u_idx[idx], v_idx[idx]), max(u_idx[idx], v_idx[idx]))

        # Exponential compounding factor based on visit frequency relative to tour mean
        freq_factor = penalties[idx] / (mean_penalty + 1.0)
        compound_mult = np.power(1.25, np.clip(freq_factor, 0.0, 5.0))

        inc_curr = (costs[idx] / avg_tour_cost) * compound_mult
        edge_increments[e_curr] = max(edge_increments.get(e_curr, 0.0), inc_curr)

        # Attenuated propagation to adjacent edges if not chronically over-penalized
        for offset in [-1, 1]:
            adj_idx = (idx + offset) % n
            if penalty_cost_ratios[adj_idx] <= adaptive_tabu_ratio:
                e_adj = (min(u_idx[adj_idx], v_idx[adj_idx]), max(u_idx[adj_idx], v_idx[adj_idx]))
                adj_freq_factor = penalties[adj_idx] / (mean_penalty + 1.0)
                adj_compound = np.power(1.15, np.clip(adj_freq_factor, 0.0, 4.0))
                inc_adj = 0.35 * (costs[adj_idx] / avg_tour_cost) * adj_compound
                edge_increments[e_adj] = max(edge_increments.get(e_adj, 0.0), inc_adj)

    for (u, v), inc in edge_increments.items():
        updated_edge_distance[u, v] += inc
        updated_edge_distance[v, u] += inc
        edge_n_used[u, v] += 1
        edge_n_used[v, u] += 1

    return updated_edge_distance
