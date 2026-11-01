#{Replaces monolithic linear penalties with a multi-objective slack-aware utility combining local tour edge excess, recurrence stagnation, and sub-linear square-root memory dampening to prevent over-penalizing critical structural bridges while escaping local optima in O(n) time.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Historical edge usage geometric decay
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    prev_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Local context: average length of incident tour edges for endpoints
    local_scale_u = 0.5 * (tour_dists + prev_dists) + 1e-12
    local_scale_v = np.roll(local_scale_u, -1)
    local_ref = 0.5 * (local_scale_u + local_scale_v)

    # Edge slack: relative length compared to local neighborhood context
    slack_ratio = tour_dists / local_ref

    # Utility prioritizing long/slack edges with lower relative penalty dampening
    utilities = (tour_dists * np.sqrt(slack_ratio)) / (1.0 + np.log1p(tour_used))

    # Fast selection of top 10% candidate edges in O(n)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    updated_tour_used = edge_n_used[u_idx, v_idx]
    mean_tour_dist = np.mean(tour_dists) + 1e-12

    # Cumulative stagnation factor based on recurrence of visited edges
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.25 * np.tanh(2.0 * overlap_ratio)

    base_penalty_scale = 0.22 * mean_tour_dist * stagnation_factor

    # Sub-linear dampening and multi-factor slack modulation
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        slack = slack_ratio[idx]
        # Sub-linear memory dampening via square-root prevents runaway penalty on essential bridges
        sublinear_usage = np.sqrt(edge_n_used[u, v])
        slack_mod = np.clip(slack, 0.75, 2.5)
        penalty = base_penalty_scale * slack_mod * sublinear_usage

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance