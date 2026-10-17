#{A dual-scale stagnation mechanism tracking incident star Gini concentration and spatial variance dynamically amplifies Pareto-selected inefficient edges with asymmetric exponential bursts on persistent cyclic sub-tours.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decayed frequency memory
    edge_n_used *= 0.96
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Neighborhood geometry scaling via k-NN
    k_nn = min(8, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(part_dists, axis=1) + 1e-12
    min_nbr = part_dists[:, 0]
    edge_min_base = 0.5 * (min_nbr[u_idx] + min_nbr[v_idx])
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    # Pareto edge selection utilities
    slackness = np.maximum(0.0, (tour_dists - edge_min_base) / edge_scale)
    rel_dists = tour_dists / (mean_dist + 1e-12)
    efficiency_utility = rel_dists / (1.0 + 0.5 * tour_used)

    rank_eff = np.argsort(np.argsort(efficiency_utility))
    rank_slack = np.argsort(np.argsort(slackness))
    pareto_score = 0.55 * rank_eff + 0.45 * rank_slack

    # Dual-scale stagnation detector:
    # 1. Incident star usage Gini concentration across incident edges of tour nodes
    # For each node on the tour, evaluate concentration of edge_n_used across its top-k neighbors
    top_nbr_indices = np.argpartition(edge_distance, k_nn, axis=1)[:, :k_nn]
    star_usage = np.take_along_axis(edge_n_used, top_nbr_indices, axis=1)
    sorted_star = np.sort(star_usage, axis=1)
    star_sum = np.sum(sorted_star, axis=1) + 1e-12
    k_range = np.arange(1, k_nn + 1)
    # Gini coefficient: (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
    gini_star = (2.0 * np.sum(k_range * sorted_star, axis=1) / (k_nn * star_sum)) - ((k_nn + 1.0) / k_nn)
    gini_tour = 0.5 * (gini_star[u_idx] + gini_star[v_idx])

    # 2. Tour spatial variance & cyclic stagnation
    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    mean_gini = np.mean(gini_star)

    # Adaptive candidate pool size
    stagnation_intensity = cycling_ratio + 1.5 * high_usage_ratio + 1.2 * np.maximum(0.0, mean_gini - 0.25)
    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_intensity), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))
    top_k_indices = np.argpartition(pareto_score, -k)[-k:]

    # Base penalty level
    base_penalty = 0.25 * mean_dist

    # Node density modulation
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)
    bridge_vulnerability = tour_dists / edge_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    # Asymmetric exponential localized burst on persistent cyclic sub-tours
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        curr_used = edge_n_used[u, v]
        edge_gini = gini_tour[idx]

        # Localized exponential burst activated for persistently reused edges under high star concentration
        threshold_excess = np.maximum(0.0, curr_used - 1.5)
        local_burst = np.exp(np.clip(0.45 * threshold_excess * (1.0 + edge_gini), 0.0, 3.5))

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        bridge_factor = 1.0 + 0.30 * np.tanh(norm_bridge[idx] - 1.0)

        penalty = base_penalty * (1.0 + 0.30 * density_mod) * bridge_factor * local_burst
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance