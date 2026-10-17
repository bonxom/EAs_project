#{Replace scalar average usage with an information-theoretic entropy and high-usage edge cycling detector that triggers an escalating non-linear penalty burst on persistent cycle attractors when tour stagnation and low visit entropy are detected.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decayed frequency tracking
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)
    rel_dists = tour_dists / (mean_dist + 1e-12)

    # Utility calculation balancing relative length against accumulated frequency
    utilities = rel_dists / (1.0 + 0.5 * tour_used)

    # Select top ~12% edges with highest utility to penalize
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Node density via k-nearest neighbor scale
    k_nn = min(8, num_nodes - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(sorted_part, axis=1) + 1e-12
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Information-theoretic entropy across tour edge visits
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_distribution = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_distribution * np.log(prob_distribution + 1e-12))
    max_entropy = np.log(num_nodes) + 1e-12
    norm_entropy = shannon_entropy / max_entropy

    # Detect persistent cycle attractors via heavy edge overlap / cycling ratio
    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)

    # Stagnation escalation triggered by low visit entropy and elevated cycle recurrence
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation

    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Penalize selected top-k edges with non-linear burst on persistent attractors
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]

        # Escalating non-linear penalty selectively targeting deeply trapped edge cycles
        cycle_escalation = 1.0 + 0.15 * current_usage + 0.05 * (current_usage ** 1.5)
        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * cycle_escalation

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance