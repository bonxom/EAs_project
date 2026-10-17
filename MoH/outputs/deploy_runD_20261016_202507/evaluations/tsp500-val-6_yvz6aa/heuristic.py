import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decaying frequency memory
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)
    rel_dists = tour_dists / (mean_dist + 1e-12)

    # Candidate selection via utility ratio
    utilities = rel_dists / (1.0 + 0.5 * tour_used)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Local spatial geometry via k-NN density
    k_nn = min(8, num_nodes - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(sorted_part, axis=1) + 1e-12
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # 2D Graph-level incident edge usage dispersion & node-level Gini concentration
    node_total_usage = np.sum(edge_n_used, axis=1)
    active_nodes = node_total_usage > 1e-6

    # Node-incident Gini impurity (1 - sum(p_e^2)): low values mean usage is locked to 1-2 edges (rigid cycle hub)
    # High values mean usage is widely distributed across many incident edges (healthy exploration)
    sum_sq_usage = np.sum(edge_n_used ** 2, axis=1)
    herfindahl_index = np.zeros(num_nodes)
    herfindahl_index[active_nodes] = sum_sq_usage[active_nodes] / (node_total_usage[active_nodes] ** 2 + 1e-12)

    # Node rigidity is high when incident edges are repeatedly reused in tight lock-in
    node_rigidity = herfindahl_index
    norm_rigidity = node_rigidity / (np.max(node_rigidity) + 1e-12)

    # Global stagnation based on graph-level lock-in across the current tour
    tour_rigidity = np.mean(node_rigidity[u_idx])
    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 1.2 * tour_rigidity
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Apply modulated penalties
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        hub_rigidity_mod = 0.5 * (norm_rigidity[u] + norm_rigidity[v])
        current_usage = edge_n_used[u, v]
        cycle_escalation = 1.0 + 0.15 * current_usage + 0.05 * (current_usage ** 1.5)

        penalty = base_penalty_scale * (1.0 + 0.30 * density_mod + 0.40 * hub_rigidity_mod) * cycle_escalation
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance