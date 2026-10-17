import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # 1. Decay historical edge usage memory
    edge_n_used *= 0.96

    # 2. Tour metrics and utility calculation
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)
    rel_dists = tour_dists / (mean_dist + 1e-12)
    utilities = rel_dists / (1.0 + 0.5 * tour_used)

    # 3. Dynamic stagnation and burst factor
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    total_tour_used = np.sum(tour_used) + 1e-12
    prob_distribution = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_distribution * np.log(prob_distribution + 1e-12))
    max_entropy = np.log(num_nodes) + 1e-12
    norm_entropy = shannon_entropy / max_entropy

    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # 4. Asymmetric bridge and bottleneck topological analysis
    k_nn = min(8, num_nodes - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    core_distance = sorted_part[:, -1] + 1e-12
    local_scale = np.mean(sorted_part, axis=1) + 1e-12
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Mutual reachability distance on tour edges: max(core_dist(u), core_dist(v), dist(u, v))
    mrd = np.maximum(np.maximum(core_distance[u_idx], core_distance[v_idx]), tour_dists)
    bridge_metric = mrd / (np.minimum(core_distance[u_idx], core_distance[v_idx]) + 1e-12)
    bridge_shield = 1.0 / (1.0 + np.exp(np.clip(bridge_metric - 2.5, -5.0, 5.0)))

    # Structural intra-cluster tension
    density_tension = np.sqrt(norm_density[u_idx] * norm_density[v_idx])
    structural_tension = (1.0 + 0.45 * density_tension) * (0.35 + 0.65 * bridge_shield)

    # 5. Apply selective edge penalties
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        current_usage = edge_n_used[u, v]
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)
        tension_multiplier = structural_tension[idx]

        penalty = base_penalty_scale * usage_factor * tension_multiplier
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance