import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Core logic: Decayed frequency tracking
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Core logic: k-NN baseline regret utility evaluation
    k_nn = min(12, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, :k_nn]
    sorted_part = np.sort(part_dists, axis=1)
    k_alt = min(5, k_nn)
    alt_pool = sorted_part[:, 1:k_alt]
    harmonic_alt = (k_alt - 1) / np.sum(1.0 / (alt_pool + 1e-12), axis=1)

    local_scale = np.mean(sorted_part[:, 1:min(8, k_nn)], axis=1) + 1e-12
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    rank_u = np.sum(part_dists[u_idx, :] < tour_dists[:, None], axis=1)
    rank_v = np.sum(part_dists[v_idx, :] < tour_dists[:, None], axis=1)
    mean_rank = 0.5 * (rank_u + rank_v)
    rank_penalty_factor = 1.0 + np.tanh(mean_rank / 2.0)

    baseline_alt = 0.5 * (harmonic_alt[u_idx] + harmonic_alt[v_idx])
    edge_regret = np.maximum(0.0, tour_dists - baseline_alt)
    norm_regret = edge_regret / edge_scale

    regret_utility = ((norm_regret * rank_penalty_factor) + (tour_dists / (mean_dist + 1e-12))) / (1.0 + 0.45 * tour_used)

    # Density profile
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Recreated: Gini impurity drift and spectral projection of visitation distribution
    sum_usage = np.sum(tour_used) + 1e-12
    p_usage = tour_used / sum_usage
    # Gini impurity of tour edge usage: 0 when single edge dominates, 1 - 1/N when perfectly uniform
    gini_impurity = 1.0 - np.sum(p_usage ** 2)
    # Stagnation/crystallization index: high when usage collapses into deterministic rigid cycles (low Gini)
    drift_rigidity = np.clip(1.0 - gini_impurity * (num_nodes / (num_nodes - 1.0 + 1e-12)), 0.0, 1.0)

    # Approximate dominant singular vector (power iteration) on the tour-subgraph visitation matrix
    # Captures principal attractor direction across correlated subgraphs
    node_degrees = np.sum(edge_n_used, axis=1)
    norm_deg = node_degrees / (np.linalg.norm(node_degrees) + 1e-12)
    subgraph_vector = np.dot(edge_n_used, norm_deg)
    attractor_weight = subgraph_vector / (np.linalg.norm(subgraph_vector) + 1e-12)
    edge_attractor_overlap = 0.5 * (attractor_weight[u_idx] + attractor_weight[v_idx])

    # Dynamic switching between micro-perturbation (bottleneck focus) and macro-rupture (anti-attractor shock)
    macro_rupture_trigger = np.tanh(3.0 * drift_rigidity + 2.0 * np.mean(tour_used) / (np.max(edge_n_used) + 1e-12))
    k_frac = np.clip(0.08 + 0.16 * macro_rupture_trigger, 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))

    # Edge selection: blends regret utility with attractor spectral overlap
    selection_score = regret_utility * (1.0 + macro_rupture_trigger * edge_attractor_overlap)
    top_k_indices = np.argpartition(selection_score, -k)[-k:]

    # Multiplier scaling
    burst_multiplier = 1.0 + 3.0 * macro_rupture_trigger
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    # Structural bridge vulnerability
    bridge_vulnerability = tour_dists / edge_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    # Core logic: selective top-k edge penalization with adaptive anti-attractor modulation
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]

        micro_factor = 1.0 + 0.35 * np.tanh(norm_bridge[idx] - 1.0)
        macro_factor = 1.0 + 0.50 * macro_rupture_trigger * edge_attractor_overlap[idx]
        usage_factor = 1.0 + 0.20 * np.sqrt(current_usage)

        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * micro_factor * macro_factor
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance