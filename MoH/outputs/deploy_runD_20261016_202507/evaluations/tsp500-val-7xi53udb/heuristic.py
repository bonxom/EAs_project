#{This algorithm identifies Pareto-inefficient edges via decayed usage and k-NN local slackness, applies amplified primary penalties to selected edges, and projects directional topological shadow penalties onto adjacent 1-hop and 2-hop edges along the tour to prevent trivial 2-opt re-linking loops.}

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

    # k-NN geometry scaling
    k_nn = min(8, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(part_dists, axis=1) + 1e-12
    min_nbr_u = part_dists[:, 0]
    edge_min_base = 0.5 * (min_nbr_u[u_idx] + min_nbr_u[v_idx])
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12
    slackness = np.maximum(0.0, (tour_dists - edge_min_base) / edge_scale)

    # Pareto utility & stagnation detection
    rel_dists = tour_dists / (mean_dist + 1e-12)
    efficiency_utility = rel_dists / (1.0 + 0.5 * tour_used)
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    total_tour_used = np.sum(tour_used) + 1e-12
    prob_dist = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    norm_entropy = shannon_entropy / (np.log(num_nodes) + 1e-12)
    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    stagnation_cycle = cycling_ratio + 1.5 * high_usage_ratio + 0.5 * entropy_stagnation

    # Adaptive top-k edge selection
    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))
    rank_eff = np.argsort(np.argsort(efficiency_utility))
    rank_slack = np.argsort(np.argsort(slackness))
    pareto_score = 0.55 * rank_eff + 0.45 * rank_slack
    top_k_indices = np.argpartition(pareto_score, -k)[-k:]

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier
    bridge_vulnerability = tour_dists / edge_scale
    norm_bridge = bridge_vulnerability / (np.mean(bridge_vulnerability) + 1e-12)

    # Vectorized computation of edge penalties
    primary_penalties = np.zeros(num_nodes, dtype=float)
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        bridge_factor = 1.0 + 0.30 * np.tanh(norm_bridge[idx] - 1.0)
        usage_factor = 1.0 + 0.20 * np.sqrt(current_usage)
        primary_penalties[idx] = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * bridge_factor

    # Ruin & Recreate: Directional shadow penalties to 1-hop and 2-hop tour adjacencies
    hop1_weight = 0.28
    hop2_weight = 0.10
    total_edge_penalties = np.copy(primary_penalties)
    total_edge_penalties += hop1_weight * (np.roll(primary_penalties, 1) + np.roll(primary_penalties, -1))
    total_edge_penalties += hop2_weight * (np.roll(primary_penalties, 2) + np.roll(primary_penalties, -2))

    # Apply accumulated penalties symmetrically
    active_mask = total_edge_penalties > 0
    u_active = u_idx[active_mask]
    v_active = v_idx[active_mask]
    pen_active = total_edge_penalties[active_mask]

    np.add.at(updated_edge_distance, (u_active, v_active), pen_active)
    np.add.at(updated_edge_distance, (v_active, u_active), pen_active)

    return updated_edge_distance