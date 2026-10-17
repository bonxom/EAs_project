#{Dynamically scale edge penalty count from 5% to 15% via tour stagnation and allocate continuous fractional memory increments driven by relative edge length and stagnation persistence to selectively eradicate entrenched long edges.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Decayed frequency memory
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    novelty_ratio = np.mean(tour_used < 0.5)
    stagnation = 1.0 - novelty_ratio

    # Adaptive Pareto selection size between 5% and 15% inversely proportional to tour novelty
    k_frac = 0.05 + 0.10 * stagnation
    k_penalize = max(1, min(n, int(np.ceil(k_frac * n))))

    # Composite utility considering guided local search utility and stagnation persistence
    raw_utilities = tour_dists / (1.0 + tour_used)
    persistence_factor = tour_used / (np.max(tour_used) + 1.0)
    selection_score = raw_utilities * (1.0 + 0.5 * persistence_factor)

    top_indices = np.argpartition(selection_score, -k_penalize)[-k_penalize:]

    # Continuous fractional usage increments to heavily weight entrenched long edges
    sig_stagnation = 1.0 / (1.0 + np.exp(-4.0 * (stagnation - 0.5)))
    rel_lengths = tour_dists[top_indices] / mean_tour_dist
    increments = 1.0 + sig_stagnation * rel_lengths

    for idx, inc in zip(top_indices, increments):
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += inc
        edge_n_used[v, u] += inc

    # Neighborhood baseline for opportunity cost
    unique_candidates = np.unique(np.concatenate([u_idx[top_indices], v_idx[top_indices]]))
    sub_dists = edge_distance[unique_candidates]
    k_nn = min(4, n - 1)
    part_k = np.partition(sub_dists, k_nn, axis=1)[:, 1:k_nn]
    node_alt_baseline = np.mean(part_k, axis=1) + 1e-12
    alt_map = np.zeros(n)
    alt_map[unique_candidates] = node_alt_baseline

    incident_sum = pred_dists + tour_dists
    skip_dists = edge_distance[w_idx, v_idx]
    zigzag_tension = incident_sum / (skip_dists + 1e-12)
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    base_penalty_scale = 0.25 * mean_tour_dist * (1.0 + 1.5 * (1.0 - novelty_ratio))

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        e_dist = tour_dists[idx]
        geom_alt = 0.5 * (alt_map[u] + alt_map[v])
        opp_ratio = e_dist / geom_alt
        opp_mod = np.clip(opp_ratio - 1.0, -0.5, 1.5)
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = e_dist / mean_tour_dist

        modulation = 1.0 + 0.30 * rel_len + 0.20 * strain_mod + 0.20 * entropy_mod + 0.25 * opp_mod
        modulation = max(0.35, modulation)
        penalty = base_penalty_scale * modulation * edge_n_used[u, v]

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance