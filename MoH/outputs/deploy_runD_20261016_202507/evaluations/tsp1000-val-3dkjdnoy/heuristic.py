import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)

    # Tour node indexing: pred_pred (z), pred (w), curr (u), succ (v), succ_succ (x)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)
    x_idx = np.roll(u_idx, -2)

    # Usage decay
    edge_n_used *= 0.95

    # Distances along the tour
    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    succ_dists = edge_distance[v_idx, x_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Candidate utility for guided local search edge selection
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Increment usage count for penalized edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # Multi-scale bypass detour ratio:
    # 1. Direct 2-opt skip chords across node u (w->v) and across node v (u->x)
    chord_wv = edge_distance[w_idx, v_idx]
    chord_ux = edge_distance[u_idx, x_idx]
    tri_bypass_eff_u = (pred_dists + tour_dists) / (chord_wv + 1e-12)
    tri_bypass_eff_v = (tour_dists + succ_dists) / (chord_ux + 1e-12)

    # 2. 4-node diamond shortcut bypassing edge (u, v): path (w->u->v->x) vs chord (w->x)
    diamond_chord_wx = edge_distance[w_idx, x_idx]
    diamond_path = pred_dists + tour_dists + succ_dists
    diamond_detour = diamond_path / (diamond_chord_wx + 1e-12)

    # Combined multi-scale geometric bypass score
    bypass_score = 0.35 * tri_bypass_eff_u + 0.35 * tri_bypass_eff_v + 0.30 * diamond_detour
    norm_bypass = bypass_score / (np.mean(bypass_score) + 1e-12)

    # Historical usage stability between predecessor and successor
    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    usage_dispersion = np.abs(succ_used - pred_used) / (succ_used + pred_used + 1.0)

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    # Vectorized / selective penalty assignment
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        rel_len = tour_dists[idx] / mean_tour_dist
        detour_mod = norm_bypass[idx]
        disp_mod = 0.5 * (usage_dispersion[idx] + usage_dispersion[(idx + 1) % n])

        # High bypass ratio indicates inefficient geometric zigzag/crossover; amplify penalty
        modulation = 1.0 + 0.35 * rel_len + 0.35 * detour_mod - 0.20 * disp_mod
        modulation = max(0.5, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance