#{Replaces the uniform global stagnation scalar with a heterogeneous multi-scale penalty profile balancing individualized edge persistence against localized 2-hop topological entropy for scale-adaptive escape in large TSP instances.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Symmetric usage decay
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # O(N) utility evaluation and selection
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # Symmetric edge usage update
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12

    # Local context extraction in O(N) along the tour structure
    w_idx = np.roll(u_idx, 1)
    w_next = np.roll(u_idx, -2)

    # 2-hop local incident edge usage distribution for each node along the tour
    # Captures localized search rigidity/entropy without O(N^2) full-matrix scans
    u_usage_pred = edge_n_used[u_idx, w_idx]
    u_usage_succ = edge_n_used[u_idx, v_idx]
    v_usage_succ2 = edge_n_used[v_idx, w_next]

    total_local_u = u_usage_pred + u_usage_succ + 1e-12
    p_u1 = u_usage_pred / total_local_u
    p_u2 = u_usage_succ / total_local_u
    h_u = -(p_u1 * np.log2(p_u1 + 1e-12) + p_u2 * np.log2(p_u2 + 1e-12))

    total_local_v = u_usage_succ + v_usage_succ2 + 1e-12
    p_v1 = u_usage_succ / total_local_v
    p_v2 = v_usage_succ2 / total_local_v
    h_v = -(p_v1 * np.log2(p_v1 + 1e-12) + p_v2 * np.log2(p_v2 + 1e-12))

    # Local topological entropy: low entropy indicates entrenched, inflexible pathways
    local_rigidity = 1.0 - 0.5 * (h_u + h_v)

    # Individual edge persistence factor: how entrenched is this specific edge relative to current tour
    updated_tour_used = edge_n_used[u_idx, v_idx]
    max_tour_used = np.max(updated_tour_used) + 1e-12
    edge_persistence = updated_tour_used / max_tour_used

    # Local geometric congestion
    tour_neighbor_dist = 0.5 * (tour_dists + edge_distance[u_idx, w_idx]) + 1e-12
    local_density = mean_tour_dist / tour_neighbor_dist
    norm_density = local_density / (np.max(local_density) + 1e-12)

    # Vectorized multi-scale penalty application on the selected top edges
    sel_u = u_idx[top_indices]
    sel_v = v_idx[top_indices]
    sel_dists = tour_dists[top_indices]
    sel_used = edge_n_used[sel_u, sel_v]

    rel_len = sel_dists / mean_tour_dist
    pers = edge_persistence[top_indices]
    rigidity = local_rigidity[top_indices]
    density_mod = 0.5 * (norm_density[sel_u] + norm_density[sel_v])

    # Heterogeneous balance: strong escape for persistent & rigid links; proportional relaxation for flexible links
    individual_escape_factor = 1.0 + 1.2 * pers + 0.8 * rigidity
    individual_mod = 1.0 + 0.4 * rel_len + 0.4 * density_mod

    penalties = 0.22 * mean_tour_dist * individual_escape_factor * individual_mod * sel_used

    for i, idx in enumerate(top_indices):
        u, v = sel_u[i], sel_v[i]
        p = penalties[i]
        updated_edge_distance[u, v] += p
        updated_edge_distance[v, u] += p

    return updated_edge_distance