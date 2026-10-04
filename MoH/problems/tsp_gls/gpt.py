import numpy as np
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    if n < 4:
        return updated_edge_distance
    u = np.asarray(local_opt_tour, dtype=int)
    v = np.roll(u, -1)
    p = np.roll(u, 1)
    c = edge_distance[u, v]
    tour_len = np.sum(c)
    mean_c = tour_len / max(n, 1)
    if mean_c <= 0.0:
        mean_c = 1.0
    d_uu = edge_distance[np.ix_(u, u)]
    d_vv = edge_distance[np.ix_(v, v)]
    delta_2opt = d_uu + d_vv - c[:, None] - c[None, :]
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    diff = np.minimum(diff, n - diff)
    valid_2opt = diff > 1
    delta_2opt_clipped = np.maximum(delta_2opt, 0.0)
    delta_2opt_masked = np.where(valid_2opt, delta_2opt_clipped, np.inf)
    min_2opt = np.min(delta_2opt_masked, axis=1)
    min_2opt = np.where(np.isinf(min_2opt), 0.0, min_2opt)
    tau = 0.1 * mean_c + 1e-8
    sterile_score_2opt = delta_2opt_clipped / (delta_2opt_clipped + tau)
    n_valid_2opt = np.maximum(np.sum(valid_2opt, axis=1), 1)
    s_2opt = np.sum(np.where(valid_2opt, sterile_score_2opt, 0.0), axis=1) / n_valid_2opt
    d_uk = d_uu
    d_kv = edge_distance[np.ix_(v, u)]
    delta_remove = edge_distance[p, v] - edge_distance[p, u] - c
    delta_node = d_uk + d_kv - c[:, None] + delta_remove[None, :]
    valid_node = diff > 1
    delta_node_clipped = np.maximum(delta_node, 0.0)
    delta_node_masked = np.where(valid_node, delta_node_clipped, np.inf)
    min_node = np.min(delta_node_masked, axis=1)
    min_node = np.where(np.isinf(min_node), 0.0, min_node)
    sterile_score_node = delta_node_clipped / (delta_node_clipped + tau)
    n_valid_node = np.maximum(np.sum(valid_node, axis=1), 1)
    s_node = np.sum(np.where(valid_node, sterile_score_node, 0.0), axis=1) / n_valid_node
    r_score = (min_2opt + 0.5 * min_node) / mean_c
    s_comb = 0.6 * s_2opt + 0.4 * s_node
    raw_rigidity = (1.0 + r_score) * (1.0 + s_comb)
    rigidity_inert = (
        0.10 * np.roll(raw_rigidity, 2)
        + 0.25 * np.roll(raw_rigidity, 1)
        + 0.30 * raw_rigidity
        + 0.25 * np.roll(raw_rigidity, -1)
        + 0.10 * np.roll(raw_rigidity, -2)
    )
    mean_inert = np.mean(rigidity_inert)
    inertia_factor = rigidity_inert / (mean_inert + 1e-8) if mean_inert > 0 else np.ones(n)
    p_counts = edge_n_used[u, v]
    base_utility = c / (1.0 + p_counts)
    gamma = 0.5
    utility = base_utility * (1.0 + gamma * inertia_factor)
    lambda_penalty = 0.3 * (tour_len / n)
    max_util = np.max(utility)
    penalize_indices = np.where(utility >= (max_util - 1e-6))[0]
    for k in penalize_indices:
        uid = u[k]
        vid = v[k]
        updated_edge_distance[uid, vid] += lambda_penalty
        updated_edge_distance[vid, uid] += lambda_penalty
    return updated_edge_distance
