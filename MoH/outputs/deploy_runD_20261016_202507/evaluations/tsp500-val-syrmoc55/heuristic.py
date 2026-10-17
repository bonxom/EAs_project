import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Decayed frequency tracking
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Utility calculation (relative distance / frequency dampener)
    rel_dists = tour_dists / (mean_dist + 1e-12)
    utilities = rel_dists / (1.0 + 0.4 * tour_used)

    # Penalize top-k fraction (~12%)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    # Degree-of-freedom calculation based on top-5 alternative nearest neighbors
    # Nodes with tight, equidistant alternatives have high degrees of freedom (branching potential),
    # whereas isolated nodes with distant alternatives have low freedom and need penalty dampening.
    k_nn = min(6, num_nodes - 1)
    part_nn = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    d_first = part_nn[:, 0] + 1e-12
    d_mean_nn = np.mean(part_nn, axis=1) + 1e-12
    # Dispersion ratio: values near 1 indicate abundant equidistant candidates (high freedom)
    ratio = d_mean_nn / d_first
    freedom = 1.0 / (np.log1p(ratio) + 1e-12)
    freedom_norm = freedom / (np.mean(freedom) + 1e-12)

    # Stagnation factor based on average tour edge reuse
    avg_usage = np.mean(tour_used)
    stagnation = 1.0 + np.tanh(0.2 * avg_usage)
    base_penalty = 0.28 * mean_dist * stagnation

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        # Asymmetric degree-of-freedom dampener
        dof_mod = 0.5 * (freedom_norm[u] + freedom_norm[v])
        penalty_scale = base_penalty * dof_mod * np.log1p(edge_n_used[u, v] + 1.0)

        updated_edge_distance[u, v] += penalty_scale
        updated_edge_distance[v, u] += penalty_scale

    return updated_edge_distance