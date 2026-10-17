import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)

    # Track tour sequence indices: prev, curr, next, and next2
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    prev_idx = np.roll(u_idx, 1)
    next2_idx = np.roll(u_idx, -2)

    # Decay usage frequency matrix
    edge_n_used *= 0.96

    # Calculate basic edge tour distances and usages
    e_curr_dist = edge_distance[u_idx, v_idx]
    e_curr_used = edge_n_used[u_idx, v_idx]
    e_prev_dist = edge_distance[prev_idx, u_idx]
    e_next_dist = edge_distance[v_idx, next2_idx]

    mean_dist = np.mean(e_curr_dist) + 1e-12
    rel_curr_dist = e_curr_dist / mean_dist

    # Base utility of isolated edges
    base_utility = rel_curr_dist / (1.0 + 0.45 * e_curr_used)

    # Path-aware curvature distortion: evaluate shortcut potential across 2-edge paths
    # (d(u, v) + d(v, w)) - d(u, w): larger excess indicates an inefficient bent detour
    shortcut_v = edge_distance[u_idx, next2_idx]
    detour_excess_v = (e_curr_dist + e_next_dist) - shortcut_v
    shortcut_u = edge_distance[prev_idx, v_idx]
    detour_excess_u = (e_prev_dist + e_curr_dist) - shortcut_u

    norm_excess = (np.maximum(0.0, detour_excess_v) + np.maximum(0.0, detour_excess_u)) / (2.0 * mean_dist)
    curvature_factor = 1.0 + 0.8 * np.tanh(norm_excess)

    # Consecutive path segment stiffness & vertex congestion
    # Vertices incident to consecutively heavily used edges create rigid local traps
    incident_usage = 0.5 * (edge_n_used[prev_idx, u_idx] + edge_n_used[u_idx, v_idx]) + \
                     0.5 * (edge_n_used[u_idx, v_idx] + edge_n_used[v_idx, next2_idx])
    mean_incident_usage = np.mean(incident_usage) + 1e-12
    rel_incident_usage = incident_usage / mean_incident_usage

    # Boost factor exponentially for pairs of consecutive suboptimal edges on rigid vertices
    stiffness_boost = np.exp(np.clip(rel_incident_usage - 1.0, -1.0, 2.0))

    # Combined path-aware utility
    combined_utilities = base_utility * curvature_factor * stiffness_boost

    # Select top-k edges with highest combined utility (approx 12% of nodes)
    k = max(1, int(np.ceil(0.12 * num_nodes)))
    top_k_indices = np.argpartition(combined_utilities, -k)[-k:]

    # Density modulation
    k_nn = min(8, num_nodes - 1)
    sorted_part = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(sorted_part, axis=1) + 1e-12
    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    avg_usage = np.mean(e_curr_used)
    stagnation = 1.0 + np.tanh(avg_usage * 0.15)
    base_penalty_scale = 0.25 * mean_dist * stagnation

    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        penalty = base_penalty_scale * (1.0 + 0.4 * density_mod) * (1.0 + 0.1 * edge_n_used[u, v])

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance