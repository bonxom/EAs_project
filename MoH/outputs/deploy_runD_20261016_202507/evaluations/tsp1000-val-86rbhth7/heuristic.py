# {Scale-tailored guided local search for TSP-1000 penalizing high-utility tour edges scaled by local density, tour-level stagnation, and incident-edge usage to efficiently break long local-optima cycles without costly full-matrix scans.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Mild memory decay for dynamic exploration
    edge_n_used *= 0.96

    tour_dist = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Edge utility balances raw length against usage frequency
    utilities = tour_dist / (1.0 + tour_used)

    # Target top 10% edges with highest utility for large instances (N=1000)
    k = max(1, int(0.10 * num_nodes))
    top_k_indices = np.argpartition(utilities, -k)[-k:]

    selected_u = u_idx[top_k_indices]
    selected_v = v_idx[top_k_indices]

    # Increment usage counts for selected edges
    np.add.at(edge_n_used, (selected_u, selected_v), 1.0)
    np.add.at(edge_n_used, (selected_v, selected_u), 1.0)

    # Stagnation factor derived from tour usage overlap
    tour_usage = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(tour_usage > 1.0)
    stagnation_factor = 1.0 + 1.2 * overlap_ratio

    # Baseline penalty magnitude proportional to average tour distance
    mean_tour_dist = np.mean(tour_dist)
    lambda_base = 0.25 * mean_tour_dist * stagnation_factor

    # Local endpoint pressure from incident edges in current tour
    prev_idx = np.roll(u_idx, 1)
    node_incident_usage = edge_n_used[u_idx, prev_idx] + edge_n_used[u_idx, v_idx]
    norm_incident_usage = node_incident_usage / (np.max(node_incident_usage) + 1e-12)

    # Modulated penalties for selected top-utility edges
    endpoint_mod = 1.0 + 0.35 * (norm_incident_usage[top_k_indices] + np.roll(norm_incident_usage, -1)[top_k_indices])
    penalties = lambda_base * (1.0 + 0.1 * edge_n_used[selected_u, selected_v]) * endpoint_mod

    np.add.at(updated_edge_distance, (selected_u, selected_v), penalties)
    np.add.at(updated_edge_distance, (selected_v, selected_u), penalties)

    return updated_edge_distance