# {Structural Pareto Deficit-Guided Penalty Update: Penalizes high-utility tour edges scaled by local graph bottleneck dynamics and usage frequency to break deep local optima while preserving global topology.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Create a copy of the edge distance matrix to avoid modifying the original
    updated_edge_distance = edge_distance.copy().astype(float)
    n = len(local_opt_tour)
    if n < 4:
        return updated_edge_distance

    # Extract tour edges
    u_idx = local_opt_tour
    v_idx = np.roll(local_opt_tour, -1)

    # Compute incident average edge distance for structural bottleneck context
    node_avg_dist = np.mean(edge_distance, axis=1)

    # Calculate costs, usage, and structural bottleneck ratio for tour edges
    tour_costs = edge_distance[u_idx, v_idx]
    tour_uses = edge_n_used[u_idx, v_idx]
    struct_factor = tour_costs / (0.5 * (node_avg_dist[u_idx] + node_avg_dist[v_idx]) + 1e-6)

    # Multi-criteria Pareto-inspired utility computation
    utilities = (tour_costs / (1.0 + tour_uses)) * (1.0 + 0.3 * struct_factor)

    # Identify maximum utility edges in the current local optimal tour
    max_util = np.max(utilities)
    max_mask = utilities >= (max_util - 1e-8)

    # Apply targeted penalty to max utility edges
    penalized_u = u_idx[max_mask]
    penalized_v = v_idx[max_mask]

    # Calculate dynamic penalty step size scaled by edge cost
    penalty_inc = tour_costs[max_mask] * 0.1

    # Symmetric penalty update
    np.add.at(updated_edge_distance, (penalized_u, penalized_v), penalty_inc)
    np.add.at(updated_edge_distance, (penalized_v, penalized_u), penalty_inc)

    return updated_edge_distance
