# {Penalizes high-utility tour edges scaled by local density, tour-segment usage stagnation, and spatial centrality, selectively targeting inefficient long edges in dense clusters while protecting sparse boundary links for TSP-1000.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Mild decay to edge usage memory to maintain dynamic exploration
    edge_n_used *= 0.96

    tour_dist = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    avg_tour_dist = np.mean(tour_dist) + 1e-12

    # GLS-style utility for tour edges
    utilities = tour_dist / (1.0 + tour_used)

    # Select top fraction of edges to penalize (scaled for N=1000)
    k = max(5, int(np.ceil(0.12 * n)))
    top_k_indices = np.argsort(utilities)[-k:]

    selected_u = u_idx[top_k_indices]
    selected_v = v_idx[top_k_indices]

    # Increment edge usage count for selected edges
    edge_n_used[selected_u, selected_v] += 1.0
    edge_n_used[selected_v, selected_u] += 1.0

    # Local density estimation via k-nearest neighbor distance proxy
    # For N=1000, vectorized partition over nearest neighbors
    k_nn = 8
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(part_dists, axis=1) + 1e-12
    rel_density = np.mean(local_scale) / local_scale
    norm_density = rel_density / (np.max(rel_density) + 1e-12)

    # Global spatial centrality (closeness to center of coordinates)
    node_mean_dist = np.mean(edge_distance, axis=1) + 1e-12
    spatial_centrality = np.mean(node_mean_dist) / node_mean_dist
    norm_spatial_centrality = spatial_centrality / (np.max(spatial_centrality) + 1e-12)

    # Stagnation and edge length factors
    stagnation_ratio = np.mean(tour_used > 1.0)
    base_lambda = (0.25 + 0.25 * np.tanh(2.0 * stagnation_ratio)) * avg_tour_dist

    # Compute edge-specific penalty modulation
    sel_u_dens = norm_density[selected_u]
    sel_v_dens = norm_density[selected_v]
    sel_u_cent = norm_spatial_centrality[selected_u]
    sel_v_cent = norm_spatial_centrality[selected_v]

    density_mod = 0.5 * (sel_u_dens + sel_v_dens)
    centrality_mod = 0.5 * (sel_u_cent + sel_v_cent)
    length_mod = tour_dist[top_k_indices] / avg_tour_dist

    modulation = 0.8 + 0.4 * density_mod + 0.3 * centrality_mod + 0.5 * length_mod
    penalties = base_lambda * modulation * (1.0 + 0.2 * edge_n_used[selected_u, selected_v])

    # Apply penalties symmetrically
    np.add.at(updated_edge_distance, (selected_u, selected_v), penalties)
    np.add.at(updated_edge_distance, (selected_v, selected_u), penalties)

    return updated_edge_distance