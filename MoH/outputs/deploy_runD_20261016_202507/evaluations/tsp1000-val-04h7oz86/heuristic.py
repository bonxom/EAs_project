#{Adaptive dual-phase stagnation controller dynamically scales memory decay to unfreeze search and smoothly transitions penalty focus from edge lengths to local k-NN topological density when tour overlap is high.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Measure current tour overlap and stagnation prior to memory update
    prev_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(prev_tour_used > 1.0)

    # Adaptive decay: accelerate forgetting when trapped in high-overlap stagnation
    adaptive_decay = 0.95 - 0.10 * np.clip(overlap_ratio, 0.0, 1.0)
    edge_n_used *= adaptive_decay

    # Tour metrics and edge utility computation
    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    utilities = tour_dists / (1.0 + tour_used)

    # Select top 10% edges with highest utility to penalize
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    stagnation_factor = 1.0 + 1.8 * overlap_ratio

    # Geometry & Irreplaceability cues
    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    # Incident edge usage entropy
    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    # Topological density cue (estimated from local incident geometry)
    k_local_dists = 0.5 * (pred_dists + tour_dists)
    local_density = mean_tour_dist / (k_local_dists + 1e-12)
    norm_density = local_density / (np.mean(local_density) + 1e-12)

    # Dual-phase penalty blending: shift from length-dominant to density-dominant under stagnation
    w_density = np.clip(overlap_ratio, 0.0, 1.0)
    w_length = 1.0 - w_density

    base_penalty_scale = (0.20 + 0.15 * overlap_ratio) * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        next_idx = (idx + 1) % n

        rel_len = tour_dists[idx] / mean_tour_dist
        dens_mod = 0.5 * (norm_density[idx] + norm_density[next_idx])
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[next_idx])
        irreplace_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[next_idx])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[next_idx])

        core_factor = w_length * (0.40 * rel_len) + w_density * (0.40 * dens_mod)
        modulation = 1.0 + core_factor + 0.25 * strain_mod + 0.20 * entropy_mod - 0.25 * irreplace_mod
        modulation = max(0.4, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance