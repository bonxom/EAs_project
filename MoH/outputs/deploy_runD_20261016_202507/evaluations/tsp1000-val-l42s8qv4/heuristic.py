#{Adaptive burst-penalty strategy that dynamically scales the penalization budget between 5% and 20% based on tour stagnation and expands high-utility seeds into contiguous high-strain sub-chains to trigger structural multi-opt escapes in large tours.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # Core decayed frequency memory
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # Stagnation factor calculation based on current tour usage
    overlap_ratio = np.mean(tour_used > 1.0)
    avg_tour_used = np.mean(tour_used)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio + 0.5 * np.tanh(avg_tour_used)

    # Dynamic scaling of penalization budget between 5% and 20%
    # Stagnation typically ranges from 1.0 to ~2.5+
    k_fraction = np.clip(0.05 + 0.10 * (stagnation_factor - 1.0), 0.05, 0.20)
    k_penalize = max(1, int(np.ceil(k_fraction * n)))

    # Base utility formulation
    utilities = tour_dists / (1.0 + tour_used)

    # Tension / strain metric via triangle inequality shortcut bypass
    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    # Cluster-ruin seed selection: identify contiguous high-strain sub-chains
    # Smooth utility across consecutive edges to identify high-strain clusters
    cluster_score = (
        0.50 * (utilities / (np.mean(utilities) + 1e-12)) +
        0.30 * norm_tension +
        0.20 * (np.roll(utilities, -1) + np.roll(utilities, 1)) / (2.0 * np.mean(utilities) + 1e-12)
    )

    # Select top candidate seeds and expand into contiguous segments
    selected_mask = np.zeros(n, dtype=bool)
    seed_order = np.argsort(cluster_score)[::-1]

    # Target segment length adapts with stagnation to break larger sub-tours
    seg_len = max(2, int(np.round(2 + 3 * (stagnation_factor - 1.0))))

    count = 0
    for seed in seed_order:
        if count >= k_penalize:
            break
        # Expand around seed
        for offset in range(-seg_len // 2, seg_len // 2 + 1):
            idx = (seed + offset) % n
            if not selected_mask[idx]:
                selected_mask[idx] = True
                count += 1
                if count >= k_penalize:
                    break

    selected_indices = np.nonzero(selected_mask)[0]

    # Update edge usage counts for selected edges
    for idx in selected_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    # Apply modulated penalties symmetrically
    for idx in selected_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist

        modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.20 * entropy_mod - 0.25 * irreplaceability_mod
        modulation = max(0.4, modulation)
        penalty = base_penalty_scale * modulation * edge_n_used[u, v]

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance