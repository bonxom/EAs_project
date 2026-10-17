import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # 1. Decay past edge usage frequencies
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # 2. Guided Local Search candidate selection
    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    # 3. Increment usage for chosen edges
    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]

    # 4. Multi-scale stagnation depth tracking
    # Continuous retention proxy (soft Jaccard similarity across recent cycles)
    retention_ratio = np.mean(np.minimum(updated_tour_used, 2.0)) / 2.0
    # Saturation of penalties: how heavily concentrated the penalties are in the current tour
    max_used = np.max(edge_n_used) + 1e-12
    saturation_ratio = np.mean(updated_tour_used) / max_used

    # Deep stagnation metric combining retention and saturation
    stagnation_depth = np.clip(0.6 * retention_ratio + 0.4 * saturation_ratio, 0.0, 1.0)

    # Adaptive stagnation factor: dynamically scales base penalty pressure
    stagnation_factor = 1.0 + 2.5 * (stagnation_depth ** 1.5)

    # 5. Geometric tension and structural metrics
    skip_dists = edge_distance[w_idx, v_idx]
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / (skip_dists + 1e-12)
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    # 6. Dynamic balance between intensification and backbone diversification
    w_div = np.clip(stagnation_depth, 0.1, 0.9)
    w_int = 1.0 - w_div

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist

        # Intensification: penalize inefficient, high-strain, replaceable edges
        intensification_mod = 1.0 + 0.45 * rel_len + 0.30 * strain_mod - 0.25 * irreplaceability_mod

        # Diversification: penalize high-frequency stubborn backbone edges regardless of local efficiency
        usage_intensity = edge_n_used[u, v] / (np.mean(updated_tour_used) + 1e-12)
        diversification_mod = 1.0 + 0.50 * usage_intensity + 0.35 * entropy_mod

        modulation = w_int * intensification_mod + w_div * diversification_mod
        modulation = max(0.35, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance