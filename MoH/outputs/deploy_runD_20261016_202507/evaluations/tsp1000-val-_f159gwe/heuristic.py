#{Propagate GLS penalties along contiguous tour segments using cluster-aware non-maximum suppression and smooth chain-diffusion to rupture coherent tangled sub-paths rather than isolated edges.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    utilities = tour_dists / (1.0 + tour_used)
    k_penalize = max(1, int(np.ceil(0.10 * n)))
    top_indices = np.argpartition(utilities, -k_penalize)[-k_penalize:]

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    updated_tour_used = edge_n_used[u_idx, v_idx]
    overlap_ratio = np.mean(updated_tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    pred_used = edge_n_used[w_idx, u_idx]
    succ_used = edge_n_used[u_idx, v_idx]
    tot_used = pred_used + succ_used + 1e-12
    p1 = pred_used / tot_used
    p2 = succ_used / tot_used
    incident_entropy = -(p1 * np.log2(np.maximum(p1, 1e-12)) + p2 * np.log2(np.maximum(p2, 1e-12)))

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    strain_mod = 0.5 * (norm_tension + np.roll(norm_tension, -1))
    irreplaceability_mod = 0.5 * (norm_irreplaceability + np.roll(norm_irreplaceability, -1))
    entropy_mod = 0.5 * (incident_entropy + np.roll(incident_entropy, -1))
    rel_len = tour_dists / mean_tour_dist

    modulation = 1.0 + 0.35 * rel_len + 0.25 * strain_mod + 0.20 * entropy_mod - 0.25 * irreplaceability_mod
    modulation = np.maximum(0.4, modulation)

    raw_penalties = np.zeros(n, dtype=np.float64)
    raw_penalties[top_indices] = base_penalty_scale * modulation[top_indices] * updated_tour_used[top_indices]

    # Ruin-and-Recreate: Spatial cluster-aware chain diffusion and non-maximum suppression
    # Propagate penalty mass to neighboring edges along contiguous chains to break full sub-paths
    # while damping repetitive adjacent impulses
    is_selected = np.zeros(n, dtype=bool)
    is_selected[top_indices] = True

    # Identify clustered adjacent candidates and diffuse penalties along the chain
    fwd_diffused = np.roll(raw_penalties, 1)
    bwd_diffused = np.roll(raw_penalties, -1)

    # Chains of selected edges reinforce the sub-path breaking penalty; isolated edges diffuse slightly
    chain_cohesion = np.where(is_selected & np.roll(is_selected, 1), 1.25, 0.85)
    effective_penalties = raw_penalties.copy()

    # Apply directional smooth damping for clustered regions and chain propagation
    chain_boost = (fwd_diffused + bwd_diffused) * 0.20 * chain_cohesion
    effective_penalties = np.maximum(effective_penalties, chain_boost)

    # Distribute final aggregated penalties to distance matrix symmetrically
    active_mask = effective_penalties > 1e-6
    active_indices = np.where(active_mask)[0]

    act_u = u_idx[active_indices]
    act_v = v_idx[active_indices]
    act_penalties = effective_penalties[active_indices]

    np.add.at(updated_edge_distance, (act_u, act_v), act_penalties)
    np.add.at(updated_edge_distance, (act_v, act_u), act_penalties)

    return updated_edge_distance