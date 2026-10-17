# {Replaces the additive modulation with a bounded multiplicative gating mechanism using hyperbolic tangent and sigmoid activations to cohesively scale tension, entropy, and irreplaceability without unbounded penalty explosion.}

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

    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor

    for idx in top_indices:
        u, v = u_idx[idx], v_idx[idx]
        next_idx = (idx + 1) % n

        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[next_idx])
        irrep_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[next_idx])
        entropy_mod = 0.5 * (incident_entropy[idx] + incident_entropy[next_idx])
        rel_len = tour_dists[idx] / mean_tour_dist

        gate_length = 1.0 + np.tanh(rel_len - 1.0)
        gate_strain = 1.0 + 0.5 * np.tanh(strain_mod - 1.0)
        gate_irrep = 1.0 / (1.0 + np.exp(2.0 * (irrep_mod - 1.0)))
        gate_entropy = 0.8 + 0.4 * entropy_mod

        modulation = gate_length * gate_strain * (0.3 + 0.7 * gate_irrep) * gate_entropy
        modulation = max(0.25, modulation)

        penalty = base_penalty_scale * modulation * edge_n_used[u, v]
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance