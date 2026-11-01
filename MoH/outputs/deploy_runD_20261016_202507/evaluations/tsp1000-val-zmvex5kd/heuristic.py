#{This algorithm replaces isolated utility edge selection with a cohesive subpath ruin-and-recreate operator, identifying top utility seed edges and propagating penalties along contiguous high-tension tour neighbors to form structured escape corridors while scaling penalties by stagnation and geometric irreplaceability.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    n = len(local_opt_tour)

    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)
    w_idx = np.roll(u_idx, 1)

    # 1. Decay memory usage
    edge_n_used *= 0.95

    tour_dists = edge_distance[u_idx, v_idx]
    pred_dists = edge_distance[w_idx, u_idx]
    tour_used = edge_n_used[u_idx, v_idx]

    # 2. Geometric tension and irreplaceability
    skip_dists = edge_distance[w_idx, v_idx]
    chord_base = skip_dists + 1e-12
    incident_sum = pred_dists + tour_dists
    zigzag_tension = incident_sum / chord_base
    norm_tension = zigzag_tension / (np.mean(zigzag_tension) + 1e-12)

    replacement_ratio = skip_dists / (tour_dists + 1e-12)
    norm_irreplaceability = replacement_ratio / (np.mean(replacement_ratio) + 1e-12)

    mean_tour_dist = np.mean(tour_dists) + 1e-12
    overlap_ratio = np.mean(tour_used > 1.0)
    stagnation_factor = 1.0 + 1.5 * overlap_ratio

    # 3. Base utility for seeds
    utilities = tour_dists / (1.0 + tour_used)

    # Target 10% total edges to penalize via contiguous clusters
    k_total = max(1, int(np.ceil(0.10 * n)))
    cluster_size = 4
    n_seeds = max(1, int(np.ceil(k_total / cluster_size)))

    # Identify top seed edges with argpartition
    seed_candidates = np.argpartition(utilities, -n_seeds)[-n_seeds:]
    seed_candidates = seed_candidates[np.argsort(-utilities[seed_candidates])]

    # 4. Propagate penalties along contiguous subpaths (cohesive ruin corridors)
    penalized_mask = np.zeros(n, dtype=bool)
    count = 0

    for seed in seed_candidates:
        if count >= k_total:
            break
        if not penalized_mask[seed]:
            penalized_mask[seed] = True
            count += 1

        # Grow corridor forward and backward prioritizing high tension
        left = seed
        right = seed
        while (right - left + 1) < cluster_size and count < k_total:
            next_right = (right + 1) % n
            prev_left = (left - 1 + n) % n

            # Tension-guided directional expansion
            t_right = norm_tension[next_right] if not penalized_mask[next_right] else -1.0
            t_left = norm_tension[prev_left] if not penalized_mask[prev_left] else -1.0

            if t_right < 0 and t_left < 0:
                break
            elif t_right >= t_left:
                penalized_mask[next_right] = True
                right = next_right
                count += 1
            else:
                penalized_mask[prev_left] = True
                left = prev_left
                count += 1

    selected_indices = np.where(penalized_mask)[0]

    # Register edge increments
    for idx in selected_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

    # 5. Modulated distance update
    base_penalty_scale = 0.25 * mean_tour_dist * stagnation_factor
    for idx in selected_indices:
        u, v = u_idx[idx], v_idx[idx]
        strain_mod = 0.5 * (norm_tension[idx] + norm_tension[(idx + 1) % n])
        irreplaceability_mod = 0.5 * (norm_irreplaceability[idx] + norm_irreplaceability[(idx + 1) % n])
        rel_len = tour_dists[idx] / mean_tour_dist

        modulation = 1.0 + 0.35 * rel_len + 0.30 * strain_mod - 0.25 * irreplaceability_mod
        modulation = max(0.4, modulation)
        penalty = base_penalty_scale * modulation * edge_n_used[u, v]

        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance