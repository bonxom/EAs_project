#{Reconstructs contiguous tour segmentation into a 2D spatial cluster ruin strategy using shared k-NN Jaccard similarity and mutual neighborhood overlap to selectively amplify penalties on intra-cluster bottlenecks while moderating inter-cluster boundary edges.}#

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Geometric decay of historical edge usage
    edge_n_used *= 0.96

    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # 2D spatial micro-neighborhood extraction via k-NN
    k_nn = min(8, num_nodes - 1)
    knn_indices = np.argpartition(edge_distance, k_nn, axis=1)[:, 1:k_nn]

    # Partitioned distances for opportunity cost & local scale
    part_dists = np.take_along_axis(edge_distance, knn_indices, axis=1)
    sorted_part = np.sort(part_dists, axis=1)
    second_cands = sorted_part[:, 1] if sorted_part.shape[1] > 1 else sorted_part[:, 0]
    third_cands = sorted_part[:, 2] if sorted_part.shape[1] > 2 else second_cands

    local_scale = np.mean(sorted_part, axis=1) + 1e-12
    edge_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12

    # Opportunity cost / regret utility calculation
    alt_cost_u = 0.6 * second_cands[u_idx] + 0.4 * third_cands[u_idx]
    alt_cost_v = 0.6 * second_cands[v_idx] + 0.4 * third_cands[v_idx]
    edge_regret = np.maximum(0.0, tour_dists - 0.5 * (alt_cost_u + alt_cost_v))
    norm_regret = edge_regret / edge_scale
    regret_utility = (norm_regret + (tour_dists / (mean_dist + 1e-12))) / (1.0 + 0.45 * tour_used)

    # Stagnation indicators
    total_tour_used = np.sum(tour_used) + 1e-12
    prob_dist = tour_used / total_tour_used
    shannon_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
    norm_entropy = shannon_entropy / (np.log(num_nodes) + 1e-12)
    cycling_ratio = np.mean(tour_used > 1.5)
    high_usage_ratio = np.mean(tour_used > 3.0)
    entropy_stagnation = np.exp(np.clip(1.0 - norm_entropy, 0.0, 3.0)) - 1.0
    stagnation_cycle = cycling_ratio + 1.5 * high_usage_ratio + 0.5 * entropy_stagnation

    k_frac = np.clip(0.10 + 0.12 * np.tanh(stagnation_cycle), 0.08, 0.25)
    k = max(2, int(np.ceil(k_frac * num_nodes)))
    top_k_indices = np.argpartition(regret_utility, -k)[-k:]

    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Ruin-and-Recreate: 2D spatial micro-cluster and boundary detection
    # Compute shared k-NN Jaccard similarity and mutual neighborhood affinity for candidate edges
    u_cand = u_idx[top_k_indices]
    v_cand = v_idx[top_k_indices]

    knn_u = knn_indices[u_cand]  # shape: (k, k_nn - 1)
    knn_v = knn_indices[v_cand]  # shape: (k, k_nn - 1)

    # Vectorized intersection count across candidate node pairs
    shared_neighbors = np.sum(knn_u[:, :, None] == knn_v[:, None, :], axis=(1, 2))
    jaccard_overlap = shared_neighbors / (2 * (k_nn - 1) - shared_neighbors + 1e-12)

    # Mutual neighborhood check: whether v in knn(u) and u in knn(v)
    is_v_in_u = np.any(knn_u == v_cand[:, None], axis=1)
    is_u_in_v = np.any(knn_v == u_cand[:, None], axis=1)
    mutual_knn = (is_v_in_u & is_u_in_v).astype(float)

    # Spatial clustering factor: High overlap implies intra-cluster bottleneck; low implies inter-cluster bridge
    spatial_cohesion = 0.6 * jaccard_overlap + 0.4 * mutual_knn

    # Intra-cluster bottlenecks receive boosted penalties to break congested clusters;
    # Inter-cluster bridges are protected from over-penalization to preserve global topology.
    cluster_modulation = 1.0 + 0.50 * np.tanh(2.5 * (spatial_cohesion - 0.25))

    bridge_vulnerability = tour_dists[top_k_indices] / edge_scale[top_k_indices]
    norm_bridge = bridge_vulnerability / (np.mean(tour_dists / edge_scale) + 1e-12)

    for i, idx in enumerate(top_k_indices):
        u, v = u_cand[i], v_cand[i]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0

        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]

        usage_factor = 1.0 + 0.20 * np.sqrt(current_usage)
        bridge_factor = 1.0 + 0.30 * np.tanh(norm_bridge[i] - 1.0)
        c_mod = cluster_modulation[i]

        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * c_mod * bridge_factor
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance