import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)
    num_nodes = len(local_opt_tour)
    u_idx = np.asarray(local_opt_tour)
    v_idx = np.roll(u_idx, -1)

    # Track edge usage with exponential decay
    edge_n_used *= 0.96
    tour_dists = edge_distance[u_idx, v_idx]
    tour_used = edge_n_used[u_idx, v_idx]
    mean_dist = np.mean(tour_dists)

    # Static k-NN local distance normalization
    k_nn = min(8, num_nodes - 1)
    part_dists = np.partition(edge_distance, k_nn, axis=1)[:, 1:k_nn]
    local_scale = np.mean(part_dists, axis=1) + 1e-12
    min_nbr_u = part_dists[:, 0]
    edge_min_base = 0.5 * (min_nbr_u[u_idx] + min_nbr_u[v_idx])

    # Slackness and efficiency utility criteria
    edge_local_scale = 0.5 * (local_scale[u_idx] + local_scale[v_idx]) + 1e-12
    slackness = np.maximum(0.0, (tour_dists - edge_min_base) / edge_local_scale)
    rel_dists = tour_dists / (mean_dist + 1e-12)
    efficiency_utility = rel_dists / (1.0 + 0.5 * tour_used)

    # Incident node congestion / frequency centrality diversity metric (fewer already-penalized neighbors preferred)
    node_usage_sum = np.sum(edge_n_used, axis=1)
    edge_congestion = 0.5 * (node_usage_sum[u_idx] + node_usage_sum[v_idx])
    dispersion_score = 1.0 / (1.0 + edge_congestion)

    # Stagnation-driven adaptive penalty scaling and quota
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

    # Non-dominated front decomposition across (efficiency, slackness, dispersion)
    # Objectives to maximize: obj1 = efficiency_utility, obj2 = slackness, obj3 = dispersion_score
    objectives = np.column_stack((efficiency_utility, slackness, dispersion_score))

    n_edges = num_nodes
    selected_indices = []
    remaining_indices = list(range(n_edges))

    while len(selected_indices) < k and len(remaining_indices) > 0:
        rem_arr = np.array(remaining_indices)
        pts = objectives[rem_arr]

        # Identify non-dominated points in rem_arr
        n_rem = len(rem_arr)
        is_dominated = np.zeros(n_rem, dtype=bool)
        for i in range(n_rem):
            if is_dominated[i]:
                continue
            # j dominates i if j >= i everywhere and strictly greater somewhere
            ge = np.all(pts >= pts[i], axis=1)
            gt = np.any(pts > pts[i], axis=1)
            dominates_i = ge & gt
            dominates_i[i] = False
            if np.any(dominates_i):
                is_dominated[i] = True
            else:
                dominated_by_i = np.all(pts[i] >= pts, axis=1) & np.any(pts[i] > pts, axis=1)
                is_dominated[dominated_by_i] = True

        front = rem_arr[~is_dominated]

        if len(selected_indices) + len(front) <= k:
            selected_indices.extend(front.tolist())
            remaining_indices = [idx for idx in remaining_indices if idx not in set(front)]
        else:
            needed = k - len(selected_indices)
            front_pts = objectives[front]
            front_size = len(front)
            crowding_dist = np.zeros(front_size, dtype=float)

            # Crowding distance computation for tie-breaking on the boundary front
            for m in range(3):
                order = np.argsort(front_pts[:, m])
                crowding_dist[order[0]] = 1e9
                crowding_dist[order[-1]] = 1e9
                m_range = front_pts[order[-1], m] - front_pts[order[0], m]
                if m_range > 1e-12:
                    crowding_dist[order[1:-1]] += (front_pts[order[2:], m] - front_pts[order[:-2], m]) / m_range

            best_in_front = front[np.argsort(-crowding_dist)[:needed]]
            selected_indices.extend(best_in_front.tolist())
            break

    top_k_indices = np.array(selected_indices, dtype=int)

    # Continuous segment length calculation for coherent path disruption
    burst_multiplier = 1.0 + 1.8 * cycling_ratio + 2.5 * high_usage_ratio + 0.8 * entropy_stagnation
    base_penalty_scale = 0.25 * mean_dist * burst_multiplier

    is_penalized = np.zeros(num_nodes, dtype=bool)
    is_penalized[top_k_indices] = True
    segment_lengths = np.zeros(num_nodes, dtype=float)
    visited = np.zeros(num_nodes, dtype=bool)

    for i in range(num_nodes):
        if is_penalized[i] and not visited[i]:
            start = i
            while is_penalized[(start - 1) % num_nodes] and not visited[(start - 1) % num_nodes]:
                start = (start - 1) % num_nodes
                if start == i:
                    break
            curr = start
            chain = []
            while is_penalized[curr] and not visited[curr]:
                visited[curr] = True
                chain.append(curr)
                curr = (curr + 1) % num_nodes
                if curr == start:
                    break
            chain_len = len(chain)
            for idx in chain:
                segment_lengths[idx] = chain_len

    node_density = 1.0 / local_scale
    norm_density = node_density / (np.mean(node_density) + 1e-12)

    # Apply diverse penalties
    for idx in top_k_indices:
        u, v = u_idx[idx], v_idx[idx]
        edge_n_used[u, v] += 1.0
        edge_n_used[v, u] += 1.0
        density_mod = 0.5 * (norm_density[u] + norm_density[v])
        current_usage = edge_n_used[u, v]
        seg_len = segment_lengths[idx]
        segment_multiplier = 1.0 + 0.4 * (seg_len - 1.0)
        usage_factor = 1.0 + 0.2 * np.sqrt(current_usage)
        penalty = base_penalty_scale * (1.0 + 0.35 * density_mod) * usage_factor * segment_multiplier
        updated_edge_distance[u, v] += penalty
        updated_edge_distance[v, u] += penalty

    return updated_edge_distance