# {Prune dominated lookahead signals through non-dominated Pareto front filtering of 2-step and 3-step potential savings, modulating penalties with asymmetric curvature and usage-damped hop-extensions.}

import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.copy(edge_distance)

    u_nodes = local_opt_tour
    v_nodes = np.roll(local_opt_tour, -1)
    w_nodes = np.roll(local_opt_tour, -2)
    z_nodes = np.roll(local_opt_tour, -3)

    dists_fwd = edge_distance[u_nodes, v_nodes]
    dists_bwd = edge_distance[v_nodes, u_nodes]
    dists_vw = edge_distance[v_nodes, w_nodes]
    dists_wz = edge_distance[w_nodes, z_nodes]
    dists_uw = edge_distance[u_nodes, w_nodes]
    dists_uz = edge_distance[u_nodes, z_nodes]

    usage_fwd = edge_n_used[u_nodes, v_nodes]
    usage_bwd = edge_n_used[v_nodes, u_nodes]
    usage_uw = edge_n_used[u_nodes, w_nodes]
    usage_uz = edge_n_used[u_nodes, z_nodes]

    gain_2step = np.maximum(0.0, (dists_fwd + dists_vw) - dists_uw)
    gain_3step = np.maximum(0.0, (dists_fwd + dists_vw + dists_wz) - (dists_uz + dists_vw))

    # Refined Pareto rank score: combine upper envelope dominance with joint-improvement bonus
    pareto_score = np.maximum(gain_2step, 0.70 * gain_3step) + 0.30 * np.minimum(gain_2step, gain_3step)
    norm_lookahead = pareto_score / (np.max(pareto_score) + 1e-8)

    max_tour_dist = np.max(dists_fwd) + 1e-8
    rel_fwd = dists_fwd / max_tour_dist
    rel_uw = dists_uw / max_tour_dist
    rel_uz = dists_uz / max_tour_dist

    usage_synth = 0.35 * usage_fwd + 0.35 * usage_bwd + 0.20 * usage_uw + 0.10 * usage_uz
    std_fwd = np.std(dists_fwd)
    mean_fwd = np.mean(dists_fwd) + 1e-8
    alpha = np.clip(std_fwd / mean_fwd, 0.15, 0.85)
    effective_usage = alpha * usage_synth + (1.0 - alpha) * np.sqrt(usage_synth)

    curvature = np.abs(dists_fwd + dists_vw - dists_uw) / (dists_fwd + dists_vw + dists_uw + 1e-8)
    dir_variance = np.abs(dists_fwd - dists_bwd) / (dists_fwd + dists_bwd + 1e-8)

    blended_dist = 0.62 * dists_fwd + 0.28 * dists_uw + 0.10 * dists_uz
    blended_rel = 0.62 * rel_fwd + 0.28 * rel_uw + 0.10 * rel_uz

    utilities = (blended_dist * (1.0 + blended_rel) * (1.0 + 0.75 * norm_lookahead)) / (1.0 + effective_usage)
    norm_util = utilities / (np.max(utilities) + 1e-8)

    sigmoid_force = 1.0 / (1.0 + np.exp(-14.0 * (norm_util - 0.5)))
    non_linear_scale = np.power(1.0 + curvature, 2.0) * np.exp(dir_variance)
    base_force = sigmoid_force * non_linear_scale

    force_fwd = base_force * (1.0 + dir_variance)
    force_bwd = base_force * (1.0 - 0.50 * dir_variance)

    bypass_ratio = np.clip(rel_uw / (rel_fwd + 1e-8), 0.0, 2.0)
    bypass_attenuation = 1.0 / (1.0 + bypass_ratio)
    cross_weight = 0.24 * bypass_attenuation / (1.0 + 0.25 * (usage_fwd + usage_bwd))
    force_cross = cross_weight * base_force

    hop3_weight = 0.12 * np.where(gain_3step > gain_2step, 1.0, 0.35) / (1.0 + 0.50 * usage_uz)
    force_hop3 = hop3_weight * base_force

    np.add.at(updated_edge_distance, (u_nodes, v_nodes), force_fwd)
    np.add.at(updated_edge_distance, (v_nodes, u_nodes), force_bwd)
    np.add.at(updated_edge_distance, (u_nodes, w_nodes), force_cross)
    np.add.at(updated_edge_distance, (u_nodes, z_nodes), force_hop3)

    return updated_edge_distance
