"""Trusted source templates; execute generated variants only in GLS workers."""

IDENTITY_SOURCE = """import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    return np.array(edge_distance, dtype=float, copy=True)
"""

PENALTY_SOURCE = """import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated = np.array(edge_distance, dtype=float, copy=True)
    scale = float(np.mean(edge_distance))
    for i, city in enumerate(local_opt_tour):
        successor = local_opt_tour[(i + 1) % len(local_opt_tour)]
        adjustment = scale / (1.0 + edge_n_used[city, successor])
        updated[city, successor] += adjustment
        updated[successor, city] += adjustment
    return updated
"""

# Adapted from MoH problems/tsp_gls/gpt.py; see third_party/moh.
UPSTREAM_BASELINE_SOURCE = """import numpy as np

# {This algorithm uses Particle Swarm Optimization for exploration, Simulated Annealing for local refinement, and Bayesian Optimization for parameter tuning to update the edge distance matrix, thus preventing trapping in a local optimum and aiding in discovering a tour with minimized distance.}

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Create a copy of the edge distance matrix to avoid modifying the original input directly
    updated_edge_distance = np.copy(edge_distance)
    
    # Calculate the current tour distance
    current_tour_distance = sum(
        edge_distance[local_opt_tour[i], local_opt_tour[i+1]] 
        for i in range(len(local_opt_tour) - 1)
    ) + edge_distance[local_opt_tour[-1], local_opt_tour[0]]
    
    # Initialize evaporation_rate, diversity_penalty, and oscillation_factor
    evaporation_rate = 0.1
    diversity_penalty = 0.05
    oscillation_factor = 0.2
    
    # Reduce the edge distance matrix values by evaporation_rate
    updated_edge_distance *= (1 - evaporation_rate)
    
    # Reinforce the edges used in the local optimal tour
    for i in range(len(local_opt_tour) - 1):
        u, v = local_opt_tour[i], local_opt_tour[i+1]
        reinforcement = current_tour_distance / (1 + edge_n_used[u, v])
        updated_edge_distance[u, v] += reinforcement
        updated_edge_distance[v, u] += reinforcement
    
    # Handle the last edge connecting back to the start
    u, v = local_opt_tour[-1], local_opt_tour[0]
    reinforcement = current_tour_distance / (1 + edge_n_used[u, v])
    updated_edge_distance[u, v] += reinforcement
    updated_edge_distance[v, u] += reinforcement
    
    # Apply diversity penalty to promote exploration
    updated_edge_distance -= diversity_penalty * np.log1p(edge_n_used)
    
    # Apply oscillation factor based on edge usage to enhance flexibility
    updated_edge_distance += oscillation_factor / (1 + edge_n_used)
    
    # Ensure no negative distances, maintaining non-negativity
    updated_edge_distance = np.maximum(updated_edge_distance, 0)
    
    return updated_edge_distance
"""
