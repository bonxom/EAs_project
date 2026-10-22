import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Greedy nearest-neighbour with a one-step look-ahead tie-breaker."""
    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    if unvisited_nodes.size == 0:
        return int(destination_node)

    best_node = int(unvisited_nodes[0])
    best_score = float("inf")
    for node in unvisited_nodes:
        d_now = distance_matrix[current_node][node]
        others = unvisited_nodes[unvisited_nodes != node]
        if others.size:
            d_future = float(np.min(distance_matrix[node][others]))
        else:
            d_future = float(distance_matrix[node][destination_node])
        score = 0.7 * d_now + 0.3 * d_future
        if score < best_score:
            best_score = score
            best_node = int(node)
    return best_node
