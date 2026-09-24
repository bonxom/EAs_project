NEAREST_NEIGHBOR_SOURCE = """def select_next_node(current_node, unvisited, coordinates):
    return min(unvisited, key=lambda j: (
        (coordinates[current_node][0] - coordinates[j][0]) ** 2
        + (coordinates[current_node][1] - coordinates[j][1]) ** 2,
        j,
    ))
"""

RANDOM_CHOICE_SOURCE = """def select_next_node(current_node, unvisited, coordinates):
    return random.choice(sorted(unvisited))
"""
