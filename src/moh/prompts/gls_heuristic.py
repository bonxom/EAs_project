"""TSP GLS prompts and bounded seed requests."""
from moh.llm.base import LLMRequest

EXPERTISE = 'You design reproducible optimization heuristics with lower gap scores.'
HEURISTIC_FORMAT = '''KIND: heuristic_program
Define update_edge_distance(edge_distance, local_opt_tour, edge_n_used).
Inputs are a distance matrix, a tour without a repeated final city, and a
symmetric penalty count matrix. Return a finite nonnegative symmetric distance
matrix of the same shape with zero diagonal; do not modify inputs.
Return a # {idea} comment and exactly one fenced python code block.
'''


def seed_direction_prompt(task_id, previous_ideas):
    return LLMRequest(EXPERTISE, 'KIND: directions\n'
                      f'Task: {task_id}. Prior ideas: {list(previous_ideas)}. '
                      'Propose different improvements. Return exactly one JSON fence '
                      'containing {"insights": ["idea", ...]}.', 1.0)


def seed_heuristic_prompt(task_id, direction):
    return LLMRequest(EXPERTISE, HEURISTIC_FORMAT +
                      f'\nTask: {task_id}. Direction: {direction}', 1.0)
