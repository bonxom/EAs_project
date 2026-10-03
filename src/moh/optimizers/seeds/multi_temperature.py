"""Adapted MoH extended seed, with ascending score order and evaluation cache."""
import json

from moh.optimizers.helpers import extract_code, extract_idea


def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = 'You design optimization algorithms minimizing utility.'
    cache = {}
    selected = []
    for temperature in [0.7, 1.0]:
        parent = population.get_random_solution(task)
        message = ('KIND: directions\n'
                   f'Improve {parent["best_sol"]} with idea {parent["idea"]}. '
                   'Return a JSON fence containing {"insights": ["idea", ...]}.')
        directions = json.loads(extract_code(language_model.prompt(
            expertise, message, temperature=temperature)))['insights']
        messages = [function_format.split('\n', 1)[0] + '\n' + function_format +
                    f'\nParent: {parent["best_sol"]}\nDirection: {direction}\n'
                    'Minimize utility. Return # {idea} then a python fence.'
                    for direction in directions[:language_model.batch_size]]
        responses = language_model.prompt_batch(expertise, messages, temperature=temperature)
        scored = []
        for idea, source in zip(extract_idea(responses), extract_code(responses)):
            if source not in cache:
                cache[source] = (idea, source, utility(source, idea, task))
            scored.append(cache[source])
        selected.extend(sorted(scored, key=lambda entry: entry[2])[:language_model.batch_size])
    return min(selected, key=lambda entry: entry[2])
