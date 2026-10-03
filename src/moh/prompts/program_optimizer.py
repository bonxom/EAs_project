"""Provider-independent optimizer program contract."""
OPTIMIZER_FORMAT = '''KIND: optimizer_program
Define improve_algorithm(population, utility, language_model, function_format, task).
Return (best_idea, best_solution, best_utility), minimizing utility.
population.get_random_solution(task) and population.get_best_solution(task) return
{"idea": str, "best_sol": source, "utility": float}.
language_model.prompt(expertise, message, temperature=None) returns a string;
request directions with first line KIND: directions, returning a JSON fence
with {"insights": [str, ...]}.
language_model.prompt_batch(expertise, messages, temperature=None) returns ordered
responses. Limit messages to language_model.batch_size.
Code requests must begin with the first line of the received function_format.
utility(source, idea, task) returns a finite score; lower is better.
Use moh.optimizers.helpers.extract_code and extract_idea to parse responses.
Return a # {idea} comment and exactly one fenced python code block.
'''
