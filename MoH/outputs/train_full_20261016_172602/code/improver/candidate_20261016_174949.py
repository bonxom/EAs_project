# Idea: This metaheuristic employs a genetic crossover-and-recombination strategy that selects top-performing parent algorithms from the population, synthesizes their complementary structural components via an LLM meta-recombiner, and generates novel hybrid offspring to exploit high-performing motifs while exploring novel algorithmic synergies.
from utils.utils import extract_code, extract_idea
import json
import random

# {This metaheuristic employs a genetic crossover-and-recombination strategy that selects top-performing parent algorithms from the population, synthesizes their complementary structural components via an LLM meta-recombiner, and generates novel hybrid offspring to exploit high-performing motifs while exploring novel algorithmic synergies.}

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics, evolutionary computation, and algorithm design. "
        "Your task is to analyze, recombine, and optimize heuristic algorithm structures to achieve minimal utility scores."
    )

    pop_size = population.get_subtask_size(task)

    # Select parent solutions: elite exploitation (index 0) and high-performing/diverse alternatives
    parent_indices = sorted(random.sample(range(min(pop_size, 5)), min(pop_size, 3)))
    parents = [population.get_solution_by_index(task, idx) for idx in parent_indices]

    # If the population is too small, supplement with random solutions
    while len(parents) < 2:
        parents.append(population.get_random_solution(task))

    parents_repr = ""
    for idx, p in enumerate(parents):
        parents_repr += f"--- Parent {idx + 1} (Utility: {p['utility']}) ---\n```python\n{p['best_sol']}\n```\n\n"

    crossover_prompt = (
        f"You are given {len(parents)} top-tier algorithmic solutions for the problem '{task}':\n\n"
        f"{parents_repr}\n"
        "Conduct a structural crossover analysis:\n"
        "1. Identify distinct, complementary functional blocks in each parent (e.g., representation, selection mechanisms, neighborhood operators, exploration/exploitation control).\n"
        "2. Formulate 4 distinct recombination hypotheses that recombine the best structural blocks or resolve deficiencies found in parents.\n"
        "Format your output strictly as a JSON object:\n"
        "```json\n"
        "{\n"
        '  "recombinations": [\n'
        '    {"title": "strategy_name", "mechanism": "detailed description of structural crossover and novel components"},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "```"
    )

    response = language_model.prompt(expertise, crossover_prompt, temperature=0.7)
    try:
        recombinations = json.loads(extract_code(response))["recombinations"]
    except Exception:
        recombinations = [
            {"title": "Hybrid Operator", "mechanism": "Combine the exploration operator of Parent 1 with the adaptive control of Parent 2."},
            {"title": "Multi-parent Synergy", "mechanism": "Synthesize structural components across all parents into a unified evolutionary framework."}
        ]

    generation_batch = []
    for plan in recombinations:
        prompt = (
            f"Implement a novel metaheuristic algorithm for task '{task}' by structurally recombining the following parent algorithms:\n\n"
            f"{parents_repr}\n"
            f"Follow this specific recombination strategy: {plan['title']} - {plan['mechanism']}.\n"
            "Ensure the resulting code is self-contained, computationally efficient, avoids infinite loops, and conforms strictly to the format.\n"
            f"You must return an improved solution formatted as follows:\n{function_format}\n"
            "Your response must firstly provide a summary of the key idea inside a brace and marked as a comment, followed by the code implementation.\n"
            "The lower the utility score, the better the solution."
        )
        generation_batch.append(prompt)

    responses = language_model.prompt_batch(expertise, generation_batch, temperature=1.0)
    new_solutions = extract_code(responses)
    new_ideas = extract_idea(responses)

    evaluated_solutions = []
    for idea, code in zip(new_ideas, new_solutions):
        try:
            score = utility(code, idea, task)
            evaluated_solutions.append((idea, code, score))
        except Exception:
            continue

    if not evaluated_solutions:
        # Fallback to the best available parent if evaluation failed
        best_parent = parents[0]
        return best_parent.get('idea', 'Elite parent fallback'), best_parent['best_sol'], best_parent['utility']

    best_idea, best_solution, best_utility = min(evaluated_solutions, key=lambda x: x[2])
    return best_idea, best_solution, best_utility
