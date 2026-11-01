# Idea: This metaheuristic employs a bi-objective Pareto-selection strategy balancing solution quality and behavioral/structural novelty using token Jaccard diversity, followed by multi-parent crossover and LLM-guided mutation along distinct Pareto trade-offs.
from utils.utils import extract_code, extract_idea
import json
import random

# {This metaheuristic employs a bi-objective Pareto-selection strategy balancing solution quality and behavioral/structural novelty using token Jaccard diversity, followed by multi-parent crossover and LLM-guided mutation along distinct Pareto trade-offs.}

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics, evolutionary computation, and combinatorial algorithm design. "
        "Your goal is to synthesize novel, high-performing algorithmic solutions with minimal utility scores."
    )

    pop_size = population.get_subtask_size(task)
    candidates = []

    # Retrieve candidates from population
    indices = sorted(list(set([0, min(1, pop_size - 1), pop_size // 2, pop_size - 1] + [random.randint(0, pop_size - 1) for _ in range(min(4, pop_size))])))
    for idx in indices:
        sol = population.get_solution_by_index(task, idx)
        candidates.append(sol)

    # Token-set representation to compute Jaccard diversity distance
    def tokenize(code_str):
        return set(code_str.split())

    token_sets = [tokenize(c['best_sol']) for c in candidates]

    # Compute average pairwise Jaccard distance as diversity metric (higher is more diverse/novel)
    diversity_scores = []
    for i, t1 in enumerate(token_sets):
        dist_sum = 0.0
        count = 0
        for j, t2 in enumerate(token_sets):
            if i != j:
                inter = len(t1.intersection(t2))
                union = len(t1.union(t2))
                dist = 1.0 - (inter / union if union > 0 else 0.0)
                dist_sum += dist
                count += 1
        diversity_scores.append(dist_sum / count if count > 0 else 0.0)

    # Non-dominated Pareto frontier extraction (Objectives: Minimize Utility, Maximize Diversity)
    pareto_frontier = []
    for i in range(len(candidates)):
        dominated = False
        u_i = candidates[i]['utility']
        d_i = diversity_scores[i]
        for j in range(len(candidates)):
            if i != j:
                u_j = candidates[j]['utility']
                d_j = diversity_scores[j]
                # Dominance check: j is at least as good in both and strictly better in one
                if (u_j <= u_i and d_j >= d_i) and (u_j < u_i or d_j > d_i):
                    dominated = True
                    break
        if not dominated:
            pareto_frontier.append(candidates[i])

    if not pareto_frontier:
        pareto_frontier = [candidates[0]]

    # Strategy synthesis: Formulate diverse evolutionary prompts across the Pareto frontier
    # Best-performing elite for exploitation, most diverse for exploration
    best_elite = min(candidates, key=lambda x: x['utility'])
    most_diverse = max(zip(candidates, diversity_scores), key=lambda x: x[1])[0]

    direction_prompt = (
        f"We are optimizing algorithms for problem '{task}'.\n"
        f"Elite exploitation candidate (Utility: {best_elite['utility']}):\n```python\n{best_elite['best_sol']}\n```\n"
        f"High-diversity candidate (Utility: {most_diverse['utility']}):\n```python\n{most_diverse['best_sol']}\n```\n"
        "Analyze the architectural trade-offs between performance and diversity. Formulate 3 distinct conceptual "
        "hybridization or mutation directions to systematically dominate this frontier.\n"
        "Format your answer as valid JSON:\n"
        "```json\n{\n  \"directions\": [\n    \"direction 1...\",\n    \"direction 2...\",\n    \"direction 3...\"\n  ]\n}\n```"
    )

    dir_response = language_model.prompt(expertise, direction_prompt, temperature=0.7)
    try:
        directions = json.loads(extract_code(dir_response))["directions"]
    except Exception:
        directions = [
            "Integrate adaptive parameter control into the elite candidate.",
            "Synthesize structural logic from the high-diversity candidate into the elite solver.",
            "Formulate an orthogonal local search heuristic balancing intensification and diversification."
        ]

    # Generate next-generation candidates guided by Pareto directions
    batch_prompts = []
    for d in directions[:3]:
        prompt = (
            f"Improve the optimization algorithm for task '{task}'.\n"
            f"Primary Foundation Code:\n```python\n{best_elite['best_sol']}\n```\n"
            f"Secondary Novelty Archetype:\n```python\n{most_diverse['best_sol']}\n```\n"
            f"Strategic Direction: {d}\n\n"
            f"Requirements:\n"
            f"1. Generate an improved heuristic algorithm under the required format:\n{function_format}\n"
            f"2. Your response must first state a concise description inside a brace marked as a comment: # {{description}}\n"
            f"3. Follow with pure python implementation, strictly adhering to the signature and constraints.\n"
        )
        batch_prompts.append(prompt)

    responses = language_model.prompt_batch(expertise, batch_prompts, temperature=0.7)
    new_solutions = extract_code(responses)
    new_ideas = extract_idea(responses)

    evaluated_candidates = []
    for idea, solution in zip(new_ideas, new_solutions):
        try:
            score = utility(solution, idea, task)
            evaluated_candidates.append((idea, solution, score))
        except Exception:
            continue

    if evaluated_candidates:
        best_idea, best_solution, best_utility = min(evaluated_candidates, key=lambda x: x[2])
    else:
        best_idea = best_elite.get('idea', 'Pareto-retained baseline')
        best_solution = best_elite['best_sol']
        best_utility = best_elite['utility']

    return best_idea, best_solution, best_utility
