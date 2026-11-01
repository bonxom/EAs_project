# Idea: Iterated Local Search (ILS) metaheuristic that performs sequential multi-step refinement loops on elite solutions with adaptive perturbation and targeted local exploitation.
from utils.utils import extract_code, extract_idea
import json

# {Iterated Local Search (ILS) metaheuristic that performs sequential multi-step refinement loops on elite solutions with adaptive perturbation and targeted local exploitation.}

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in the domain of designing meta optimization strategy and combinatorial optimization problems. "
        "Your task is to design heuristics that can effectively solve optimization problems."
    )

    # Select the current best elite solution for exploitation
    elite = population.get_solution_by_index(task, 0)
    current_sol = elite['best_sol']
    current_idea = elite.get('idea', 'Initial elite heuristic')
    current_utility = elite['utility']

    best_idea = current_idea
    best_solution = current_sol
    best_utility = current_utility

    # Iterated Local Search across a bounded number of refinement stages (<= 5)
    num_iterations = 3
    num_candidates_per_iter = 3

    for step in range(num_iterations):
        # 1. Perturbation & Directed Exploration: Prompt for distinct refinement/perturbation angles
        direction_prompt = (
            f"Current best heuristic for task '{task}':\n"
            f"Code: {current_sol}\n"
            f"Idea: {current_idea}\n"
            f"Current utility score: {current_utility} (lower is better).\n"
            f"Propose {num_candidates_per_iter} distinct, high-impact algorithmic refinements or structural perturbations "
            f"that address potential bottlenecks, balance exploration vs exploitation, and reduce utility score.\n"
            f"Your response must be formatted strictly as a json structure:\n"
            f"```json\n{{\"insights\": [\"refinement 1\", \"refinement 2\", \"refinement 3\"]}}\n```"
        )
        try:
            resp_directions = language_model.prompt(expertise, direction_prompt, temperature=0.7)
            parsed = json.loads(extract_code(resp_directions))
            directions = parsed.get("insights", [])[:num_candidates_per_iter]
        except Exception:
            directions = [
                "Refine search parameters and enhance intensification near promising search regions.",
                "Incorporate adaptive diversity preservation mechanisms to prevent stagnation.",
                "Streamline candidate selection logic using greedy-randomized heuristics."
            ]

        # 2. Local Search / Candidate Generation: Generate neighbor solutions in parallel
        message_batch = []
        for direction in directions:
            msg = (
                f"Improve the following solution for task '{task}' via local refinement:\n"
                f"```python\n{current_sol}\n```\n"
                f"Current score: {current_utility}\n"
                f"Target refinement direction: {direction}\n\n"
                f"Follow strictly this format:\n{function_format}\n"
                f"Your response must firstly provide a summary of the key idea inside a brace and marked as a comment, "
                f"followed by the complete, executable Python code implementation.\n"
                f"Remember, lower utility score indicates a superior solution."
            )
            message_batch.append(msg)

        responses = language_model.prompt_batch(expertise, message_batch, temperature=0.7)
        new_solutions = extract_code(responses)
        new_ideas = extract_idea(responses)

        # 3. Evaluation & Acceptance Criterion
        step_best_sol = None
        step_best_idea = None
        step_best_utility = float('inf')

        for idea, sol in zip(new_ideas, new_solutions):
            try:
                score = utility(sol, idea, task)
                if score < step_best_utility:
                    step_best_utility = score
                    step_best_sol = sol
                    step_best_idea = idea
            except Exception:
                continue

        # Update global best if an improvement is found
        if step_best_utility < best_utility:
            best_utility = step_best_utility
            best_solution = step_best_sol
            best_idea = step_best_idea
            # Intensify around the new global optimum
            current_sol = step_best_sol
            current_idea = step_best_idea
            current_utility = step_best_utility
        elif step_best_utility < current_utility:
            # Accept local improvement
            current_sol = step_best_sol
            current_idea = step_best_idea
            current_utility = step_best_utility
        else:
            # Perturb by sampling another solution from population to escape local optima
            alt_solution = population.get_random_solution(task)
            current_sol = alt_solution['best_sol']
            current_idea = alt_solution.get('idea', 'Alternative population solution')
            current_utility = alt_solution['utility']

    return best_idea, best_solution, best_utility
