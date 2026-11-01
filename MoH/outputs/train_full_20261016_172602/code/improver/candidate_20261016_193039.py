# Idea: A population-aware path-relinking and differential crossover metaheuristic that diagnoses shared invariants versus divergent algorithmic traits between elite and diverse reference solutions to generate hybrid, complementary reconstruction candidates via parallel LLM synthesis.
#{A population-aware path-relinking and differential crossover metaheuristic that diagnoses shared invariants versus divergent algorithmic traits between elite and diverse reference solutions to generate hybrid, complementary reconstruction candidates via parallel LLM synthesis.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    pop_size = population.get_subtask_size(task)
    elite_sol = population.get_solution_by_index(task, 0)
    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # Select complementary reference solutions for differential crossover / path-relinking
    if pop_size > 2:
        ref_sol_1 = population.get_solution_by_index(task, 1)
        ref_sol_2 = population.get_random_solution(task)
    elif pop_size == 2:
        ref_sol_1 = population.get_solution_by_index(task, 1)
        ref_sol_2 = ref_sol_1
    else:
        ref_sol_1 = elite_sol
        ref_sol_2 = elite_sol

    ref_code_1 = ref_sol_1['best_sol']
    ref_utility_1 = ref_sol_1.get('utility', 'N/A')
    ref_code_2 = ref_sol_2['best_sol']
    ref_utility_2 = ref_sol_2.get('utility', 'N/A')

    # Diagnostic Phase: Identify invariants, divergent building blocks, and crossover trajectories
    diagnosis_prompt = (
        f"You are conducting a differential crossover and path-relinking analysis between an elite solution and diverse reference solutions for '{task}'.\n\n"
        f"--- ELITE SOLUTION (Utility: {current_best_utility}) ---\n"
        f"```python\n{current_best_code}\n```\n\n"
        f"--- REFERENCE SOLUTION 1 (Utility: {ref_utility_1}) ---\n"
        f"```python\n{ref_code_1}\n```\n\n"
        f"--- REFERENCE SOLUTION 2 (Utility: {ref_utility_2}) ---\n"
        f"```python\n{ref_code_2}\n```\n\n"
        "Perform Path-Relinking and Differential Structural Analysis:\n"
        "1. Identify the 'Shared Invariants': algorithmic primitives, representations, or invariant core components that are common and beneficial.\n"
        "2. Identify 'Divergent Traits': orthogonal or contrasting mechanisms (e.g., perturbation strategies, acceptance criteria, exploitation vs. exploration balances).\n"
        "3. Propose 4 distinct hybrid recombination directions (path-relinking intermediate states or differential mutations):\n"
        "   - Direction 1 (Forward Relinking): Inject the strongest innovative mechanism from Reference 1 into the Elite invariant backbone.\n"
        "   - Direction 2 (Differential Synthesis): Graft an alternative structural mutation/exploration trait from Reference 2 onto the Elite while stripping suboptimal bottlenecks.\n"
        "   - Direction 3 (Orthogonal Fusion): Synthesize a hybrid operator reconciling opposing search dynamics from Elite and Reference 1.\n"
        "   - Direction 4 (Extrapolative Innovation): Step beyond both solutions along the trajectory of difference between Elite and References to introduce an upgraded, advanced metaheuristic operator.\n\n"
        "Return your response strictly in valid JSON format:\n"
        "```json\n"
        "{\n"
        '  "shared_invariants": "<core invariant principles>",\n'
        '  "crossover_operators": [\n'
        '    {"strategy_name": "<name>", "description": "<precise fusion instructions>"},\n'
        '    {"strategy_name": "<name>", "description": "<precise fusion instructions>"},\n'
        '    {"strategy_name": "<name>", "description": "<precise fusion instructions>"},\n'
        '    {"strategy_name": "<name>", "description": "<precise fusion instructions>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        shared_invariants = diag_data.get("shared_invariants", "Preserve high-performing search structures and convergence control")
        operators = diag_data.get("crossover_operators", [])
    except Exception:
        shared_invariants = "Retain proven algorithmic backbone and efficient candidate scoring"
        operators = [
            {"strategy_name": "Forward Relinking", "description": "Graft high-performing exploration operators from reference solution into the elite core."},
            {"strategy_name": "Differential Synthesis", "description": "Replace elite stagnating neighborhoods with diversified operators extracted from reference solutions."},
            {"strategy_name": "Orthogonal Fusion", "description": "Combine the intensification mechanics of the elite with the diversification mechanisms of reference solutions."},
            {"strategy_name": "Extrapolative Innovation", "description": "Formulate an advanced adaptive ruin-and-recreate operator extending beyond both solutions."}
        ]

    # Reconstruct Phase: Parallel batched generation along distinct recombination paths
    message_batch = []
    for op in operators[:4]:
        name = op.get("strategy_name", "Hybrid Recombination")
        desc = op.get("description", "Recombine invariant core with diverse features")

        reconstruct_prompt = (
            f"Implement a high-performance optimization algorithm for task '{task}' via Path-Relinking & Differential Crossover.\n\n"
            f"--- PARENT 1 (ELITE, Utility: {current_best_utility}) ---\n"
            f"```python\n{current_best_code}\n```\n\n"
            f"--- PARENT 2 (REFERENCE, Utility: {ref_utility_1}) ---\n"
            f"```python\n{ref_code_1}\n```\n\n"
            f"SHARED INVARIANTS TO MAINTAIN:\n{shared_invariants}\n\n"
            f"RECOMBINATION DIRECTIVE ({name}):\n{desc}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    # Evaluation and greedy elitist selection
    candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            if score is not None and not np.isnan(score):
                candidates.append((idea, code, score))
        except Exception:
            continue

    if candidates:
        iter_best_idea, iter_best_code, iter_best_utility = min(candidates, key=lambda x: x[2])
        if iter_best_utility < current_best_utility:
            current_best_idea = iter_best_idea
            current_best_code = iter_best_code
            current_best_utility = iter_best_utility

    return current_best_idea, current_best_code, current_best_utility
