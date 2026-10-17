# Idea: Multi-parent differential ruin-and-recreate diagnoses structural invariants across elite and diverse population exemplars to pinpoint destructive divergence points and synthesize recombined, high-performing metaheuristics via batched exploration.
#{Multi-parent differential ruin-and-recreate diagnoses structural invariants across elite and diverse population exemplars to pinpoint destructive divergence points and synthesize recombined, high-performing metaheuristics via batched exploration.}

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

    # Select complementary reference solutions for differential structural comparison
    # 1. Runner-up / Exploitative reference
    ref_idx = min(1, pop_size - 1) if pop_size > 1 else 0
    ref_sol = population.get_solution_by_index(task, ref_idx)

    # 2. Explorative / Diverse random reference
    rand_sol = population.get_random_solution(task) if pop_size > 2 else ref_sol

    # Multi-parent differential diagnosis: compare elite vs reference solutions
    differential_diagnosis_prompt = (
        f"Perform a Multi-Parent Differential Ruin-and-Recreate diagnosis for the optimization task '{task}'.\n\n"
        f"--- PARENT A (Elite Solution, Utility: {current_best_utility:.6f}) ---\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n\n"
        f"--- PARENT B (Comparative Reference, Utility: {ref_sol.get('utility', 'N/A')}) ---\n"
        f"Code:\n```python\n{ref_sol.get('best_sol', '')}\n```\n\n"
        f"--- PARENT C (Exploratory Reference, Utility: {rand_sol.get('utility', 'N/A')}) ---\n"
        f"Code:\n```python\n{rand_sol.get('best_sol', '')}\n```\n\n"
        "Instructions:\n"
        "1. Identify the 'Invariant Core': The essential algorithmic mechanisms shared or proven effective across high-performing parents.\n"
        "2. Detect 'Divergent Bottlenecks': Structural divergence points where suboptimal parents fail, or where the elite solution has latent weaknesses.\n"
        "3. Propose 4 distinct Crossover-Guided Ruin-and-Recreate operators that excise a specific bottleneck from Parent A and rebuild it by hybridizing or innovating upon the best features of Parents B/C.\n\n"
        "Return your analysis strictly in valid JSON format:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<synthesized essential invariant logic>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific bottleneck component in Parent A>", "reconstruct_strategy": "<crossover-guided innovative rebuild strategy>"},\n'
        '    {"target_to_ruin": "<specific bottleneck component in Parent A>", "reconstruct_strategy": "<crossover-guided innovative rebuild strategy>"},\n'
        '    {"target_to_ruin": "<specific bottleneck component in Parent A>", "reconstruct_strategy": "<crossover-guided innovative rebuild strategy>"},\n'
        '    {"target_to_ruin": "<specific bottleneck component in Parent A>", "reconstruct_strategy": "<crossover-guided innovative rebuild strategy>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, differential_diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Adaptive search progression and core objective evaluation")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search and evaluation loop"
        operators = [
            {"target_to_ruin": "Single-point perturbation", "reconstruct_strategy": "Multi-parent differential crossover operator with adaptive mutation bounds"},
            {"target_to_ruin": "Greedy selection criterion", "reconstruct_strategy": "Pareto or energy-based probabilistic acceptance with adaptive cooling"},
            {"target_to_ruin": "Static neighborhood exploration", "reconstruct_strategy": "Adaptive multi-neighborhood search guided by differential diversity"},
            {"target_to_ruin": "Stagnation recovery", "reconstruct_strategy": "Opposition-based learning and ruin-and-repair dispersion"}
        ]

    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic component")
        rebuild_strat = op.get("reconstruct_strategy", "Differential hybrid rebuild mechanism")

        reconstruct_prompt = (
            f"You are executing an advanced Multi-Parent Differential Ruin-and-Recreate step.\n"
            f"Base Elite Code:\n```python\n{current_best_code}\n```\n"
            f"Base Utility: {current_best_utility:.6f} (lower is better)\n\n"
            f"PRESERVE INVARIANT CORE: {invariable_core}\n"
            f"RUIN TARGET: Completely tear down and replace: {ruin_target}\n"
            f"RECREATE USING DIFFERENTIAL CROSSOVER: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
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
