# Idea: Multi-parent differential contrastive diagnosis extracts positive invariant building blocks and negative motifs by contrasting an elite solution with an inferior or diverse counterpart, directing targeted ruin-and-recreate perturbations across multiple parallel search trajectories.
#{Multi-parent differential contrastive diagnosis extracts positive invariant building blocks and negative motifs by contrasting an elite solution with an inferior or diverse counterpart, directing targeted ruin-and-recreate perturbations across multiple parallel search trajectories.}

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

    # Select a contrasting reference solution (e.g., lower-ranked or random non-elite)
    if pop_size > 2:
        contrast_sol = population.get_solution_by_index(task, pop_size - 1)
    elif pop_size == 2:
        contrast_sol = population.get_solution_by_index(task, 1)
    else:
        contrast_sol = population.get_random_solution(task)

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline metaheuristic')
    current_best_utility = elite_sol['utility']

    contrast_code = contrast_sol.get('best_sol', current_best_code)
    contrast_idea = contrast_sol.get('idea', 'Inferior/diverse baseline candidate')
    contrast_utility = contrast_sol.get('utility', current_best_utility)

    # Multi-parent differential contrastive diagnosis prompt
    contrastive_diagnosis_prompt = (
        f"Perform a Multi-Parent Differential Contrastive Diagnosis on two algorithms for the task '{task}':\n\n"
        f"--- Parent A (Superior Elite, Utility: {current_best_utility} [lower is better]) ---\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n\n"
        f"--- Parent B (Contrast Reference, Utility: {contrast_utility} [lower is better]) ---\n"
        f"Idea: {contrast_idea}\n"
        f"Code:\n```python\n{contrast_code}\n```\n\n"
        "Comparative Analysis Steps:\n"
        "1. Identify 'Invariant Building Blocks': Algorithmic components that account for Parent A's superior performance and must be preserved.\n"
        "2. Identify 'Negative Motifs / Bottlenecks': Sub-optimal patterns, stagnations, or redundancies in both solutions that must be eliminated.\n"
        "3. Synthesize 4 distinct, radical 'Ruin-and-Recreate' perturbation plans that tear down identified negative motifs and rebuild them via novel metaheuristic operations.\n\n"
        "Return your analysis strictly in valid JSON format:\n"
        "```json\n"
        "{\n"
        '  "invariant_core": "<critical high-utility components to strictly preserve>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific negative motif to excise>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific negative motif to excise>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific negative motif to excise>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific negative motif to excise>", "reconstruct_strategy": "<innovative recreation mechanism>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, contrastive_diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariant_core = diag_data.get("invariant_core", "Core adaptive search structure of Parent A")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariant_core = "Core intensification and elite progression logic"
        operators = [
            {"target_to_ruin": "Stagnant local search and neighborhood transitions", "reconstruct_strategy": "Adaptive variable neighborhood search with ruin-recreate kicks"},
            {"target_to_ruin": "Greedy exploitation and rigid acceptance criteria", "reconstruct_strategy": "Non-monotonic adaptive simulated annealing with tabu memory"},
            {"target_to_ruin": "Single-point perturbation operators", "reconstruct_strategy": "Multi-parent differential recombination with guided structural mutation"},
            {"target_to_ruin": "Static exploration parameters", "reconstruct_strategy": "Self-tuning feedback-driven exploration-exploitation balance"}
        ]

    # Reconstruct candidate generation batch
    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic components")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced metaheuristic operator")
        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Differential Ruin-and-Recreate on the task '{task}'.\n\n"
            f"Reference Elite Code:\n```python\n{current_best_code}\n```\n"
            f"Current Best Utility: {current_best_utility} (lower is better)\n\n"
            f"PRESERVE CORE INVARIANTS: {invariant_core}\n"
            f"RUIN NEGATIVE MOTIF: Completely strip away and discard: {ruin_target}\n"
            f"RECONSTRUCT WITH: {rebuild_strat}\n\n"
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
