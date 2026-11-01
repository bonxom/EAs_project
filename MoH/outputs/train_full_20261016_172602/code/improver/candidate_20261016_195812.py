# Idea: A multi-parent differential ruin-and-recreate metaheuristic that contrasts an elite solution with diverse high-performing peers to diagnose functional differentials, reconstruct complementary algorithmic components, and synthesize hybrid offspring.
# {A multi-parent differential ruin-and-recreate metaheuristic that contrasts an elite solution with diverse high-performing peers to diagnose functional differentials, reconstruct complementary algorithmic components, and synthesize hybrid offspring.}

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

    # Select a diverse peer for differential crossover: prefer rank 1 or 2, or random non-elite if available
    if pop_size > 2:
        candidate_indices = [1, 2, min(pop_size - 1, 3)]
        peer_idx = int(np.random.choice(candidate_indices))
        peer_sol = population.get_solution_by_index(task, peer_idx)
    elif pop_size == 2:
        peer_sol = population.get_solution_by_index(task, 1)
    else:
        peer_sol = elite_sol

    peer_code = peer_sol['best_sol']
    peer_idea = peer_sol.get('idea', 'Alternative peer heuristic')
    peer_utility = peer_sol['utility']

    differential_diagnosis_prompt = (
        f"Perform a Multi-Parent Differential Crossover Analysis for the optimization task '{task}'.\n\n"
        f"--- PARENT A (Global Elite, Utility: {current_best_utility}) ---\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n\n"
        f"--- PARENT B (Diverse High-Performing Peer, Utility: {peer_utility}) ---\n"
        f"Idea: {peer_idea}\n"
        f"Code:\n```python\n{peer_code}\n```\n\n"
        "Comparative Diagnosis:\n"
        "1. Identify the 'Core Invariable Invariants': Shared or unique fundamental mechanisms in Parent A responsible for its elite status.\n"
        "2. Formulate 4 distinct Differential Ruin-and-Recreate plans. Each plan must pinpoint a sub-component in Parent A that is inferior, stagnant, or rigid compared to concepts in Parent B, tear it down ('ruin target'), and recreate it by cross-synthesizing structural advantages from Parent B with novel algorithmic innovations.\n\n"
        "Return your analysis strictly in JSON format matching this schema:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<critical logic to preserve from Parent A>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<sub-component of Parent A to replace>", "crossover_innovation": "<how to synthesize Parent B strengths into the recreated component>"},\n'
        '    {"target_to_ruin": "<sub-component of Parent A to replace>", "crossover_innovation": "<how to synthesize Parent B strengths into the recreated component>"},\n'
        '    {"target_to_ruin": "<sub-component of Parent A to replace>", "crossover_innovation": "<how to synthesize Parent B strengths into the recreated component>"},\n'
        '    {"target_to_ruin": "<sub-component of Parent A to replace>", "crossover_innovation": "<how to synthesize Parent B strengths into the recreated component>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, differential_diagnosis_prompt, temperature=0.7)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core heuristic search progression")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression and elite exploitation backbone"
        operators = [
            {
                "target_to_ruin": "Local perturbation and neighborhood generation operator",
                "crossover_innovation": "Synthesize Parent B's exploratory dispersion with Parent A's structured intensification"
            },
            {
                "target_to_ruin": "Acceptance criteria and cooling/temperature dynamics",
                "crossover_innovation": "Adopt Parent B's escape heuristics coupled with Parent A's greedy refinement"
            },
            {
                "target_to_ruin": "Candidate solution construction and initialization",
                "crossover_innovation": "Hybridize Parent B's randomized adaptive construction with Parent A's heuristic guidance"
            },
            {
                "target_to_ruin": "Stagnation detection and restart strategy",
                "crossover_innovation": "Incorporate Parent B's memory-guided diversity preservation into Parent A's restart loop"
            }
        ]

    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Rigid algorithmic sub-modules")
        crossover_strat = op.get("crossover_innovation", "Differential crossover synthesis of Parents A and B")
        reconstruct_prompt = (
            f"You are executing Differential Ruin-and-Recreate Crossover on Parent A informed by Parent B.\n\n"
            f"PARENT A (Base Elite Code, Score: {current_best_utility}):\n```python\n{current_best_code}\n```\n"
            f"PARENT B (Alternative Peer Code, Score: {peer_utility}):\n```python\n{peer_code}\n```\n\n"
            f"INVARIABLE CORE TO PRESERVE: {invariable_core}\n"
            f"RUIN TARGET IN PARENT A: {ruin_target}\n"
            f"RECREATE VIA CROSSOVER: {crossover_strat}\n\n"
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
