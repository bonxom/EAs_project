# Idea: Dynamic temperature and ruin-intensity scaling based on population diversity and stagnation metrics drive an adaptive diagnostic decomposition and parallel ruin-and-recreate code synthesis pipeline.
#{Dynamic temperature and ruin-intensity scaling based on population diversity and stagnation metrics drive an adaptive diagnostic decomposition and parallel ruin-and-recreate code synthesis pipeline.}

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

    # Assess population diversity and stagnation signals
    sampled_utilities = [elite_sol['utility']]
    sample_count = min(pop_size, 5)
    for i in range(1, sample_count):
        sampled_utilities.append(population.get_solution_by_index(task, i)['utility'])

    # Calculate utility variance / stagnation metric
    u_std = float(np.std(sampled_utilities)) if len(sampled_utilities) > 1 else 0.0
    u_mean = float(np.mean(sampled_utilities)) if len(sampled_utilities) > 0 else 1.0
    cv = (u_std / (abs(u_mean) + 1e-8))  # coefficient of variation

    # If coefficient of variation is low, search is stagnating -> scale up entropy/ruin
    stagnation_factor = float(np.clip(1.0 - cv * 5.0, 0.0, 1.0))
    diag_temp = round(0.5 + 0.4 * stagnation_factor, 2)
    gen_temp = round(0.7 + 0.5 * stagnation_factor, 2)
    ruin_severity = "severe and disruptive structural overhaul" if stagnation_factor > 0.5 else "balanced surgical perturbation"

    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n"
        f"Current Search State: Stagnation Index = {stagnation_factor:.2f}. Required Ruin Level: {ruin_severity}.\n\n"
        "Apply an Adaptive Ruin-and-Recreate (Iterated Local Search) philosophy:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Propose 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components "
        "(e.g., representation, neighborhood exploration, selection pressure, or stagnation handling) and rebuild them with superior algorithmic logic.\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<brief summary of what must be retained>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>", "ruin_intensity": "<moderate|aggressive>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>", "ruin_intensity": "<moderate|aggressive>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>", "ruin_intensity": "<moderate|aggressive>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>", "ruin_intensity": "<moderate|aggressive>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=diag_temp)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core heuristic search progression")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression"
        operators = [
            {"target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator", "ruin_intensity": "aggressive"},
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Simulated annealing cooling schedule with history-based restarts", "ruin_intensity": "moderate"},
            {"target_to_ruin": "Candidate generation", "reconstruct_strategy": "Guided greedy randomized adaptive construction", "ruin_intensity": "moderate"},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Heavy structural mutation with memory-guided repair", "ruin_intensity": "aggressive"}
        ]

    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic blocks")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")
        op_intensity = op.get("ruin_intensity", "moderate")

        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n"
            f"Search Regime: Intensity={op_intensity}, Stagnation Factor={stagnation_factor:.2f}\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=gen_temp)
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
