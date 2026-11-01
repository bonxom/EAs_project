# Idea: An adaptive entropy-annealing Iterated Local Search that dynamically scales LLM sampling temperatures and exploration depth based on population utility variance and stagnation distance to systematically orchestrate ruin-and-recreate perturbations.
#{An adaptive entropy-annealing Iterated Local Search that dynamically scales LLM sampling temperatures and exploration depth based on population utility variance and stagnation distance to systematically orchestrate ruin-and-recreate perturbations.}

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

    # Calculate population utility diversity and stagnation metrics
    utilities = []
    for i in range(min(pop_size, 5)):
        sol = population.get_solution_by_index(task, i)
        if sol and 'utility' in sol and sol['utility'] is not None:
            utilities.append(sol['utility'])

    if len(utilities) > 1:
        u_std = float(np.std(utilities))
        u_mean = float(np.mean(utilities))
        coeff_var = u_std / (abs(u_mean) + 1e-9)
    else:
        coeff_var = 0.0

    # Stagnation proxy: low variance indicates population convergence / stagnation
    stagnation_factor = float(np.clip(1.0 - coeff_var * 2.0, 0.1, 1.0))

    # Adaptive diagnosis prompt assessing core invariants vs high-leverage ruin targets
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n"
        f"Convergence Stagnation Level: {stagnation_factor:.2f} (1.0 = highly stagnated/clustered, 0.1 = high natural diversity)\n\n"
        "Apply an Adaptive Ruin-and-Recreate philosophy:\n"
        "1. Identify the 'invariable_core': Algorithmic backbone that guarantees correctness and efficiency.\n"
        "2. Define 4 ranked ruin-and-recreate operators transitioning systematically from focused exploitation to radical structural recreation.\n"
        "Return your analysis strictly in JSON format:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<invariable core mechanisms>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<localized inefficient component>", "reconstruct_strategy": "<precise intensification rebuild>", "radicalness": 0.2},\n'
        '    {"target_to_ruin": "<sub-optimal mechanism>", "reconstruct_strategy": "<balanced adaptive rebuild>", "radicalness": 0.5},\n'
        '    {"target_to_ruin": "<restrictive heuristic block>", "reconstruct_strategy": "<explorative structural rebuild>", "radicalness": 0.8},\n'
        '    {"target_to_ruin": "<entire perturbation/search logic>", "reconstruct_strategy": "<radical paradigm shift rebuild>", "radicalness": 1.0}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core search pipeline and validity checks")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression"
        operators = [
            {"target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator", "radicalness": 0.25},
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Simulated annealing cooling schedule with history-based restarts", "radicalness": 0.5},
            {"target_to_ruin": "Candidate generation", "reconstruct_strategy": "Guided greedy randomized adaptive construction", "radicalness": 0.75},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Heavy structural mutation with memory-guided repair", "radicalness": 1.0}
        ]

    # Dynamically scale temperature and exploration intensity per candidate
    message_batch = []
    temperatures = []

    # Base annealing range adjusted by stagnation:
    # High stagnation shifts the baseline temperature higher to escape local optima
    min_temp = 0.4 + 0.3 * stagnation_factor
    max_temp = 0.8 + 0.4 * stagnation_factor

    for idx, op in enumerate(operators[:4]):
        ruin_target = op.get("target_to_ruin", "Local search routine")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")
        radicalness = float(op.get("radicalness", (idx + 1) / 4.0))

        # Adaptive temperature scaling per candidate operator
        temp = min_temp + (max_temp - min_temp) * radicalness
        temp = float(np.clip(temp, 0.3, 1.2))
        temperatures.append(temp)

        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n"
            f"Exploration Radicalness Level: {radicalness:.2f}\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    # Execute prompt batch using the median/representative annealed temperature
    mean_annealed_temp = float(np.mean(temperatures)) if temperatures else 0.85
    responses = language_model.prompt_batch(expertise, message_batch, temperature=mean_annealed_temp)

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
