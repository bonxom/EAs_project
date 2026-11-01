# Idea: A ruin-and-recreate meta-algorithm decomposing the elite solution into strategic operator targets, paired with operator-profiled multi-tier temperature partitioning across parallel LLM prompts to balance precision intensification and disruptive diversification.
#{A ruin-and-recreate meta-algorithm decomposing the elite solution into strategic operator targets, paired with operator-profiled multi-tier temperature partitioning across parallel LLM prompts to balance precision intensification and disruptive diversification.}

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

    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate philosophy across distinct exploration profiles:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies categorized by exploration intent:\n"
        "   - 'micro_intensification': Target precision logic, parameter tuning, or numerical stabilization (Exploitation).\n"
        "   - 'neighborhood_remodel': Redesign neighborhood structures, move evaluations, or local steps.\n"
        "   - 'meta_restructure': Overhaul stagnation handling, memory/tabu mechanisms, or acceptance logic.\n"
        "   - 'macro_diversification': Radical disruption of the global candidate generation or representation (Exploration).\n\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<brief summary of what must be retained>",\n'
        '  "reconstruction_operators": [\n'
        '    {"profile": "micro_intensification", "target_to_ruin": "<component>", "reconstruct_strategy": "<precise refinement>"},\n'
        '    {"profile": "neighborhood_remodel", "target_to_ruin": "<component>", "reconstruct_strategy": "<neighborhood innovation>"},\n'
        '    {"profile": "meta_restructure", "target_to_ruin": "<component>", "reconstruct_strategy": "<control overhaul>"},\n'
        '    {"profile": "macro_diversification", "target_to_ruin": "<component>", "reconstruct_strategy": "<radical topological paradigm>"}\n'
        '  ]\n'
        "}\n"
        "```"
    )

    default_operators = [
        {"profile": "micro_intensification", "target_to_ruin": "Local tuning parameters and precision steps", "reconstruct_strategy": "Analytical convergence damping with fine-grained local refinement"},
        {"profile": "neighborhood_remodel", "target_to_ruin": "Neighborhood perturbation operator", "reconstruct_strategy": "Adaptive variable-depth multi-neighborhood ruin-and-rebuild operator"},
        {"profile": "meta_restructure", "target_to_ruin": "Acceptance and stagnation handling", "reconstruct_strategy": "Non-monotonic adaptive cooling schedule with elite-frequency restart triggers"},
        {"profile": "macro_diversification", "target_to_ruin": "Global candidate generation mechanism", "reconstruct_strategy": "Hybridized greedy-randomized construction with orthogonal structural restarts"}
    ]

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core search pipeline")
        operators = diag_data.get("reconstruction_operators", default_operators)
        if len(operators) < 4:
            operators.extend(default_operators[len(operators):])
    except Exception:
        invariable_core = "Core heuristic search progression"
        operators = default_operators

    profile_temperature_map = {
        "micro_intensification": 0.3,
        "neighborhood_remodel": 0.65,
        "meta_restructure": 0.9,
        "macro_diversification": 1.15
    }
    fallback_temperatures = [0.3, 0.65, 0.9, 1.15]

    grouped_prompts = {}

    for idx, op in enumerate(operators[:4]):
        profile = op.get("profile", None)
        temp = profile_temperature_map.get(profile, fallback_temperatures[idx])
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic component")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")

        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"EXPLORATION TIER: {profile if profile else 'Adaptive Tier'} (Sampling Temperature: {temp})\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )

        grouped_prompts.setdefault(temp, []).append(reconstruct_prompt)

    all_responses = []
    for temp, batch in grouped_prompts.items():
        try:
            if len(batch) == 1:
                resp = language_model.prompt(expertise, batch[0], temperature=temp)
                all_responses.append(resp)
            else:
                resps = language_model.prompt_batch(expertise, batch, temperature=temp)
                all_responses.extend(resps)
        except Exception:
            continue

    candidate_codes = extract_code(all_responses)
    candidate_ideas = extract_idea(all_responses)

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
