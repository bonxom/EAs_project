# Idea: A multi-parent differential ruin-and-recreate metaheuristic that extracts structural contrasts between elite and reference solutions (best vs. median/diverse) to identify functional discrepancies and guide parallel targeted recreation of damaged sub-components.
#{A multi-parent differential ruin-and-recreate metaheuristic that extracts structural contrasts between elite and reference solutions (best vs. median/diverse) to identify functional discrepancies and guide parallel targeted recreation of damaged sub-components.}

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

    # Differential parent selection: Elite (Base), Secondary Elite / Median (Differential Vector)
    median_idx = max(0, pop_size // 2)
    secondary_sol = population.get_solution_by_index(task, median_idx) if pop_size > 1 else elite_sol
    exploratory_sol = population.get_random_solution(task) if pop_size > 2 else secondary_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    ref_code_1 = secondary_sol.get('best_sol', current_best_code)
    ref_util_1 = secondary_sol.get('utility', current_best_utility)

    ref_code_2 = exploratory_sol.get('best_sol', current_best_code)
    ref_util_2 = exploratory_sol.get('utility', current_best_utility)

    # Multi-Parent Differential Diagnosis: Contrast elite against diverse reference points
    differential_diagnosis_prompt = (
        f"Perform a Multi-Parent Differential Analysis for optimization task '{task}'.\n\n"
        f"--- Primary Elite Solution (Utility: {current_best_utility}) ---\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n\n"
        f"--- Reference Solution A [Median/Sub-elite] (Utility: {ref_util_1}) ---\n"
        f"Code:\n```python\n{ref_code_1}\n```\n\n"
        f"--- Reference Solution B [Exploratory/Random] (Utility: {ref_util_2}) ---\n"
        f"Code:\n```python\n{ref_code_2}\n```\n\n"
        "Instructions:\n"
        "1. Identify the 'Invariable Core': High-performance architectural patterns in the Primary Elite that must be preserved.\n"
        "2. Conduct a 'Differential Discrepancy Analysis': Identify structural contrasts, omitted mechanisms, or orthogonal search dynamics "
        "present in Reference A or B (or missing in Elite) that could eliminate Elite's bottlenecks.\n"
        "3. Specify 4 targeted 'Ruin-and-Recreate' directives where suboptimal logic in the Elite is dismantled and reconstructed "
        "by synthesizing superior differential traits from the population.\n\n"
        "Return strictly in JSON format:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<critical logic to preserve>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific elite weakness to remove>", "reconstruct_strategy": "<crossover-driven rebuild strategy>"},\n'
        '    {"target_to_ruin": "<specific elite weakness to remove>", "reconstruct_strategy": "<crossover-driven rebuild strategy>"},\n'
        '    {"target_to_ruin": "<specific elite weakness to remove>", "reconstruct_strategy": "<crossover-driven rebuild strategy>"},\n'
        '    {"target_to_ruin": "<specific elite weakness to remove>", "reconstruct_strategy": "<crossover-driven rebuild strategy>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, differential_diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core algorithmic pipeline")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic progression and elite search pipeline"
        operators = [
            {"target_to_ruin": "Single-point local perturbation", "reconstruct_strategy": "Differential multi-vector perturbation leveraging population diversity"},
            {"target_to_ruin": "Static parameterization and acceptance", "reconstruct_strategy": "Adaptive fitness-distance balanced acceptance mechanism"},
            {"target_to_ruin": "Exploration-exploitation transition", "reconstruct_strategy": "Dual-phase global diversification with memory-guided local repair"},
            {"target_to_ruin": "Premature stagnation recovery", "reconstruct_strategy": "Orthogonal ruin-recreate perturbation extracting alternative reference features"}
        ]

    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Rigid algorithmic components")
        rebuild_strat = op.get("reconstruct_strategy", "Differential hybrid recombination")
        reconstruct_prompt = (
            f"You are executing an Advanced Differential Ruin-and-Recreate operation on the following primary elite code:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"INVARIABLE CORE (MUST PRESERVE): {invariable_core}\n"
            f"RUIN TARGET: Completely strip out: {ruin_target}\n"
            f"RECREATE USING DIFFERENTIAL SYNTHESIS: {rebuild_strat}\n\n"
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
