# Idea: A multi-parent differential ruin-and-recombination metaheuristic that diagnoses systemic failure modes and unique strengths by contrasting elite, runner-up, and exploratory solutions, systematically dismantling flawed components while cross-pollinating high-performing algorithmic building blocks.
# {A multi-parent differential ruin-and-recombination metaheuristic that diagnoses systemic failure modes and unique strengths by contrasting elite, runner-up, and exploratory solutions, systematically dismantling flawed components while cross-pollinating high-performing algorithmic building blocks.}

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

    # Select differential multi-parent references: runner-up (exploitation/refinement) and distant/exploratory (diversity)
    if pop_size > 1:
        ref_runner_up = population.get_solution_by_index(task, 1)
    else:
        ref_runner_up = elite_sol

    if pop_size > 2:
        ref_exploratory = population.get_solution_by_index(task, pop_size - 1)
    else:
        ref_exploratory = population.get_random_solution(task)

    runner_up_code = ref_runner_up['best_sol']
    runner_up_idea = ref_runner_up.get('idea', 'Secondary reference heuristic')
    runner_up_util = ref_runner_up['utility']

    exploratory_code = ref_exploratory['best_sol']
    exploratory_idea = ref_exploratory.get('idea', 'Exploratory diversity heuristic')
    exploratory_util = ref_exploratory['utility']

    # Differential Diagnostic Reflection across the multi-parent population
    diagnosis_prompt = (
        f"Analyze the population landscape for task '{task}':\n\n"
        f"--- Elite Parent (Rank 0, Utility: {current_best_utility}) ---\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n\n"
        f"--- Runner-up Parent (Rank 1, Utility: {runner_up_util}) ---\n"
        f"Idea: {runner_up_idea}\n"
        f"Code:\n```python\n{runner_up_code}\n```\n\n"
        f"--- Exploratory Variant (Utility: {exploratory_util}) ---\n"
        f"Idea: {exploratory_idea}\n"
        f"Code:\n```python\n{exploratory_code}\n```\n\n"
        "Apply a Multi-Parent Differential Ruin-and-Recombination methodology:\n"
        "1. Identify the 'Invariable Core': The critical structural backbone from the Elite that must be preserved.\n"
        "2. Pinpoint 4 distinct sub-components in the Elite that suffer from sub-optimality or stagnation.\n"
        "3. For each ruined component, prescribe a recombination strategy that cross-pollinates unique strengths, "
        "differentiates away from the exploratory solution's failure modes, and grafts modular algorithmic building blocks.\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<critical logic to retain from the elite>",\n'
        '  "recombination_operators": [\n'
        '    {"ruin_target": "<component to ruin in elite>", "recombination_strategy": "<differential cross-pollination and repair logic>"},\n'
        '    {"ruin_target": "<component to ruin in elite>", "recombination_strategy": "<differential cross-pollination and repair logic>"},\n'
        '    {"ruin_target": "<component to ruin in elite>", "recombination_strategy": "<differential cross-pollination and repair logic>"},\n'
        '    {"ruin_target": "<component to ruin in elite>", "recombination_strategy": "<differential cross-pollination and repair logic>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core elite algorithmic search progression")
        operators = diag_data.get("recombination_operators", [])
    except Exception:
        invariable_core = "Elite algorithmic pipeline and high-performing baseline structure"
        operators = [
            {
                "ruin_target": "Local search and neighborhood perturbation mechanism",
                "recombination_strategy": "Differential crossover of multi-neighborhood structures adapted from the runner-up"
            },
            {
                "ruin_target": "Stagnation detection and acceptance criterion",
                "recombination_strategy": "Metropolis-guided adaptive cooling with diversity injection to escape exploratory traps"
            },
            {
                "ruin_target": "Candidate generation and constructive step",
                "recombination_strategy": "Synergistic recombination of elite greedy logic and randomized heuristic grafting"
            },
            {
                "ruin_target": "Exploration-exploitation control schedule",
                "recombination_strategy": "Dynamic feedback-driven parameter adjustment informed by differential performance gaps"
            }
        ]

    # Generate recombined and rebuilt algorithm candidates in parallel batch
    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("ruin_target", "Sub-optimal elite component")
        recomb_strat = op.get("recombination_strategy", "Differential multi-parent crossover and synthesis")

        reconstruct_prompt = (
            f"You are executing Multi-Parent Differential Ruin-and-Recombination on the optimization algorithm for '{task}'.\n\n"
            f"ELITE BASELINE CODE:\n```python\n{current_best_code}\n```\n"
            f"Current Best Utility: {current_best_utility} (lower is better)\n\n"
            f"RUNNER-UP INSPIRATION:\nIdea: {runner_up_idea}\nUtility: {runner_up_util}\n\n"
            f"INVARIABLE CORE TO PRESERVE: {invariable_core}\n"
            f"TARGET TO RUIN IN ELITE: {ruin_target}\n"
            f"RECOMBINATION & RECONSTRUCTION: {recomb_strat}\n\n"
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
