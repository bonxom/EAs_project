# Idea: A multi-parent differential crossover metaheuristic that diagnoses architectural and behavioral discrepancies across an elite, median, and exploratory outlier to parallel-generate targeted synergistic recombinations and differential operator hybrids.
#{A multi-parent differential crossover metaheuristic that diagnoses architectural and behavioral discrepancies across an elite, median, and exploratory outlier to parallel-generate targeted synergistic recombinations and differential operator hybrids.}#

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

    # Select representative multi-parent triage: Elite, Median, and Exploratory Outlier
    if pop_size >= 3:
        median_idx = pop_size // 2
        median_sol = population.get_solution_by_index(task, median_idx)
        outlier_idx = pop_size - 1
        outlier_sol = population.get_solution_by_index(task, outlier_idx)
    elif pop_size == 2:
        median_sol = population.get_solution_by_index(task, 1)
        outlier_sol = population.get_random_solution(task)
    else:
        median_sol = elite_sol
        outlier_sol = elite_sol

    # Phase 1: Differential Profiling and Recombination Analysis
    diagnosis_prompt = (
        f"Perform a multi-parent differential analysis across three distinct algorithms for task '{task}':\n\n"
        f"--- PARENT A (Elite, Utility: {elite_sol['utility']}) ---\n"
        f"Idea: {elite_sol.get('idea', 'Elite performer')}\n"
        f"Code snippet / core logic:\n```python\n{elite_sol['best_sol'][:1200]}\n```\n\n"
        f"--- PARENT B (Median, Utility: {median_sol['utility']}) ---\n"
        f"Idea: {median_sol.get('idea', 'Balanced performer')}\n"
        f"Code snippet / core logic:\n```python\n{median_sol['best_sol'][:1000]}\n```\n\n"
        f"--- PARENT C (Exploratory Outlier, Utility: {outlier_sol['utility']}) ---\n"
        f"Idea: {outlier_sol.get('idea', 'Divergent explorer')}\n"
        f"Code snippet / core logic:\n```python\n{outlier_sol['best_sol'][:1000]}\n```\n\n"
        "Analyze the structural and operator differences between Parent A, B, and C.\n"
        "Design 4 distinct differential crossover operators:\n"
        "1. Backbone Dominance with Outlier Perturbation (Inject C's novel mechanism into A's core).\n"
        "2. Differential Vector Synthesis (Reconstruct A using the positive vector difference [B -> A] projected on C).\n"
        "3. Hybrid Modular Recombination (Fuse A's high-exploitation components with B/C's exploratory components).\n"
        "4. Complementary Synergistic Rebuild (Synthesize a completely unified algorithm overcoming A's stagnation pitfalls).\n\n"
        "Return your response strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "differential_diagnosis": "<key strengths of A and unexploited mechanisms in B/C>",\n'
        '  "crossover_strategies": [\n'
        '    {"strategy_name": "<strategy 1>", "differential_direction": "<detailed recombination recipe>", "focus": "<mechanisms to merge>"},\n'
        '    {"strategy_name": "<strategy 2>", "differential_direction": "<detailed recombination recipe>", "focus": "<mechanisms to merge>"},\n'
        '    {"strategy_name": "<strategy 3>", "differential_direction": "<detailed recombination recipe>", "focus": "<mechanisms to merge>"},\n'
        '    {"strategy_name": "<strategy 4>", "differential_direction": "<detailed recombination recipe>", "focus": "<mechanisms to merge>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        strategies = diag_data.get("crossover_strategies", [])
    except Exception:
        strategies = [
            {"strategy_name": "Backbone Recombination", "differential_direction": "Preserve Parent A's core search, integrate novel exploratory operators from Parent C.", "focus": "Neighborhood exploration and diversity"},
            {"strategy_name": "Differential Direction Shift", "differential_direction": "Identify what separates A from B, amplify that mechanism and apply it to avoid local optima.", "focus": "Search intensification"},
            {"strategy_name": "Synergistic Modularity", "differential_direction": "Combine A's refinement engine with B's global diversification.", "focus": "Exploration-exploitation balance"},
            {"strategy_name": "Adaptive Hybrid Rebuild", "differential_direction": "Unify the best state transitions from all parents with adaptive parameter control.", "focus": "Stagnation recovery"}
        ]

    # Phase 2: Parallel Batch Recombination Generation
    message_batch = []
    for strat in strategies[:4]:
        strat_name = strat.get("strategy_name", "Differential Hybrid")
        direction = strat.get("differential_direction", "Blend Parent A core with Parent C diversification.")
        focus = strat.get("focus", "Synergistic optimization")

        recombine_prompt = (
            f"You are executing a Multi-Parent Differential Crossover on task '{task}'.\n\n"
            f"REFERENCE ELITE (Parent A, Utility: {elite_sol['utility']}):\n"
            f"```python\n{elite_sol['best_sol']}\n```\n\n"
            f"CROSSOVER STRATEGY: {strat_name}\n"
            f"RECOMBINATION DIRECTIVE: {direction}\n"
            f"OPERATOR FOCUS: {focus}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, contains robust error handling, and executes efficiently."
        )
        message_batch.append(recombine_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    # Phase 3: Robust Evaluation & Best Solution Update
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
