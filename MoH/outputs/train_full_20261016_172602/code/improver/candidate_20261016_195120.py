# Idea: A diagnostic Ruin-and-Recreate metaheuristic that profiles algorithm bottlenecks into targeted perturbation operators and queries the LLM across a temperature ladder, pairing deep structural overhauls at higher temperatures with fine-grained micro-optimizations at lower temperatures.
#{A diagnostic Ruin-and-Recreate metaheuristic that profiles algorithm bottlenecks into targeted perturbation operators and queries the LLM across a temperature ladder, pairing deep structural overhauls at higher temperatures with fine-grained micro-optimizations at lower temperatures.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    # Retrieve elite and reference solutions
    elite_sol = population.get_solution_by_index(task, 0)
    pop_size = population.get_subtask_size(task)
    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate philosophy by formulating 4 targeted perturbation operators ranked by exploration depth:\n"
        "1. Radical Structural Redesign: Tear down high-level algorithmic architecture or metaheuristic framework.\n"
        "2. Neighborhood & Perturbation Overhaul: Redesign core search dynamics, mutation, or ruin operators.\n"
        "3. Adaptive Search Control: Overhaul selection pressure, cooling schedules, or stagnation detection.\n"
        "4. Micro-optimization & Tuning: Fine-tune local heuristics, parameter adaptation, and execution efficiency.\n\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<brief summary of what must be retained>",\n'
        '  "reconstruction_operators": [\n'
        '    {"ruin_target": "<component to ruin>", "reconstruct_strategy": "<innovative recreation mechanism>", "scope": "radical"},\n'
        '    {"ruin_target": "<component to ruin>", "reconstruct_strategy": "<innovative recreation mechanism>", "scope": "structural"},\n'
        '    {"ruin_target": "<component to ruin>", "reconstruct_strategy": "<innovative recreation mechanism>", "scope": "adaptive_control"},\n'
        '    {"ruin_target": "<component to ruin>", "reconstruct_strategy": "<innovative recreation mechanism>", "scope": "micro_tuning"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core search pipeline and problem constraints")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression and valid evaluation pipeline"
        operators = [
            {"ruin_target": "Algorithmic architecture", "reconstruct_strategy": "Hybrid cooperative multi-phase search with co-evolving sub-heuristics", "scope": "radical"},
            {"ruin_target": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator with memory guidance", "scope": "structural"},
            {"ruin_target": "Selection and acceptance mechanism", "reconstruct_strategy": "Non-monotonic adaptive cooling schedule with stagnation-driven restarts", "scope": "adaptive_control"},
            {"ruin_target": "Greedy component & parameters", "reconstruct_strategy": "Self-adaptive hyperparameter modulation and vectorized tie-breaking", "scope": "micro_tuning"}
        ]

    # Temperature ladder mapping exploration scope to sampling temperature
    temp_ladder = [1.05, 0.85, 0.65, 0.40]

    candidates = []

    # Process prompts grouped by temperature or staggered batch to enable diverse exploration profiles
    for i, op in enumerate(operators[:4]):
        temp = temp_ladder[i] if i < len(temp_ladder) else 0.7
        ruin_target = op.get("ruin_target", "Inefficient heuristic blocks")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")
        scope = op.get("scope", "perturbation")

        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"CORE BACKBONE TO PRESERVE: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n"
            f"OPTIMIZATION LEVEL ({scope}): "
            + ("High-variance novel structural leap." if temp >= 0.9 else 
               "Balanced algorithmic enhancement." if temp >= 0.6 else 
               "High-precision fine-tuning, deterministic robustness, and micro-optimization.") + "\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )

        try:
            resp = language_model.prompt(expertise, reconstruct_prompt, temperature=temp)
            code = extract_code(resp)
            idea = extract_idea(resp)
            if code and code.strip():
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
