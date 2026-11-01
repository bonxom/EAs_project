# Idea: An evolutionary metaheuristic combining structured diagnostic reflection with heterogeneous prompt specialization across exploration, intensification, mathematical hybridization, and robust parameter hardening to discover superior combinatorial algorithms.
#{An evolutionary metaheuristic combining structured diagnostic reflection with heterogeneous prompt specialization across exploration, intensification, mathematical hybridization, and robust parameter hardening to discover superior combinatorial algorithms.}

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
    ref_code = reference_sol.get('best_sol', current_best_code)

    # Phase 1: Structured Diagnostic Reflection
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Perform a rigorous decomposition:\n"
        "1. Identify the 'Invariable Core': The essential algorithmic backbone and invariants that must be preserved.\n"
        "2. Identify the 'Primary Bottlenecks': Structural weaknesses, convergence stagnation, or computational inefficiencies.\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<critical logic to preserve>",\n'
        '  "bottlenecks": "<specific algorithmic deficiencies>"\n'
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.5)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core search pipeline and state representation")
        bottlenecks = diag_data.get("bottlenecks", "Premature convergence and sub-optimal search neighborhood")
    except Exception:
        invariable_core = "Core search pipeline and state representation"
        bottlenecks = "Premature convergence and sub-optimal search neighborhood"

    # Phase 2: Heterogeneous Multi-Operator Prompt Specialization
    roles = [
        {
            "name": "Radical Structural Innovator",
            "directive": (
                "RADICAL PARADIGM RESTRUCTURING: Completely tear down and replace the primary search loop. "
                "Introduce an unconventional metaheuristic framework (e.g., population-based scatter search, "
                "variable neighborhood decomposition, or multi-objective Pareto archive) that breaks past the identified bottlenecks."
            ),
            "temp": 0.95
        },
        {
            "name": "Fine-Grained Intensification Specialist",
            "directive": (
                "ADVANCED LOCAL INTENSIFICATION: Preserve the macro structure but thoroughly rewrite neighborhood "
                "exploration and move evaluation. Implement multi-tier local search, cache-efficient delta evaluation, "
                "adaptive variable neighborhood descent (VND), or highly targeted local repair moves to squeeze maximum utility."
            ),
            "temp": 0.65
        },
        {
            "name": "Mathematical & Heuristic Hybridizer",
            "directive": (
                "MATHEMATICAL & BOUND-GUIDED HYBRIDIZATION: Integrate analytical problem bounds, frequency-based "
                "memory structures, adaptive Lagrangian/dual penalties, or greedy randomized adaptive search (GRASP) "
                "constructions to systematically direct the search toward high-quality feasible basins."
            ),
            "temp": 0.8
        },
        {
            "name": "Adaptive Stagnation & Hardening Engineer",
            "directive": (
                "ROBUSTNESS & ADAPTIVE EVASION: Re-engineer the selection pressure, diversification restarts, and "
                "cooling schedules. Implement dynamic parameter adaptation (reactive search), robust edge-case handling, "
                "and heavy ruined-solution recovery to guarantee escape from deep local optima."
            ),
            "temp": 0.85
        }
    ]

    message_batch = []
    temps = []
    base_instructions = (
        f"Format requirements:\n{function_format}\n"
        "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
        "followed by the complete, executable, and fully defined Python code implementation. "
        "Ensure the algorithm avoids infinite loops, contains no syntax errors, and optimizes execution efficiency."
    )

    for role in roles:
        prompt_text = (
            f"You are the {role['name']} in an advanced metaheuristic optimization ensemble.\n\n"
            f"Problem Task: {task}\n"
            f"Current Best Utility: {current_best_utility} (lower is better)\n"
            f"Current Best Code:\n```python\n{current_best_code}\n```\n"
            f"Alternative Reference Code:\n```python\n{ref_code}\n```\n\n"
            f"DIAGNOSTIC BACKBONE TO PRESERVE: {invariable_core}\n"
            f"DIAGNOSED BOTTLENECKS: {bottlenecks}\n\n"
            f"YOUR SPECIALIZED DIRECTIVE: {role['directive']}\n\n"
            f"{base_instructions}"
        )
        message_batch.append(prompt_text)
        temps.append(role["temp"])

    # Execute batch generation
    avg_temp = float(np.mean(temps))
    responses = language_model.prompt_batch(expertise, message_batch, temperature=avg_temp)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            if score is not None and not np.isnan(score):
                candidates.append((idea, code, score))
        except Exception:
            continue

    # Phase 3: Selection and Elitism
    if candidates:
        iter_best_idea, iter_best_code, iter_best_utility = min(candidates, key=lambda x: x[2])
        if iter_best_utility < current_best_utility:
            current_best_idea = iter_best_idea
            current_best_code = iter_best_code
            current_best_utility = iter_best_utility

    return current_best_idea, current_best_code, current_best_utility
