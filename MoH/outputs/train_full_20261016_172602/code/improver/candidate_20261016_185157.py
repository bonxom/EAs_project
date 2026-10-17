# Idea: An orthogonal multi-role metaheuristic architecture that diagnoses an invariable algorithmic core and coordinates specialized search personas across distinct temperature regimes and tabu-guided prompt vectors for structured exploration and exploitation.
# {An orthogonal multi-role metaheuristic architecture that diagnoses an invariable algorithmic core and coordinates specialized search personas across distinct temperature regimes and tabu-guided prompt vectors for structured exploration and exploitation.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    # 1. Population retrieval and baseline initialization
    elite_sol = population.get_solution_by_index(task, 0)
    pop_size = population.get_subtask_size(task)
    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # 2. Diagnostic Meta-Decomposition Phase: Extract Invariable Core & Stagnation Targets
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate philosophy:\n"
        "1. Identify the 'Invariable Core': The critical algorithmic invariants and high-utility logic that must be retained.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components.\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<critical logic to preserve>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<sub-component>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<sub-component>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<sub-component>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<sub-component>", "reconstruct_strategy": "<innovative recreation mechanism>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core heuristic search progression and structure")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search pipeline and elite solution preservation"
        operators = [
            {"target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator"},
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Simulated annealing cooling schedule with history-based restarts"},
            {"target_to_ruin": "Candidate generation", "reconstruct_strategy": "Guided greedy randomized adaptive construction"},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Heavy structural mutation with memory-guided repair"}
        ]

    # Ensure fallback padding if fewer than 4 operators were returned
    while len(operators) < 4:
        operators.append({
            "target_to_ruin": "Sub-optimal exploration components",
            "reconstruct_strategy": "Pareto-guided localized intensification and diversified crossover"
        })

    # 3. Orthogonal Search Roles & Temperature Partitioning
    # Partition generation into distinct exploration/exploitation roles across adaptive temperature tiers
    search_roles = [
        {
            "role": "Precision Exploitation & Micro-Optimization",
            "temperature": 0.3,
            "instruction": "Perform surgical refinement. Optimize mathematical formulas, clean edge-cases, eliminate numerical instabilities, and tighten search bounds without altering the core algorithmic paradigm."
        },
        {
            "role": "Balanced Structural Refactoring",
            "temperature": 0.7,
            "instruction": "Refactor algorithmic data flow and balance neighborhood transitions. Introduce adaptive dynamic parameter scaling and structured feedback mechanisms."
        },
        {
            "role": "Heuristic Paradigm Shift",
            "temperature": 0.9,
            "instruction": "Introduce innovative metaheuristic hybridization. Re-engineer the candidate generation loop with novel stochastic and priority-guided operators."
        },
        {
            "role": "Radical Exploration with Tabu Constraints",
            "temperature": 1.1,
            "instruction": f"Break conventional patterns. TABU RESTRICTION: Do not reuse standard greedy or trivial local steps from the previous idea '{current_best_idea}'. Implement a bold, orthogonal search mechanism."
        }
    ]

    # 4. Generate candidate batch grouped by temperature schedule
    candidates = []

    for idx, (op, role_cfg) in enumerate(zip(operators[:4], search_roles)):
        ruin_target = op.get("target_to_ruin", "Local search operator")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced multi-neighborhood search")

        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"SEARCH ROLE: {role_cfg['role']}\n"
            f"DIRECTIVE: {role_cfg['instruction']}\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )

        try:
            # Query the model for this specific role and assigned temperature
            response = language_model.prompt(expertise, reconstruct_prompt, temperature=role_cfg["temperature"])
            code = extract_code(response)
            idea = extract_idea(response)

            if code and code.strip():
                score = utility(code, idea, task)
                candidates.append((idea, code, score))
        except Exception:
            continue

    # 5. Elite Selection and Strict Improvement Tracking
    if candidates:
        iter_best_idea, iter_best_code, iter_best_utility = min(candidates, key=lambda x: x[2])
        if iter_best_utility < current_best_utility:
            current_best_idea = iter_best_idea
            current_best_code = iter_best_code
            current_best_utility = iter_best_utility

    return current_best_idea, current_best_code, current_best_utility
