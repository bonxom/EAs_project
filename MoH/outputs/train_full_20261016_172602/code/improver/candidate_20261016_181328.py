# Idea: An adaptive ruin-and-recreate metaheuristic integrating structured LLM diagnostic decomposition with a quality-diversity survivor archive that balances Pareto fitness-distance trade-offs and simulated annealing acceptance to prevent deceptive stagnation.
#{An adaptive ruin-and-recreate metaheuristic integrating structured LLM diagnostic decomposition with a quality-diversity survivor archive that balances Pareto fitness-distance trade-offs and simulated annealing acceptance to prevent deceptive stagnation.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    # 1. Base elite extraction
    elite_sol = population.get_solution_by_index(task, 0)
    pop_size = population.get_subtask_size(task)
    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # 2. Diagnostic Ruin-and-Recreate decomposition via LLM
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate (Iterated Local Search) philosophy:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components to ruin "
        "(e.g., representation/encoding, neighborhood exploration, selection pressure/acceptance, or stagnation handling) "
        "and rebuild them with superior, orthogonal algorithmic logic.\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<brief summary of what must be retained>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.7)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core search pipeline")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression"
        operators = [
            {"target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator"},
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Simulated annealing cooling schedule with history-based restarts"},
            {"target_to_ruin": "Candidate generation", "reconstruct_strategy": "Guided greedy randomized adaptive construction"},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Heavy structural mutation with memory-guided repair"}
        ]

    # 3. Batched Reconstruct Prompting
    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic blocks")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")
        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=1.0)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    # 4. Evaluation and Candidate Extraction
    evaluated_candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            evaluated_candidates.append({
                'idea': idea,
                'code': code,
                'utility': float(score)
            })
        except Exception:
            continue

    if not evaluated_candidates:
        return current_best_idea, current_best_code, current_best_utility

    # Check for direct strictly improved best
    global_min_cand = min(evaluated_candidates, key=lambda x: x['utility'])
    if global_min_cand['utility'] < current_best_utility:
        current_best_idea = global_min_cand['idea']
        current_best_code = global_min_cand['code']
        current_best_utility = global_min_cand['utility']
        return current_best_idea, current_best_code, current_best_utility

    # 5. Quality-Diversity / Simulated Annealing Survivor Selection
    # If no strict improvement occurred, maintain exploration by selecting a high-diversity near-optimal candidate
    def compute_code_distance(c1, c2):
        tokens1 = set(c1.split())
        tokens2 = set(c2.split())
        if not tokens1 or not tokens2:
            return 0.0
        return 1.0 - (len(tokens1 & tokens2) / float(len(tokens1 | tokens2)))

    # Compute distances to current elite
    for cand in evaluated_candidates:
        cand['distance'] = compute_code_distance(cand['code'], current_best_code)

    # Simulated Annealing acceptance over diverse candidates
    temperature = max(1e-4, 0.05 * abs(current_best_utility))
    accepted_candidates = []

    for cand in evaluated_candidates:
        delta = cand['utility'] - current_best_utility
        prob = np.exp(-delta / temperature) if delta > 0 else 1.0
        if np.random.rand() < prob:
            accepted_candidates.append(cand)

    pool = accepted_candidates if accepted_candidates else evaluated_candidates

    # Rank-based fitness-distance trade-off: Score = normalized_utility - alpha * normalized_distance
    utilities = [c['utility'] for c in pool]
    distances = [c['distance'] for c in pool]

    u_min, u_max = min(utilities), max(utilities)
    d_min, d_max = min(distances), max(distances)

    u_range = (u_max - u_min) if (u_max - u_min) > 1e-9 else 1.0
    d_range = (d_max - d_min) if (d_max - d_min) > 1e-9 else 1.0

    alpha = 0.35  # Balance exploration distance vs exploitation quality
    best_survivor = None
    best_score = float('inf')

    for cand in pool:
        norm_u = (cand['utility'] - u_min) / u_range
        norm_d = (cand['distance'] - d_min) / d_range
        score = norm_u - alpha * norm_d
        if score < best_score:
            best_score = score
            best_survivor = cand

    if best_survivor is not None:
        current_best_idea = best_survivor['idea']
        current_best_code = best_survivor['code']
        current_best_utility = best_survivor['utility']

    return current_best_idea, current_best_code, current_best_utility
