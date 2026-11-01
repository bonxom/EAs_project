# Idea: An adaptive ruin-and-recreate metaheuristic integrating structured diagnostic decomposition with an entropy-and-stagnation guided non-monotonic acceptance filter that permits promising diverse candidates to escape local optima.
#{An adaptive ruin-and-recreate metaheuristic integrating structured diagnostic decomposition with an entropy-and-stagnation guided non-monotonic acceptance filter that permits promising diverse candidates to escape local optima.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    # 1. Retrieve elite solution and reference population context
    pop_size = population.get_subtask_size(task)
    elite_sol = population.get_solution_by_index(task, 0)
    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # Compute population diversity benchmark for dynamic thresholding
    pop_utilities = []
    sample_size = min(pop_size, 5)
    for idx in range(sample_size):
        try:
            sol = population.get_solution_by_index(task, idx)
            pop_utilities.append(sol['utility'])
        except Exception:
            break

    util_std = float(np.std(pop_utilities)) if len(pop_utilities) > 1 else max(abs(current_best_utility) * 0.05, 1e-4)
    stagnation_ratio = float(np.mean(pop_utilities) - current_best_utility) if pop_utilities else 0.0
    stagnated = stagnation_ratio < 1e-5

    # 2. Ruin-and-Recreate Diagnostic Phase
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n"
        f"Stagnation Status: {'High Stagnation Detected (needs radical diversification)' if stagnated else 'Normal Convergence Progress'}\n\n"
        "Apply an Adaptive Ruin-and-Recreate decomposition:\n"
        "1. Identify the 'Invariable Core': The algorithmic anchor/backbone that ensures baseline feasibility and performance.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components (e.g., representation, neighborhood operators, perturbation strength, acceptance logic) and rebuild them.\n"
        "Return your analysis strictly in JSON format:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<concise description of core invariant>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>"}\n'
        '  ]\n'
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.7)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core algorithmic search pipeline")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression and structure"
        operators = [
            {"target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator"},
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Non-monotonic threshold acceptance with diversity preservation"},
            {"target_to_ruin": "Candidate generation", "reconstruct_strategy": "Guided greedy randomized adaptive construction"},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Heavy structural mutation with memory-guided repair"}
        ]

    # 3. Parallel Batched Reconstruction Phase
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

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9 if stagnated else 0.7)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    # 4. Evaluation and Non-Monotonic Escape Filtering
    candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            candidates.append({
                'idea': idea,
                'code': code,
                'utility': score,
                'char_len': len(code)
            })
        except Exception:
            continue

    if not candidates:
        return current_best_idea, current_best_code, current_best_utility

    # Find greedy best candidate
    best_candidate = min(candidates, key=lambda x: x['utility'])

    # If strictly better, accept immediately (greedy exploitation)
    if best_candidate['utility'] < current_best_utility:
        return best_candidate['idea'], best_candidate['code'], best_candidate['utility']

    # 5. Non-Monotonic Adaptive Acceptance Mechanism
    # When stagnated or facing local optima, accept near-elite solutions exhibiting structural divergence
    adaptive_tolerance = util_std * (1.5 if stagnated else 0.5)
    near_elite = [
        c for c in candidates 
        if (c['utility'] - current_best_utility) <= adaptive_tolerance
    ]

    if near_elite and stagnated:
        # Novelty/diversity proxy: structural difference via code length variation and conceptual divergence
        base_len = len(current_best_code)
        for cand in near_elite:
            cand['novelty'] = abs(cand['char_len'] - base_len) / max(base_len, 1)
        # Select the near-elite candidate maximizing novelty to escape the basin
        selected = max(near_elite, key=lambda x: x['novelty'])
        return selected['idea'], selected['code'], selected['utility']

    return current_best_idea, current_best_code, current_best_utility
