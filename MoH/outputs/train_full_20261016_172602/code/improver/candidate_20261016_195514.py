# Idea: An Adaptive Quality-Diversity and Pareto-ranking acceptance filter that replaces greedy selection by evaluating candidates on both raw utility and behavioral/structural novelty metrics, preventing premature convergence and maintaining high-potential search trajectories.
#{An Adaptive Quality-Diversity and Pareto-ranking acceptance filter that replaces greedy selection by evaluating candidates on both raw utility and behavioral/structural novelty metrics, preventing premature convergence and maintaining high-potential search trajectories.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    # 1. Retrieve elite and diverse references from the population
    elite_sol = population.get_solution_by_index(task, 0)
    pop_size = population.get_subtask_size(task)
    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol
    random_sol = population.get_random_solution(task) if pop_size > 2 else elite_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # 2. Structured LLM diagnosis: identify core invariants and ruin-and-recreate vectors
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Reference Alternative Idea: " + str(reference_sol.get('idea', 'Secondary reference')) + "\n\n"
        "Apply an Adaptive Ruin-and-Recreate philosophy with Behavioral Diversity:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components to ruin "
        "(e.g., representation, neighborhood exploration, selection pressure, or stagnation handling) while incorporating "
        "divergent computational mechanics (e.g., adaptive memory, surrogate guidance, non-monotonic cooling, multi-scale ruin).\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<brief summary of what must be retained>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>", "behavioral_trait": "<e.g. high-exploration, fine-grained exploitation, dynamic-parameterization>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>", "behavioral_trait": "<...>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>", "behavioral_trait": "<...>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative recreation mechanism>", "behavioral_trait": "<...>"}\n'
        '  ]\n'
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
            {"target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator", "behavioral_trait": "high-exploration"},
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Information-theoretic quality-diversity acceptance with non-monotonic cooling", "behavioral_trait": "diversity-preservation"},
            {"target_to_ruin": "Candidate generation", "reconstruct_strategy": "Guided greedy randomized adaptive construction with memory", "behavioral_trait": "structured-construction"},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Heavy structural multi-scale mutation with surrogate repair", "behavioral_trait": "escape-perturbation"}
        ]

    # 3. Parallelized Prompt-Batch Construction
    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic blocks")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")
        trait = op.get("behavioral_trait", "novel-exploration")
        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n"
            f"TARGET BEHAVIORAL TRAIT: {trait}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=1.0)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    # 4. Multi-objective Quality-Diversity Acceptance Filter
    evaluated_candidates = []

    # Behavioral descriptor computation helper:
    # Characterizes candidate code by normalized length and token variety relative to baseline
    baseline_tokens = set(current_best_code.split())
    baseline_len = max(1, len(current_best_code))

    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            # Behavioral novelty descriptor: Jaccard distance of token space + structural length variance
            cand_tokens = set(code.split())
            jaccard_dist = 1.0 - (len(baseline_tokens & cand_tokens) / max(1, len(baseline_tokens | cand_tokens)))
            length_diff = abs(len(code) - baseline_len) / baseline_len
            novelty_score = 0.7 * jaccard_dist + 0.3 * min(1.0, length_diff)

            evaluated_candidates.append({
                "idea": idea,
                "code": code,
                "utility": score,
                "novelty": novelty_score
            })
        except Exception:
            continue

    if not evaluated_candidates:
        return current_best_idea, current_best_code, current_best_utility

    # 5. Pareto / Quality-Diversity Selection:
    # Check for direct strictly-superior solutions first
    strictly_better = [c for c in evaluated_candidates if c["utility"] < current_best_utility]

    if strictly_better:
        # Among strictly better solutions, balance utility improvement with algorithmic novelty
        best_cand = min(strictly_better, key=lambda x: (x["utility"], -x["novelty"]))
        return best_cand["idea"], best_cand["code"], best_cand["utility"]
    else:
        # Near-miss acceptance criterion: accept a highly novel candidate if within a tight epsilon margin
        # This prevents premature convergence in local optima plateaus
        epsilon = 0.02 * abs(current_best_utility)
        near_candidates = [c for c in evaluated_candidates if c["utility"] <= current_best_utility + epsilon]
        if near_candidates:
            # Pick the most behaviorally diverse candidate within the epsilon neighborhood
            most_novel = max(near_candidates, key=lambda x: x["novelty"])
            if most_novel["novelty"] > 0.35:  # Significant architectural innovation threshold
                return most_novel["idea"], most_novel["code"], most_novel["utility"]

    # Fallback to current best if no candidate satisfies Quality-Diversity criteria
    return current_best_idea, current_best_code, current_best_utility
