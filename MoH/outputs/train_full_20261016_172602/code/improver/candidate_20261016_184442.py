# Idea: An adaptive ruin-and-recreate metaheuristic coupling structured invariant diagnosis with Pareto diversity-utility selection and simulated annealing acceptance across an elite-diversity archive.
#{An adaptive ruin-and-recreate metaheuristic coupling structured invariant diagnosis with Pareto diversity-utility selection and simulated annealing acceptance across an elite-diversity archive.}

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
    rand_sol = population.get_random_solution(task) if pop_size > 1 else elite_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = float(elite_sol['utility'])

    # Phase 1: Invariant Identification & Bottleneck Diagnosis
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Current Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate philosophy:\n"
        "1. Identify the 'Invariable Core': Algorithmic backbone that MUST be preserved for performance.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components:\n"
        "   - Diversity injection / Exploration mechanism\n"
        "   - Local neighborhood intensification / Mutation\n"
        "   - Acceptance criterion / Stagnation escape\n"
        "   - Adaptive parameter control / Multi-operator routing\n"
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
            {"target_to_ruin": "Exploration mechanism", "reconstruct_strategy": "Pareto archive search balancing diversity and score"},
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Adaptive Metropolis-Hastings simulated annealing"},
            {"target_to_ruin": "Candidate perturbation", "reconstruct_strategy": "Bi-directional mutation with structural repair"},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Orthogonal neighborhood crossover and dynamic restarts"}
        ]

    # Phase 2: Batched Ruin-and-Recreate Prompt Generation
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

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.85)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    # Phase 3: Archive Evaluation with Pareto-Diversity and Metropolis-Hastings Acceptance
    evaluated_candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = float(utility(code, idea, task))
            # Calculate syntactic novelty distance relative to current best
            code_len_diff = abs(len(code) - len(current_best_code)) / max(len(current_best_code), 1)
            char_overlap = len(set(code) & set(current_best_code)) / max(len(set(code) | set(current_best_code)), 1)
            novelty = code_len_diff + (1.0 - char_overlap)
            evaluated_candidates.append({
                "idea": idea,
                "code": code,
                "utility": score,
                "novelty": novelty
            })
        except Exception:
            continue

    if evaluated_candidates:
        # Pareto non-dominated filtering (minimizing utility, maximizing novelty)
        pareto_front = []
        for c1 in evaluated_candidates:
            dominated = False
            for c2 in evaluated_candidates:
                if (c2["utility"] <= c1["utility"] and c2["novelty"] >= c1["novelty"]) and \
                   (c2["utility"] < c1["utility"] or c2["novelty"] > c1["novelty"]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(c1)

        pool = pareto_front if pareto_front else evaluated_candidates

        # Sort pool by raw utility to find the best candidate in this iteration
        pool.sort(key=lambda x: x["utility"])
        best_candidate = pool[0]

        # Metropolis-Hastings probabilistic acceptance criterion
        delta = best_candidate["utility"] - current_best_utility
        temperature = max(0.01, 0.05 * abs(current_best_utility))

        if delta < 0:
            accept = True
        else:
            accept_prob = np.exp(-delta / temperature)
            accept = np.random.rand() < accept_prob

        if accept:
            current_best_idea = best_candidate["idea"]
            current_best_code = best_candidate["code"]
            current_best_utility = best_candidate["utility"]

    return current_best_idea, current_best_code, current_best_utility
