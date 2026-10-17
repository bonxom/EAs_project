# Idea: An Iterated Ruin-and-Recreate metaheuristic combining structured diagnostic perturbation with a two-tier tournament and self-reflective repair pipeline to debug and optimize the most promising generated candidate.
# Utility: 0.035612740937071036
#{An Iterated Ruin-and-Recreate metaheuristic combining structured diagnostic perturbation with a two-tier tournament and self-reflective repair pipeline to debug and optimize the most promising generated candidate.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    elite_sol = population.get_solution_by_index(task, 0)
    pop_size = population.get_subtask_size(task)
    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # Phase 1: Structured Diagnosis & Targeted Ruin-and-Recreate decomposition
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate (Iterated Local Search) philosophy:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components to ruin (e.g., representation, neighborhood exploration, selection pressure, or stagnation handling) and rebuild them with superior algorithmic logic.\n"
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

    # Phase 2: Parallel Ruin-and-Recreate Generation
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

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    evaluated_candidates = []
    unevaluated_or_failed = []

    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            evaluated_candidates.append({"idea": idea, "code": code, "utility": score})
        except Exception as e:
            unevaluated_or_failed.append({"idea": idea, "code": code, "error": str(e)})

    # Phase 3: Two-tier Tournament Selection with Self-Reflective Repair
    if evaluated_candidates:
        # Tier 1: Tournament sampling to select top candidate
        tournament_size = min(3, len(evaluated_candidates))
        selected_indices = np.random.choice(len(evaluated_candidates), size=tournament_size, replace=False)
        tournament_pool = [evaluated_candidates[i] for i in selected_indices]
        top_candidate = min(tournament_pool, key=lambda x: x["utility"])

        # Also track the absolute best from the batch
        batch_best = min(evaluated_candidates, key=lambda x: x["utility"])
        if batch_best["utility"] < top_candidate["utility"]:
            top_candidate = batch_best

        # Tier 2: Check if top candidate failed to improve or has room for repair
        if top_candidate["utility"] >= current_best_utility:
            repair_prompt = (
                f"The following candidate algorithm achieved utility {top_candidate['utility']}, which did not surpass "
                f"the benchmark {current_best_utility} for task '{task}'.\n"
                f"Idea: {top_candidate['idea']}\n"
                f"Code:\n```python\n{top_candidate['code']}\n```\n\n"
                "Perform a targeted self-reflective repair on this algorithm:\n"
                "1. Eliminate asymptotic runtime bottlenecks and inefficient redundant evaluations.\n"
                "2. Tighten parameter balance between exploration and exploitation.\n"
                "3. Ensure seamless boundary handling and fallback safety.\n\n"
                f"Format requirements:\n{function_format}\n"
                "Provide a concise one-sentence summary in braces marked as a comment, followed by the fully executable code."
            )
            try:
                repair_response = language_model.prompt(expertise, repair_prompt, temperature=0.4)
                rep_code = extract_code(repair_response)
                rep_idea = extract_idea(repair_response)
                if rep_code and rep_code.strip():
                    rep_score = utility(rep_code, rep_idea, task)
                    if rep_score < top_candidate["utility"]:
                        top_candidate = {"idea": rep_idea, "code": rep_code, "utility": rep_score}
            except Exception:
                pass

        if top_candidate["utility"] < current_best_utility:
            current_best_idea = top_candidate["idea"]
            current_best_code = top_candidate["code"]
            current_best_utility = top_candidate["utility"]

    elif unevaluated_or_failed:
        # Emergency repair on a syntactically invalid or failing candidate
        broken_candidate = unevaluated_or_failed[0]
        repair_prompt = (
            f"The candidate algorithm for task '{task}' failed during execution with error: {broken_candidate.get('error', 'Execution Error')}\n"
            f"Failing Code:\n```python\n{broken_candidate['code']}\n```\n\n"
            "Diagnose and completely repair this implementation. Fix all syntax, type, and logic errors while maintaining algorithmic ingenuity.\n\n"
            f"Format requirements:\n{function_format}\n"
            "Provide a concise one-sentence summary in braces marked as a comment, followed by the fully executable code."
        )
        try:
            repair_response = language_model.prompt(expertise, repair_prompt, temperature=0.3)
            rep_code = extract_code(repair_response)
            rep_idea = extract_idea(repair_response)
            if rep_code and rep_code.strip():
                rep_score = utility(rep_code, rep_idea, task)
                if rep_score < current_best_utility:
                    current_best_idea = rep_idea
                    current_best_code = rep_code
                    current_best_utility = rep_score
        except Exception:
            pass

    return current_best_idea, current_best_code, current_best_utility
