# Idea: A meta-algorithmic Ruin-and-Recreate framework that profiles population performance disparities and algorithmic bottlenecks to dynamically attribute and target ruin operators towards empirical failure modes prior to parallel LLM reconstruction and elitist selection.
#{A meta-algorithmic Ruin-and-Recreate framework that profiles population performance disparities and algorithmic bottlenecks to dynamically attribute and target ruin operators towards empirical failure modes prior to parallel LLM reconstruction and elitist selection.}

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

    # Collect performance profile across top population solutions to extract empirical bottleneck telemetry
    history_telemetry = []
    sample_count = min(pop_size, 4)
    for idx in range(sample_count):
        sol = population.get_solution_by_index(task, idx)
        history_telemetry.append({
            "rank": idx,
            "utility": sol['utility'],
            "idea_snippet": str(sol.get('idea', 'Heuristic variant'))[:120]
        })

    diversity_gap = 0.0
    if pop_size > 1:
        worst_sampled = population.get_solution_by_index(task, pop_size - 1)
        diversity_gap = abs(worst_sampled['utility'] - current_best_utility)

    diagnosis_prompt = (
        f"Perform a Performance-Guided Bottleneck Attribution analysis for the metaheuristic on task '{task}':\n"
        f"Incumbent Code:\n```python\n{current_best_code}\n```\n"
        f"Incumbent Utility: {current_best_utility} (lower is better)\n"
        f"Population Utility Spread Gap: {diversity_gap:.6f}\n"
        f"Recent Population Telemetry: {json.dumps(history_telemetry)}\n\n"
        "Instructions:\n"
        "1. Identify the 'invariable_core': The indispensable algorithmic backbone delivering competitive performance that MUST NOT be broken.\n"
        "2. Conduct Bottleneck Attribution: Attribute where the algorithm currently experiences premature convergence, search stagnation, computational waste, or weak exploration.\n"
        "3. Produce 4 targeted Ruin-and-Recreate operators specifically conditioned on the identified bottlenecks with distinct perturbation intensities.\n\n"
        "Return strictly valid JSON:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<indispensable search backbone to preserve>",\n'
        '  "bottleneck_diagnosis": "<identified performance deficiency or stagnation cause>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific deficient component>", "reconstruct_strategy": "<innovative recreation mechanism>", "intensity": "high"},\n'
        '    {"target_to_ruin": "<specific deficient component>", "reconstruct_strategy": "<innovative recreation mechanism>", "intensity": "medium"},\n'
        '    {"target_to_ruin": "<specific deficient component>", "reconstruct_strategy": "<innovative recreation mechanism>", "intensity": "exploratory"},\n'
        '    {"target_to_ruin": "<specific deficient component>", "reconstruct_strategy": "<innovative recreation mechanism>", "intensity": "intensification"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_json_str = extract_code(diag_response)
        diag_data = json.loads(diag_json_str if diag_json_str else diag_response)
        invariable_core = diag_data.get("invariable_core", "Core search pipeline and state representation")
        bottleneck_diag = diag_data.get("bottleneck_diagnosis", "Search stagnation and neighborhood limitation")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core search pipeline and state representation"
        bottleneck_diag = "Premature convergence and insufficient exploration around local optima"
        operators = [
            {"target_to_ruin": "Neighborhood perturbation and move dynamics", "reconstruct_strategy": "Adaptive variable-depth destroy and multi-operator repair", "intensity": "high"},
            {"target_to_ruin": "Acceptance criteria and stagnation control", "reconstruct_strategy": "Non-monotonic adaptive cooling schedule with tabu-guided escapes", "intensity": "medium"},
            {"target_to_ruin": "Candidate generation policy", "reconstruct_strategy": "Guided greedy-randomized construction with frequency-based memory", "intensity": "exploratory"},
            {"target_to_ruin": "Intensification routines", "reconstruct_strategy": "Path relinking between elite historical states and current incumbent", "intensity": "intensification"}
        ]

    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Local search perturbations")
        rebuild_strat = op.get("reconstruct_strategy", "Adaptive neighborhood exploration")
        intensity = op.get("intensity", "balanced")

        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on task '{task}'.\n"
            f"Current Incumbent Code:\n```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n"
            f"IDENTIFIED BOTTLENECK: {bottleneck_diag}\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely tear down and replace: {ruin_target}\n"
            f"RECREATE STRATEGY ({intensity} mode): {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids infinite loops, adheres to function signatures, and handles boundary conditions cleanly."
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
            if score is not None and not np.isnan(score):
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
