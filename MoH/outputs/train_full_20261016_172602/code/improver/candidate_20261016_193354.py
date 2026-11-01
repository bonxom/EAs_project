# Idea: An adaptive Ruin-and-Recreate framework with diagnostic sub-component decomposition, self-reflective failure repair, and operator performance tracking to iteratively refine optimization algorithms.
#{An adaptive Ruin-and-Recreate framework with diagnostic sub-component decomposition, self-reflective failure repair, and operator performance tracking to iteratively refine optimization algorithms.}

from utils.utils import extract_code, extract_idea
import json
import traceback

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

    # Historical operator portfolio with dynamic perturbation intensities and tracking
    operator_portfolio = [
        {"name": "Representation & Encoding", "target": "State representation, candidate initialization, and structural bounds", "intensity": "Moderate", "success_count": 0},
        {"name": "Neighborhood Dynamics", "target": "Neighborhood operators, search trajectory exploration, and step size control", "intensity": "High", "success_count": 0},
        {"name": "Selection & Pressure", "target": "Acceptance criteria, fitness exploitation, and elite preservation", "intensity": "Adaptive", "success_count": 0},
        {"name": "Stagnation & Escapes", "target": "Restart triggers, perturbation magnitude, and memory-guided diversity", "intensity": "High", "success_count": 0}
    ]

    # Phase 1: Structured diagnostic decomposition of the current elite
    diagnosis_prompt = (
        f"Analyze the current elite optimization algorithm for task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score: {current_best_utility} (lower is better)\n\n"
        "Apply an Adaptive Ruin-and-Recreate philosophy:\n"
        "1. Identify the 'Invariable Core': The critical algorithmic logic that must be preserved.\n"
        "2. Formulate 4 targeted Ruin-and-Recreate strategies matching the following operator domains:\n"
        f"- {operator_portfolio[0]['name']}: ruin target '{operator_portfolio[0]['target']}'\n"
        f"- {operator_portfolio[1]['name']}: ruin target '{operator_portfolio[1]['target']}'\n"
        f"- {operator_portfolio[2]['name']}: ruin target '{operator_portfolio[2]['target']}'\n"
        f"- {operator_portfolio[3]['name']}: ruin target '{operator_portfolio[3]['target']}'\n\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<brief summary of what must be retained>",\n'
        '  "reconstruction_operators": [\n'
        '    {"domain": "Representation & Encoding", "ruin_target": "<specific code block>", "recreate_strategy": "<advanced reconstruction>"},\n'
        '    {"domain": "Neighborhood Dynamics", "ruin_target": "<specific code block>", "recreate_strategy": "<advanced reconstruction>"},\n'
        '    {"domain": "Selection & Pressure", "ruin_target": "<specific code block>", "recreate_strategy": "<advanced reconstruction>"},\n'
        '    {"domain": "Stagnation & Escapes", "ruin_target": "<specific code block>", "recreate_strategy": "<advanced reconstruction>"}\n'
        '  ]\n'
        "}\n"
        "```"
    )

    invariable_core = "Core metaheuristic loop structure and elite candidate persistence"
    reconstruction_plan = []

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", invariable_core)
        reconstruction_plan = diag_data.get("reconstruction_operators", [])
    except Exception:
        reconstruction_plan = [
            {"domain": op["name"], "ruin_target": op["target"], "recreate_strategy": f"Adaptive {op['name']} refinement with robust bounds"}
            for op in operator_portfolio
        ]

    if not reconstruction_plan:
        reconstruction_plan = [
            {"domain": op["name"], "ruin_target": op["target"], "recreate_strategy": f"Enhanced {op['name']} logic"}
            for op in operator_portfolio
        ]

    # Phase 2: Parallel batched generation with fine-grained perturbation directives
    reconstruct_prompts = []
    for plan in reconstruction_plan[:4]:
        domain = plan.get("domain", "General")
        ruin_target = plan.get("ruin_target", "Inefficient heuristic components")
        recreate_strategy = plan.get("recreate_strategy", "Advanced metaheuristic refinement")

        prompt_content = (
            f"You are executing an Iterated Local Search Ruin-and-Recreate operator on this algorithm:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"OPERATOR DOMAIN: {domain}\n"
            f"RUIN TARGET: Completely tear down and replace: {ruin_target}\n"
            f"RECREATE STRATEGY: Rebuild with: {recreate_strategy}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        reconstruct_prompts.append(prompt_content)

    responses = language_model.prompt_batch(expertise, reconstruct_prompts, temperature=0.9)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    evaluated_candidates = []
    failed_repairs = []

    # Phase 3: Candidate evaluation with error capture for self-reflective repair
    for idx, (idea, code) in enumerate(zip(candidate_ideas, candidate_codes)):
        if not code or not code.strip():
            failed_repairs.append((idx, "Generated code was empty or unparseable."))
            continue
        try:
            score = utility(code, idea, task)
            evaluated_candidates.append((idea, code, score, idx))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            failed_repairs.append((idx, f"Execution error: {str(e)}\nTraceback: {tb}"))

    # Phase 4: Self-reflective error repair for failed candidates
    if failed_repairs:
        repair_prompts = []
        repair_indices = []
        for idx, err_msg in failed_repairs[:2]:  # Focus repair budget on top failures
            broken_code = candidate_codes[idx] if idx < len(candidate_codes) else current_best_code
            repair_prompt = (
                f"A candidate algorithm generated during Ruin-and-Recreate failed during execution on task '{task}'.\n"
                f"Error details:\n{err_msg}\n\n"
                f"Defective Code:\n```python\n{broken_code}\n```\n\n"
                f"Carefully diagnose the crash, fix all runtime and edge case errors, and preserve algorithmic validity.\n"
                f"Format requirements:\n{function_format}\n"
                "Return the fixed implementation starting with a one-sentence comment in braces followed by complete code."
            )
            repair_prompts.append(repair_prompt)
            repair_indices.append(idx)

        if repair_prompts:
            repaired_responses = language_model.prompt_batch(expertise, repair_prompts, temperature=0.4)
            repaired_codes = extract_code(repaired_responses)
            repaired_ideas = extract_idea(repaired_responses)
            for idea, code in zip(repaired_ideas, repaired_codes):
                if code and code.strip():
                    try:
                        score = utility(code, idea, task)
                        evaluated_candidates.append((idea, code, score, -1))
                    except Exception:
                        continue

    # Phase 5: Selection, operator tracking, and elitist update
    if evaluated_candidates:
        best_cand_idea, best_cand_code, best_cand_utility, op_idx = min(evaluated_candidates, key=lambda x: x[2])
        if best_cand_utility < current_best_utility:
            current_best_idea = best_cand_idea
            current_best_code = best_cand_code
            current_best_utility = best_cand_utility
            if 0 <= op_idx < len(operator_portfolio):
                operator_portfolio[op_idx]["success_count"] += 1

    return current_best_idea, current_best_code, current_best_utility
