# Idea: A Ruin-and-Recreate metaheuristic combining structured multi-operator LLM diagnosis with a diversity-aware Metropolis-Hastings acceptance criterion to dynamically balance strict elitism and exploratory leaps across iteration landscapes.
# {A Ruin-and-Recreate metaheuristic combining structured multi-operator LLM diagnosis with a diversity-aware Metropolis-Hastings acceptance criterion to dynamically balance strict elitism and exploratory leaps across iteration landscapes.}

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

    # Measure baseline population diversity via solution length / utility variance
    all_utilities = []
    for idx in range(min(pop_size, 5)):
        sol = population.get_solution_by_index(task, idx)
        if sol and 'utility' in sol:
            all_utilities.append(sol['utility'])
    pop_utility_std = float(np.std(all_utilities)) if len(all_utilities) > 1 else 0.05

    # Phase 1: Structured Diagnosis & Decomposition
    diagnosis_prompt = (
        f"Analyze the current elite algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate (Iterated Local Search) philosophy:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components "
        "(e.g., representation, neighborhood search, diversification/restart mechanisms, or adaptive parameter control).\n"
        "Return your analysis strictly in valid JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<concise summary of invariant search backbone>",\n'
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
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_json_str = extract_code(diag_response)
        diag_data = json.loads(diag_json_str) if diag_json_str else {}
        invariable_core = diag_data.get("invariable_core", "Two-phase modular search pipeline")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Two-phase modular search pipeline"
        operators = []

    if not operators or len(operators) < 4:
        operators = [
            {"target_to_ruin": "Static perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator"},
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Metropolis-Hastings simulated annealing with diversity bonus"},
            {"target_to_ruin": "Candidate construction", "reconstruct_strategy": "Information-theoretic entropy-guided adaptive sampling"},
            {"target_to_ruin": "Stagnation detection", "reconstruct_strategy": "Dynamic hyper-heuristic diversification with memory restarts"}
        ]

    # Phase 2: Parallel Batch Re-creation
    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Local search routine")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced metaheuristic diversification")

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

    evaluated_candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            if score is not None and not np.isnan(score):
                evaluated_candidates.append((idea, code, float(score)))
        except Exception:
            continue

    if not evaluated_candidates:
        return current_best_idea, current_best_code, current_best_utility

    # Phase 3: Diversity-Aware Metropolis-Hastings Probabilistic Acceptance
    # Identify the best candidate from the current batch
    batch_best_idea, batch_best_code, batch_best_utility = min(evaluated_candidates, key=lambda x: x[2])

    if batch_best_utility < current_best_utility:
        # Strict improvement is unconditionally accepted
        return batch_best_idea, batch_best_code, batch_best_utility
    else:
        # Controlled probabilistic acceptance for near-elite solutions to escape local optima
        # Temperature is scaled dynamically with population diversity
        temperature = max(0.01, pop_utility_std * 0.5)

        # Calculate structural novelty / length difference as a diversity heuristic
        code_diff = abs(len(batch_best_code) - len(current_best_code)) / max(1, len(current_best_code))
        diversity_factor = 1.0 + min(1.0, code_diff)

        delta = (batch_best_utility - current_best_utility) / (current_best_utility + 1e-8)
        acceptance_prob = np.exp(-delta / (temperature * diversity_factor))

        if np.random.rand() < min(0.35, acceptance_prob):
            return batch_best_idea, batch_best_code, batch_best_utility

    return current_best_idea, current_best_code, current_best_utility
