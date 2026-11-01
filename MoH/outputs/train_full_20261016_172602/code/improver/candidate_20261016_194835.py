# Idea: A performance-aware reflective ruin-and-recreate metaheuristic that measures historical stagnation and utility divergence across elite solutions to dynamically scale ruin depth from micro-refinement to macro-architectural overhaul, evaluating parallelized candidate reconstructions through robust score filtering.
#{A performance-aware reflective ruin-and-recreate metaheuristic that measures historical stagnation and utility divergence across elite solutions to dynamically scale ruin depth from micro-refinement to macro-architectural overhaul, evaluating parallelized candidate reconstructions through robust score filtering.}

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
    current_best_utility = float(elite_sol['utility'])

    # Profile population stagnation and diversity
    all_scores = []
    for idx in range(min(pop_size, 5)):
        sol = population.get_solution_by_index(task, idx)
        if sol and 'utility' in sol:
            all_scores.append(float(sol['utility']))

    if len(all_scores) > 1:
        stagnation_spread = max(all_scores) - min(all_scores)
        relative_stagnation = stagnation_spread / (abs(current_best_utility) + 1e-8)
    else:
        relative_stagnation = 0.0

    # Determine dynamic perturbation spectrum based on stagnation metrics
    is_deeply_stagnant = relative_stagnation < 0.02 or pop_size <= 2

    # Adaptive Ruin-and-Recreate Diagnostic Phase
    diagnosis_prompt = (
        f"You are conducting a Performance-Aware Reflective Ruin-and-Recreate diagnosis on '{task}'.\n"
        f"Current Best Utility: {current_best_utility:.6f} (lower is better).\n"
        f"Population Size: {pop_size}, Top Utility Spread: {relative_stagnation:.4e} "
        f"({'High Stagnation: Triggering macro architectural ruins' if is_deeply_stagnant else 'Active Search: Triggering balanced multi-scale ruins'}).\n\n"
        f"Current Algorithm Idea:\n{current_best_idea}\n\n"
        f"Current Algorithm Implementation:\n```python\n{current_best_code}\n```\n\n"
        "Analyze the algorithmic execution bottlenecks, asymptotic bottlenecks, and structural vulnerabilities.\n"
        "Generate 4 targeted Ruin-and-Recreate instructions with graduated ruin depths:\n"
        "- Operator 1 (Surgical / Micro-repair): Target local exploitation, parameter cooling, or fine-grained heuristics.\n"
        "- Operator 2 (Subsystem Ruin): Target neighborhood exploration, candidate generation, or perturbing operators.\n"
        "- Operator 3 (Structural Overhaul): Target memory retention, stagnation recovery, or multi-phase scheduling.\n"
        "- Operator 4 (Macro / Paradigm Shift): Tear down the dominant mechanism and rebuild around orthogonal metaheuristic concepts.\n\n"
        "Return your response strictly as valid JSON formatted as:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<invariant core algorithmic principles that must be retained>",\n'
        '  "reconstruction_operators": [\n'
        '    {"depth": "micro", "target_to_ruin": "<specific localized function/routine to ruin>", "reconstruct_strategy": "<exact innovative recreation>"},\n'
        '    {"depth": "subsystem", "target_to_ruin": "<specific search component to ruin>", "reconstruct_strategy": "<exact innovative recreation>"},\n'
        '    {"depth": "structural", "target_to_ruin": "<macro state/selection mechanism to ruin>", "reconstruct_strategy": "<exact innovative recreation>"},\n'
        '    {"depth": "paradigm", "target_to_ruin": "<fundamental assumption/pipeline to replace>", "reconstruct_strategy": "<exact innovative recreation>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core objective evaluation and progressive solution improvement")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core robust evaluation pipeline and non-divergent solution state transitions"
        operators = [
            {"depth": "micro", "target_to_ruin": "Step-size and local acceptance threshold", "reconstruct_strategy": "Self-adaptive noise injection and dynamic temperature normalization"},
            {"depth": "subsystem", "target_to_ruin": "Local neighborhood generation", "reconstruct_strategy": "Pareto-guided adaptive multi-neighborhood search with tabu history"},
            {"depth": "structural", "target_to_ruin": "Stagnation detection and restart handling", "reconstruct_strategy": "Entropy-driven ruin-and-recreate with long-term frequency memory"},
            {"depth": "paradigm", "target_to_ruin": "Single-trajectory local search framework", "reconstruct_strategy": "Dual-phase cooperative co-evolution with surrogate ranking and path relinking"}
        ]

    # Parallel Reconstruct Phase with Temperature Diversity
    message_batch = []
    temperatures = [0.4, 0.7, 0.9, 1.0]

    for op in operators[:4]:
        depth = op.get("depth", "adaptive")
        ruin_target = op.get("target_to_ruin", "Bottleneck search component")
        rebuild_strat = op.get("reconstruct_strategy", "Superior combinatorial exploration mechanic")

        reconstruct_prompt = (
            f"Apply Ruin-and-Recreate at Ruin-Depth: [{depth.upper()}] to optimize the following algorithm:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility:.6f} (lower is better)\n\n"
            f"CORE CONTRACT (Must retain): {invariable_core}\n"
            f"RUIN TARGET (Completely replace and discard): {ruin_target}\n"
            f"RECONSTRUCTION MECHANISM: {rebuild_strat}\n\n"
            f"Requirements:\n{function_format}\n"
            "First, output a concise one-sentence description of the new metaheuristic enclosed in braces marked as a comment. "
            "Then, provide the complete, standalone, self-contained Python function implementation. "
            "Ensure rigorous boundary verification, zero external non-standard dependencies, and protection against premature convergence."
        )
        message_batch.append(reconstruct_prompt)

    try:
        # Prompt model in batch for efficiency and diversity
        responses = language_model.prompt_batch(expertise, message_batch, temperature=0.85)
        candidate_codes = extract_code(responses)
        candidate_ideas = extract_idea(responses)
    except Exception:
        candidate_codes = []
        candidate_ideas = []

    candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip() or len(code.strip()) < 50:
            continue
        try:
            score = utility(code, idea, task)
            if score is not None and not np.isnan(score) and not np.isinf(score):
                candidates.append((idea, code, float(score)))
        except Exception:
            continue

    if candidates:
        iter_best_idea, iter_best_code, iter_best_utility = min(candidates, key=lambda x: x[2])
        if iter_best_utility < current_best_utility:
            current_best_idea = iter_best_idea
            current_best_code = iter_best_code
            current_best_utility = iter_best_utility

    return current_best_idea, current_best_code, current_best_utility
