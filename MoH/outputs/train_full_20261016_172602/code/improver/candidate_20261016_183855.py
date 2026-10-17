# Idea: An Adaptive Ruin-and-Recreate metaheuristic featuring historical population trajectory profiling, dynamic perturbation scaling across structural tiers, and dual-parent genetic operator synthesis evaluated in parallel.
# {An Adaptive Ruin-and-Recreate metaheuristic featuring historical population trajectory profiling, dynamic perturbation scaling across structural tiers, and dual-parent genetic operator synthesis evaluated in parallel.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics, combinatorial optimization, and algorithm engineering. "
        "Your mission is to invent competitive, mathematically sound, and computationally efficient optimization algorithms."
    )

    pop_size = population.get_subtask_size(task)
    elite_sol = population.get_solution_by_index(task, 0)

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite Baseline Search Architecture')
    current_best_utility = elite_sol['utility']

    # Extract population trajectory and variance to inform perturbation scales
    history_profiles = []
    sample_indices = sorted(list({0, min(1, pop_size - 1), min(2, pop_size - 1), max(0, pop_size // 2)}))
    for idx in sample_indices:
        sol = population.get_solution_by_index(task, idx)
        history_profiles.append({
            "rank": idx,
            "utility": sol['utility'],
            "idea_summary": sol.get('idea', 'Unknown heuristic')[:120]
        })

    random_guide = population.get_random_solution(task)
    guide_code = random_guide['best_sol']
    guide_utility = random_guide['utility']
    guide_idea = random_guide.get('idea', 'Secondary reference heuristic')

    # Adaptive Meta-Feedback Diagnosis with Tiered Perturbation Scales
    diagnosis_prompt = (
        f"Perform an Adaptive Meta-Diagnostic Ruin-and-Recreate assessment on task '{task}'.\n\n"
        f"Population Utility Trajectory (lower is better):\n{json.dumps(history_profiles, indent=2)}\n\n"
        f"Current Elite Solution (Utility: {current_best_utility}):\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n\n"
        f"Diverse Secondary Reference (Utility: {guide_utility}):\n"
        f"Idea: {guide_idea}\n\n"
        "Analyze why the elite algorithm might stagnate and generate exactly 4 Ruin-and-Recreate operators "
        "calibrated across dynamic perturbation scales:\n"
        "1. Scale 'Micro-Repair': Ruin a fine-grained subcomponent (e.g., acceptance threshold, cooling schedule, local step size) while preserving core search mechanics.\n"
        "2. Scale 'Meso-Refactor': Ruin and reconstruct the primary neighborhood exploration or candidate generation strategy.\n"
        "3. Scale 'Macro-Rebuild': Aggressively tear down the main loop orchestration, stagnation detection, and diversification/restart architecture.\n"
        "4. Scale 'Hybrid-Crossover': Synthesize the elite solution's core invariant with the distinct mechanism from the secondary reference.\n\n"
        "Respond strictly in valid JSON matching this schema:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<invariant algorithmic strengths to preserve>",\n'
        '  "stagnation_bottleneck": "<primary structural flaw causing suboptimality>",\n'
        '  "operators": [\n'
        '    {"scale": "Micro-Repair", "target_to_ruin": "<specific target>", "reconstruct_strategy": "<innovative repair>"},\n'
        '    {"scale": "Meso-Refactor", "target_to_ruin": "<specific target>", "reconstruct_strategy": "<innovative refactor>"},\n'
        '    {"scale": "Macro-Rebuild", "target_to_ruin": "<specific target>", "reconstruct_strategy": "<innovative redesign>"},\n'
        '    {"scale": "Hybrid-Crossover", "target_to_ruin": "<specific target>", "reconstruct_strategy": "<innovative hybridization>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Elite optimization backbone and memory preservation")
        bottleneck = diag_data.get("stagnation_bottleneck", "Premature search convergence and neighborhood entrapment")
        operators = diag_data.get("operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression and elite tracking"
        bottleneck = "Stochastic stagnation and lack of adaptive perturbation scaling"
        operators = [
            {"scale": "Micro-Repair", "target_to_ruin": "Step-size and cooling schedule", "reconstruct_strategy": "Non-monotonic adaptive thermodynamic schedule with adaptive temperature reset"},
            {"scale": "Meso-Refactor", "target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Multi-neighborhood variable depth search with biased roulette selection"},
            {"scale": "Macro-Rebuild", "target_to_ruin": "Outer loop control and stagnation escape", "reconstruct_strategy": "Memory-directed tabu ruin-and-recreate with long-term frequency penalty"},
            {"scale": "Hybrid-Crossover", "target_to_ruin": "Isolated elite search policy", "reconstruct_strategy": "Path-relinking crossover marrying the elite core with diverse exploration guidance"}
        ]

    message_batch = []
    for op in operators[:4]:
        scale = op.get("scale", "Meso-Refactor")
        target = op.get("target_to_ruin", "Inefficient heuristic mechanisms")
        strategy = op.get("reconstruct_strategy", "Advanced adaptive search mechanisms")

        reconstruct_prompt = (
            f"You are executing an Adaptive Ruin-and-Recreate transformation at scale '{scale}'.\n\n"
            f"TARGET CODE TO REFINE (Elite Utility: {current_best_utility}):\n"
            f"```python\n{current_best_code}\n```\n\n"
            f"IDENTIFIED BOTTLENECK: {bottleneck}\n"
            f"PRESERVE INVARIABLE CORE: {invariable_core}\n"
            f"COMPONENT TO RUIN AND REMOVE: {target}\n"
            f"RECONSTRUCTION DIRECTIVE: {strategy}\n\n"
            f"Format requirements:\n{function_format}\n\n"
            "Constraints:\n"
            "- First, output a one-sentence summary of the new idea enclosed in braces and marked as a comment: #{...}.\n"
            "- Provide the complete, clean, self-contained Python function 'improve_algorithm'.\n"
            "- Fully define all logic, imports, and variables; no ellipses (...) or placeholders.\n"
            "- Implement robust edge-case handling and protect against runtime failures or infinite loops."
        )
        message_batch.append(reconstruct_prompt)

    # Parallel generation of candidates across distinct perturbation scales
    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.85)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
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
