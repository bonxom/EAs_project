# Idea: A structured dual-phase metaheuristic combining analytical JSON-guided ruin-and-recreate diagnostics with temperature-bifurcated candidate reconstruction that simultaneously drives high-temperature exploration and low-temperature exploitation.
#{A structured dual-phase metaheuristic combining analytical JSON-guided ruin-and-recreate diagnostics with temperature-bifurcated candidate reconstruction that simultaneously drives high-temperature exploration and low-temperature exploitation.}

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
    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # Phase 1: Structural Diagnosis via Structured Ruin-and-Recreate Formulation
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate philosophy:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components:\n"
        "   - 2 Radical Exploration Operators: High-variance structural overhauls, representation changes, or novel metaheuristic hybrids.\n"
        "   - 2 Precision Intensification Operators: Targeted algorithmic micro-optimizations, adaptive parameter control, or aggressive neighborhood exploitation.\n"
        "Return your analysis strictly in JSON format as follows:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<brief summary of what must be retained>",\n'
        '  "exploration_operators": [\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative radical recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific component to tear down>", "reconstruct_strategy": "<innovative radical recreation mechanism>"}\n'
        '  ],\n'
        '  "intensification_operators": [\n'
        '    {"target_to_ruin": "<specific component to refine>", "reconstruct_strategy": "<precise micro-optimization mechanism>"},\n'
        '    {"target_to_ruin": "<specific component to refine>", "reconstruct_strategy": "<precise micro-optimization mechanism>"}\n'
        '  ]\n'
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core heuristic search progression")
        explore_ops = diag_data.get("exploration_operators", [])
        intensify_ops = diag_data.get("intensification_operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression"
        explore_ops = [
            {"target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator"},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Heavy structural mutation with memory-guided diversity restarts"}
        ]
        intensify_ops = [
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Deterministic non-monotonic threshold acceptance with fine-grained temperature cooling"},
            {"target_to_ruin": "Candidate refinement", "reconstruct_strategy": "Micro-gradient proxy and adaptive elitist parameter tuning"}
        ]

    # Ensure fallback minimum lengths
    if len(explore_ops) < 2:
        explore_ops = explore_ops + [
            {"target_to_ruin": "Global search mechanism", "reconstruct_strategy": "Co-evolutionary dual-population diversification"}
        ] * (2 - len(explore_ops))
    if len(intensify_ops) < 2:
        intensify_ops = intensify_ops + [
            {"target_to_ruin": "Solution refinement loop", "reconstruct_strategy": "Greedy sequential local search polishing"}
        ] * (2 - len(intensify_ops))

    # Phase 2: Role-Specialized Reconstruction with Temperature Bifurcation
    # High-temperature exploration batch
    explore_batch = []
    for op in explore_ops[:2]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic blocks")
        rebuild_strat = op.get("reconstruct_strategy", "Radical structural innovation")
        explore_prompt = (
            f"You are an EXPLORATION SPECIALIST executing an Aggressive Ruin-and-Recreate on:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely redesign and replace: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n"
            f"MINDSET: Bold paradigm shift, explore unvisited algorithmic topologies, prevent premature local convergence.\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm implements robust edge handling and avoids unbounded loops."
        )
        explore_batch.append(explore_prompt)

    # Low-temperature intensification batch
    intensify_batch = []
    for op in intensify_ops[:2]:
        ruin_target = op.get("target_to_ruin", "Suboptimal parameter schedules or acceptance gates")
        rebuild_strat = op.get("reconstruct_strategy", "Precision tuning and greedy exploitation")
        intensify_prompt = (
            f"You are an INTENSIFICATION SPECIALIST executing a High-Precision Polish on:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Micro-optimize and streamline: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n"
            f"MINDSET: Numerical precision, optimal greedy decisions, minimal overhead, deterministic quality guarantees.\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm implements robust edge handling and optimizes execution efficiency."
        )
        intensify_batch.append(intensify_prompt)

    # Execute decoupled batches with distinct temperatures
    explore_responses = language_model.prompt_batch(expertise, explore_batch, temperature=1.0)
    intensify_responses = language_model.prompt_batch(expertise, intensify_batch, temperature=0.3)

    all_responses = explore_responses + intensify_responses
    candidate_codes = extract_code(all_responses)
    candidate_ideas = extract_idea(all_responses)

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
