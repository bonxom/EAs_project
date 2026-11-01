# Idea: An adaptive Ruin-and-Recreate metaheuristic employing a two-phase diagnostic decomposition coupled with a multi-arm exploration-intensification temperature ladder that dynamically scales sampling temperatures from conservative exploitation to high-entropy structural diversification across operator prompts.
#{An adaptive Ruin-and-Recreate metaheuristic employing a two-phase diagnostic decomposition coupled with a multi-arm exploration-intensification temperature ladder that dynamically scales sampling temperatures from conservative exploitation to high-entropy structural diversification across operator prompts.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    # Retrieve current elite and comparative solutions
    pop_size = population.get_subtask_size(task)
    elite_sol = population.get_solution_by_index(task, 0)
    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol
    ref_utility = reference_sol.get('utility', current_best_utility)

    # Phase 1: Diagnostic Ruin-and-Recreate decomposition
    stagnation_risk = "high" if abs(current_best_utility - ref_utility) < 1e-6 else "low"
    diagnosis_prompt = (
        f"Analyze the current best algorithm for task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n"
        f"Population diversity status: stagnation_risk={stagnation_risk}\n\n"
        "Deconstruct the architecture using Ruin-and-Recreate principles into 4 structured arms ranging from exploitation to radical exploration:\n"
        "Arm 1 (Micro-Refinement): Fine-grained exploitation targeting local search parameters, acceptance conditions, or fast neighborhood sorting.\n"
        "Arm 2 (Balanced Perturbation): Mid-level reconfiguration replacing neighborhood structures or balancing search phases.\n"
        "Arm 3 (Macro Ruin-and-Recreate): Major structural tear-down and rebuild of the core heuristic representation or search engine.\n"
        "Arm 4 (Radical Paradigm Shift): High-entropy diversification introducing novel algorithmic paradigms or anti-stagnation mechanisms.\n\n"
        "Return strictly valid JSON:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<essential structural backbone to preserve>",\n'
        '  "arms": [\n'
        '    {"arm_type": "micro_refinement", "target_to_ruin": "<specific component>", "reconstruct_strategy": "<precise improvement>"},\n'
        '    {"arm_type": "balanced_perturbation", "target_to_ruin": "<specific component>", "reconstruct_strategy": "<balanced redesign>"},\n'
        '    {"arm_type": "macro_ruin_recreate", "target_to_ruin": "<specific component>", "reconstruct_strategy": "<major structural overhaul>"},\n'
        '    {"arm_type": "radical_shift", "target_to_ruin": "<specific component>", "reconstruct_strategy": "<novel heuristic paradigm>"}\n'
        '  ]\n'
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core search pipeline and representation")
        arms = diag_data.get("arms", [])
    except Exception:
        invariable_core = "Core iterative search progression and state management"
        arms = [
            {"arm_type": "micro_refinement", "target_to_ruin": "Local tuning parameters and cooling schedule", "reconstruct_strategy": "Self-adaptive parameter control with responsive cooling"},
            {"arm_type": "balanced_perturbation", "target_to_ruin": "Local perturbation operator", "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild search"},
            {"arm_type": "macro_ruin_recreate", "target_to_ruin": "Candidate generation and search memory", "reconstruct_strategy": "Path-relinking memory mechanism with guided construction"},
            {"arm_type": "radical_shift", "target_to_ruin": "Global search topology", "reconstruct_strategy": "Cooperative multi-agent co-evolutionary search with anti-stagnation jump"}
        ]

    # Ensure exactly 4 arms
    default_arms = [
        {"arm_type": "micro_refinement", "target_to_ruin": "Local tuning", "reconstruct_strategy": "Self-adaptive parameters"},
        {"arm_type": "balanced_perturbation", "target_to_ruin": "Neighborhood operator", "reconstruct_strategy": "Multi-neighborhood search"},
        {"arm_type": "macro_ruin_recreate", "target_to_ruin": "Search memory", "reconstruct_strategy": "Guided path-relinking"},
        {"arm_type": "radical_shift", "target_to_ruin": "Search topology", "reconstruct_strategy": "Co-evolutionary multi-phase exploration"}
    ]
    while len(arms) < 4:
        arms.append(default_arms[len(arms)])

    # Phase 2: Multi-Arm Temperature Ladder Execution
    # Micro-refinement exploits at low temperature; Radical shifts explore at high temperature
    temperature_ladder = [0.3, 0.65, 0.95, 1.2]

    # If the population is stagnating, elevate temperatures slightly across the ladder
    if stagnation_risk == "high":
        temperature_ladder = [t + 0.1 for t in temperature_ladder]

    candidates = []

    # Build prompts matching each arm with its calibrated temperature
    # We group calls or query by calibrated temperature
    for arm, temp in zip(arms[:4], temperature_ladder):
        arm_type = arm.get("arm_type", "reconstruction")
        ruin_target = arm.get("target_to_ruin", "Inefficient heuristic segments")
        rebuild_strat = arm.get("reconstruct_strategy", "Advanced search intensification")

        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution for task '{task}':\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n"
            f"Optimization Tier: {arm_type.upper()} (Sampling Temperature: {temp:.2f})\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely tear down and replace: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )

        try:
            # Query LLM at the calibrated temperature for this specific arm
            response = language_model.prompt(expertise, reconstruct_prompt, temperature=temp)
            code = extract_code(response)
            idea = extract_idea(response)

            if code and code.strip():
                score = utility(code, idea, task)
                candidates.append((idea, code, score))
        except Exception:
            continue

    # Phase 3: Selection and Elitism Update
    if candidates:
        iter_best_idea, iter_best_code, iter_best_utility = min(candidates, key=lambda x: x[2])
        if iter_best_utility < current_best_utility:
            current_best_idea = iter_best_idea
            current_best_code = iter_best_code
            current_best_utility = iter_best_utility

    return current_best_idea, current_best_code, current_best_utility
