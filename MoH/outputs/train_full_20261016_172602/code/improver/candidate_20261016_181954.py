# Idea: An adaptive ruin-and-recreate metaheuristic that extracts structural invariants via diagnostic decomposition, then queries the LLM across four orthogonal, role-differentiated perturbation operators with concrete algorithmic motifs to achieve balanced multi-directional code optimization.
# Utility: 0.035612740937071036
#{An adaptive ruin-and-recreate metaheuristic that extracts structural invariants via diagnostic decomposition, then queries the LLM across four orthogonal, role-differentiated perturbation operators with concrete algorithmic motifs to achieve balanced multi-directional code optimization.}

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

    # Secondary reference solution for diversity context if available
    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol
    ref_utility = reference_sol.get('utility', current_best_utility)

    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Current Best Idea: {current_best_idea}\n"
        f"Current Best Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Perform a Diagnostic Ruin-and-Recreate decomposition:\n"
        "1. Identify the 'invariable_core': The essential logic, data representations, or algorithmic invariant that must be maintained.\n"
        "2. Identify bottlenecks or suboptimal components ripe for targeted replacement across four specialized axes.\n"
        "Return strictly valid JSON:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<invariant algorithmic logic to retain>",\n'
        '  "target_complexity": "<component with excessive overhead or redundant steps>",\n'
        '  "target_paradigm": "<component stuck in conventional local search to transform radically>",\n'
        '  "target_hyperparameters": "<fixed parameters, acceptance thresholds, or static schedules to make adaptive>",\n'
        '  "target_intensification": "<insufficient local exploitation or weak constraint/boundary handling to harden>"\n'
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Core heuristic search pipeline")
        target_complexity = diag_data.get("target_complexity", "Redundant candidate evaluations and naive state copying")
        target_paradigm = diag_data.get("target_paradigm", "Standard perturbation and single-point mutation")
        target_hyperparams = diag_data.get("target_hyperparameters", "Static temperature and fixed iteration bounds")
        target_intensify = diag_data.get("target_intensification", "Shallow local search and unguided repair logic")
    except Exception:
        invariable_core = "Core metaheuristic evaluation and solution maintenance"
        target_complexity = "Redundant candidate evaluations and naive state copying"
        target_paradigm = "Standard perturbation and single-point mutation"
        target_hyperparams = "Static cooling and acceptance schedules"
        target_intensify = "Shallow local search and unguided repair logic"

    # Define 4 role-differentiated orthogonal perturbation prompts with concrete motifs
    orthogonal_roles = [
        {
            "role": "Role 1: Algorithmic Complexity Reducer & Vectorized Evaluator",
            "target": target_complexity,
            "motif": (
                "Algorithmic Motif: Prune computational redundancies, replace Python-level loops with vectorized "
                "NumPy operations where applicable, implement delta-evaluations or caching, and eliminate deepcopy bottlenecks."
            ),
            "temp": 0.5
        },
        {
            "role": "Role 2: Radical Paradigm Shifter (Exploration Catalyst)",
            "target": target_paradigm,
            "motif": (
                "Algorithmic Motif: Introduce non-standard combinatorial frameworks such as Large Neighborhood Search (LNS) "
                "with structural ruin-and-recreate, Pareto dominance archiving, or population-based cooperative co-evolution."
            ),
            "temp": 0.95
        },
        {
            "role": "Role 3: Self-Adaptive Hyperparameter & Feedback Tuner",
            "target": target_hyperparams,
            "motif": (
                "Algorithmic Motif: Replace all static scalars with dynamic feedback-driven adaptation (e.g., 1/5th success rule, "
                "reactive tabu tenure, exponential or adaptive Cauchy annealing schedules, and stagnation counters)."
            ),
            "temp": 0.7
        },
        {
            "role": "Role 4: Constraint-Robust Intensifier & Elite Polisher",
            "target": target_intensify,
            "motif": (
                "Algorithmic Motif: Deepen the exploitation phase using multi-operator variable neighborhood descent (VND), "
                "guided local repair with frequency-based memory penalties, and rigorous boundary-clamping guards."
            ),
            "temp": 0.65
        }
    ]

    message_batch = []
    temperatures = []
    for r in orthogonal_roles:
        prompt_text = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better, population reference: {ref_utility})\n\n"
            f"ROLE SPECIALIZATION: {r['role']}\n"
            f"PRESERVE INVARIABLE CORE: {invariable_core}\n"
            f"RUIN TARGET: Completely tear down and replace: {r['target']}\n"
            f"RECREATE STRATEGY: {r['motif']}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(prompt_text)
        temperatures.append(r["temp"])

    # Query language model with role-based batched prompts
    # Using an effective representative temperature across the batch
    avg_temp = float(np.mean(temperatures))
    responses = language_model.prompt_batch(expertise, message_batch, temperature=avg_temp)
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
