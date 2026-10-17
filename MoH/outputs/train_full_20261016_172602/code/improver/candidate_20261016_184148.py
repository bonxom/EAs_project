# Idea: A co-evolutionary portfolio-driven Ruin-and-Recreate metaheuristic that synergistically couples structured diagnostic decomposition with complementary role-assigned candidate generation spanning aggressive diversification, guided crossover, fine-grained exploitation, and repair.
#{A co-evolutionary portfolio-driven Ruin-and-Recreate metaheuristic that synergistically couples structured diagnostic decomposition with complementary role-assigned candidate generation spanning aggressive diversification, guided crossover, fine-grained exploitation, and repair.}

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

    # Retrieve a diverse reference solution for co-evolutionary cross-fertilization
    reference_sol = (
        population.get_solution_by_index(task, min(2, pop_size - 1)) 
        if pop_size > 2 else population.get_random_solution(task)
    )

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    ref_code = reference_sol.get('best_sol', current_best_code)
    ref_idea = reference_sol.get('idea', 'Secondary reference heuristic')
    ref_utility = reference_sol.get('utility', current_best_utility)

    # Phase 1: Structured Diagnosis & Invariant Decomposition
    diagnosis_prompt = (
        f"Analyze the elite algorithmic architecture for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Current Utility: {current_best_utility} (lower is better)\n\n"
        "Diagnose algorithmic bottlenecks to prepare a collaborative Ruin-and-Recreate portfolio.\n"
        "Return your analysis strictly in JSON format:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<invariant algorithmic core that guarantees correctness and performance>",\n'
        '  "critical_weakness": "<primary performance bottleneck or premature convergence risk>",\n'
        '  "promising_mechanisms": ["<mechanism 1>", "<mechanism 2>"]\n'
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Two-phase selection and adaptive local search progression")
        critical_weakness = diag_data.get("critical_weakness", "Stagnation and lack of complementary search directions")
    except Exception:
        invariable_core = "Two-phase selection and adaptive local search progression"
        critical_weakness = "Stagnation and lack of complementary search directions"

    # Phase 2: Role-Assigned Co-Evolutionary Portfolio Formulation
    # Spawn heterogeneous candidates with dedicated roles: Exploiter, Explorer, Hybrid/Crossover, and Structural Repair
    portfolio_roles = [
        {
            "role": "INTENSIFICATION_EXPLOITER",
            "ruin_instruction": "Refine and deeply optimize neighborhood search, state transitions, and step-size adaptation around current elite basins.",
            "recreate_mechanism": "Adaptive multi-scale local descent with elite memory and high-efficiency gradient/heuristic guidance.",
            "temperature": 0.5
        },
        {
            "role": "DIVERSIFICATION_EXPLORER",
            "ruin_instruction": "Completely dismantle rigid neighborhood topologies and deterministic selection to escape local minima.",
            "recreate_mechanism": "Orthogonal multi-neighborhood ruin-and-recreate with non-monotonic acceptance and dynamic temperature jumps.",
            "temperature": 1.0
        },
        {
            "role": "CO_EVOLUTIONARY_HYBRID",
            "ruin_instruction": f"Synthesize core advantages of the Elite (Utility: {current_best_utility}) and the Partner Reference (Utility: {ref_utility}).",
            "recreate_mechanism": f"Recombinant crossover that extracts complementary logic from Reference Idea: '{ref_idea}' and blends it with Elite invariant core.",
            "temperature": 0.8
        },
        {
            "role": "STRUCTURAL_REPAIR_AND_SURGE",
            "ruin_instruction": f"Identify and aggressively eliminate the diagnosed weakness: {critical_weakness}.",
            "recreate_mechanism": "Robust fallback heuristics, self-adaptive perturbation thresholds, and fail-safe diversification schedules.",
            "temperature": 0.9
        }
    ]

    message_batch = []
    for member in portfolio_roles:
        reconstruct_prompt = (
            f"You are an expert algorithm designer executing an Iterated Co-Evolutionary Ruin-and-Recreate optimization.\n"
            f"ELITE BASELINE SOLUTION:\n```python\n{current_best_code}\n```\n"
            f"Baseline Utility: {current_best_utility} (lower is better)\n"
            f"INVARIABLE CORE TO PRESERVE: {invariable_core}\n\n"
            f"ASSIGNED PORTFOLIO ROLE: {member['role']}\n"
            f"RUIN DIRECTIVE: {member['ruin_instruction']}\n"
            f"RECREATE DIRECTIVE: {member['recreate_mechanism']}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids infinite loops, adheres strictly to the interface, handles edge cases, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    # Batched generation for computational speed and variance
    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.85)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    # Phase 3: Portfolio Evaluation and Elitism Selection
    candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            if score is not None and not np.isnan(score):
                candidates.append((idea, code, float(score)))
        except Exception:
            continue

    if candidates:
        best_candidate_idea, best_candidate_code, best_candidate_utility = min(candidates, key=lambda x: x[2])
        if best_candidate_utility < current_best_utility:
            current_best_idea = best_candidate_idea
            current_best_code = best_candidate_code
            current_best_utility = best_candidate_utility

    return current_best_idea, current_best_code, current_best_utility
