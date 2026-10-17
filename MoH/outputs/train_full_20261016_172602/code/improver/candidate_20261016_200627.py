# Idea: A diagnostic ruin-and-recreate metaheuristic that replaces greedy elite selection with a Metropolis-Hastings simulated annealing acceptance criterion combined with a fitness-diversity Pareto archive to escape local optima and preserve diverse near-optimal algorithmic structures.
#{A diagnostic ruin-and-recreate metaheuristic that replaces greedy elite selection with a Metropolis-Hastings simulated annealing acceptance criterion combined with a fitness-diversity Pareto archive to escape local optima and preserve diverse near-optimal algorithmic structures.}

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

    # Active state for simulated annealing walk
    curr_state_code = current_best_code
    curr_state_idea = current_best_idea
    curr_state_utility = current_best_utility

    # Step 1: LLM Diagnostic Ruin-and-Recreate Operator Formulation
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate (Iterated Local Search) philosophy:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components to ruin "
        "(e.g., representation, neighborhood exploration, selection pressure, or stagnation handling) and rebuild them with superior algorithmic logic.\n"
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

    # Step 2: Batch Generation of Reconstructed Algorithms
    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic blocks")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")
        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{curr_state_code}\n```\n"
            f"Current Utility: {curr_state_utility} (lower is better)\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=1.0)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    evaluated_candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = float(utility(code, idea, task))
            if not np.isnan(score):
                evaluated_candidates.append((idea, code, score))
        except Exception:
            continue

    if not evaluated_candidates:
        return current_best_idea, current_best_code, current_best_utility

    # Step 3: Fitness-Diversity Pareto Archive Construction
    # Measure structural novelty relative to current best code using token length difference and length ratio
    archive = [(current_best_idea, current_best_code, current_best_utility, 0.0)]
    for idea, code, score in evaluated_candidates:
        # Distance proxy based on length and normalized edit difference
        len_diff = abs(len(code) - len(current_best_code)) / max(1, len(current_best_code))
        div_proxy = float(len_diff)
        archive.append((idea, code, score, div_proxy))

    # Identify non-dominated solutions on (Utility [minimize], Diversity [maximize])
    pareto_front = []
    for sol in archive:
        is_dominated = False
        for other in archive:
            # other dominates sol if other has lower/equal utility AND higher/equal diversity (strictly better in at least one)
            if (other[2] <= sol[2] and other[3] >= sol[3]) and (other[2] < sol[2] or other[3] > sol[3]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(sol)

    # Step 4: Metropolis-Hastings Simulated Annealing Selection
    # Adaptive temperature scaled to current best utility
    temperature = max(1e-4, 0.05 * abs(current_best_utility))

    # Sort candidates by utility
    evaluated_candidates.sort(key=lambda x: x[2])

    # Check if a new global best was found
    if evaluated_candidates[0][2] < current_best_utility:
        current_best_idea, current_best_code, current_best_utility = evaluated_candidates[0]
    else:
        # Probabilistically select a promising Pareto or sub-optimal candidate via Metropolis-Hastings criterion
        for idea, code, score in evaluated_candidates:
            delta = score - curr_state_utility
            if delta < 0:
                acceptance_prob = 1.0
            else:
                acceptance_prob = np.exp(-delta / temperature)
                # Boost acceptance probability if candidate resides in the Pareto diversity front
                if any(p[1] == code for p in pareto_front):
                    acceptance_prob = min(1.0, acceptance_prob * 1.5)

            if np.random.rand() < acceptance_prob:
                curr_state_idea = idea
                curr_state_code = code
                curr_state_utility = score
                break

    return current_best_idea, current_best_code, current_best_utility
