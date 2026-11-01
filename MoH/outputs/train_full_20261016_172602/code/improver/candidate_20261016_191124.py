# Idea: This metaheuristic combines diagnostic ruin-and-recreate operator generation with a temperature-regulated Metropolis-Hastings acceptance criterion and an elite archive to probabilistically accept structurally diverse, near-optimal candidates and escape local plateaus.
#{This metaheuristic combines diagnostic ruin-and-recreate operator generation with a temperature-regulated Metropolis-Hastings acceptance criterion and an elite archive to probabilistically accept structurally diverse, near-optimal candidates and escape local plateaus.}

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

    # Maintain an elite archive baseline and a current traversal state
    best_overall_code = elite_sol['best_sol']
    best_overall_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    best_overall_utility = float(elite_sol['utility'])

    # Start traversal from elite or a high-performing reference solution
    curr_code = best_overall_code
    curr_idea = best_overall_idea
    curr_utility = best_overall_utility

    diagnosis_prompt = (
        f"Analyze the current algorithm for task '{task}':\n"
        f"Idea: {curr_idea}\n"
        f"Code:\n```python\n{curr_code}\n```\n"
        f"Utility Score (lower is better): {curr_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate decomposition:\n"
        "1. Identify the 'Invariable Core': The critical structural backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components "
        "(e.g., representation, neighborhood exploration, adaptive cooling/acceptance, or stagnation recovery) "
        "to rebuild them with superior algorithmic mechanisms.\n"
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
            {"target_to_ruin": "Selection and acceptance mechanism", "reconstruct_strategy": "Metropolis-Hastings simulated annealing cooling schedule with adaptive history"},
            {"target_to_ruin": "Candidate generation", "reconstruct_strategy": "Guided greedy randomized adaptive construction"},
            {"target_to_ruin": "Stagnation handling", "reconstruct_strategy": "Heavy structural mutation with memory-guided repair"}
        ]

    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic blocks")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")
        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
            f"```python\n{curr_code}\n```\n"
            f"Current Utility: {curr_utility} (lower is better)\n\n"
            f"PRESERVE CORE LOGIC: {invariable_core}\n"
            f"RUIN TARGET: Completely replace and rewrite: {ruin_target}\n"
            f"RECREATE USING: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9)
    candidate_codes = extract_code(responses)
    candidate_ideas = extract_idea(responses)

    evaluated_candidates = []
    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue
        try:
            score = float(utility(code, idea, task))
            evaluated_candidates.append((idea, code, score))
        except Exception:
            continue

    if evaluated_candidates:
        # Sort candidates from best (lowest) to worst
        evaluated_candidates.sort(key=lambda x: x[2])

        # Update the global best elite archive
        best_cand_idea, best_cand_code, best_cand_utility = evaluated_candidates[0]
        if best_cand_utility < best_overall_utility:
            best_overall_idea = best_cand_idea
            best_overall_code = best_cand_code
            best_overall_utility = best_cand_utility

        # Metropolis-Hastings acceptance mechanism to allow non-greedy step transitions
        # Dynamic annealing temperature scaled to baseline magnitude
        temperature = max(abs(curr_utility) * 0.05, 1e-4)
        accepted_step = None

        for cand_idea, cand_code, cand_utility in evaluated_candidates:
            delta = cand_utility - curr_utility
            if delta < 0:
                accepted_step = (cand_idea, cand_code, cand_utility)
                break
            else:
                prob = np.exp(-delta / temperature)
                if np.random.rand() < prob:
                    accepted_step = (cand_idea, cand_code, cand_utility)
                    break

        # If an exploratory step was accepted and is competitive, adopt it; otherwise return elite archive
        if accepted_step is not None:
            curr_idea, curr_code, curr_utility = accepted_step
            if curr_utility <= best_overall_utility:
                return curr_idea, curr_code, curr_utility

    return best_overall_idea, best_overall_code, best_overall_utility
