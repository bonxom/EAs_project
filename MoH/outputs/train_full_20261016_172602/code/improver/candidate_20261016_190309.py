# Idea: A population-guided path relinking metaheuristic that extracts structural consensus invariants across elite solutions and subjects divergent algorithmic blocks to targeted ruin-and-recreate crossover perturbations.
# {A population-guided path relinking metaheuristic that extracts structural consensus invariants across elite solutions and subjects divergent algorithmic blocks to targeted ruin-and-recreate crossover perturbations.}

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

    # Select complementary reference solutions for structural consensus and path relinking
    reference_solutions = []
    if pop_size > 1:
        # Choose runner-up elite
        reference_solutions.append(population.get_solution_by_index(task, 1))
    if pop_size > 2:
        # Choose a diverse / random solution from the population
        reference_solutions.append(population.get_random_solution(task))
    elif pop_size == 1:
        reference_solutions.append(elite_sol)

    # Prepare multi-solution context for comparative structural diagnosis
    ref_summaries = []
    for idx, ref in enumerate(reference_solutions):
        ref_idea = ref.get('idea', f'Reference candidate {idx+1}')
        ref_util = ref.get('utility', 'N/A')
        ref_code = ref.get('best_sol', '')
        ref_summaries.append(
            f"--- Reference Solution {idx+1} (Utility: {ref_util}) ---\n"
            f"Idea: {ref_idea}\n"
            f"Code snippet / structure:\n```python\n{ref_code[:1200]}\n```\n"
        )
    ref_context = "\n".join(ref_summaries)

    diagnosis_prompt = (
        f"Perform Path Relinking and Structural Invariant Analysis for task '{task}'.\n\n"
        f"Elite Solution (Utility: {current_best_utility}):\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n\n"
        f"Population Reference Solutions for Relinking:\n{ref_context}\n\n"
        "Your objectives:\n"
        "1. Identify the 'Consensus Invariable Core': Shared algorithmic structures, core inductive biases, and routines that yield superior performance across top solutions.\n"
        "2. Identify 'Structural Divergences': algorithmic modules where the Elite and Reference solutions diverge (e.g. neighborhood topology, memory management, cooling dynamics, repair strategies).\n"
        "3. Specify 4 distinct 'Path-Relinking & Crossover Ruin-and-Recreate' perturbation plans that graft superior features from reference trajectories into the elite core while tearing down weak blocks.\n\n"
        "Return your analysis strictly in JSON format:\n"
        "```json\n"
        "{\n"
        '  "consensus_core": "<shared high-performing invariants to preserve>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<divergent block to discard from elite>", "source_feature": "<reference feature or logic to incorporate>", "reconstruct_strategy": "<synthesis and recreation operator>"},\n'
        '    {"target_to_ruin": "<divergent block to discard from elite>", "source_feature": "<reference feature or logic to incorporate>", "reconstruct_strategy": "<synthesis and recreation operator>"},\n'
        '    {"target_to_ruin": "<divergent block to discard from elite>", "source_feature": "<reference feature or logic to incorporate>", "reconstruct_strategy": "<synthesis and recreation operator>"},\n'
        '    {"target_to_ruin": "<divergent block to discard from elite>", "source_feature": "<reference feature or logic to incorporate>", "reconstruct_strategy": "<synthesis and recreation operator>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.6)
        diag_data = json.loads(extract_code(diag_response))
        consensus_core = diag_data.get("consensus_core", "Two-phase modular search pipeline")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        consensus_core = "Core heuristic search progression and adaptive selection"
        operators = [
            {
                "target_to_ruin": "Local perturbation operator",
                "source_feature": "Multi-neighborhood ruin-and-recreate with adaptive perturbation radius",
                "reconstruct_strategy": "Path-relinked ruin and repair crossover"
            },
            {
                "target_to_ruin": "Selection and acceptance mechanism",
                "source_feature": "Non-monotonic adaptive threshold acceptance",
                "reconstruct_strategy": "Combine temperature schedule with stagnation-triggered escape jumps"
            },
            {
                "target_to_ruin": "Search memory and history tracking",
                "source_feature": "Frequency-based tabu attributes and elite pool relinking",
                "reconstruct_strategy": "Dynamic memory-guided diversification"
            },
            {
                "target_to_ruin": "Construction and repair heuristics",
                "source_feature": "Greedy randomized adaptive construction with lookahead",
                "reconstruct_strategy": "Structural crossover between elite representation and reference best elements"
            }
        ]

    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic blocks")
        source_feat = op.get("source_feature", "High-utility reference mechanisms")
        rebuild_strat = op.get("reconstruct_strategy", "Path relinking crossover reconstruction")

        reconstruct_prompt = (
            f"You are executing Path Relinking and Ruin-and-Recreate on the following elite algorithm:\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Elite Utility: {current_best_utility} (lower is better)\n\n"
            f"PRESERVE CONSENSUS CORE: {consensus_core}\n"
            f"RUIN TARGET: Completely tear down and replace: {ruin_target}\n"
            f"RECOMBINE REFERENCE MECHANISM: {source_feat}\n"
            f"RECONSTRUCTION DIRECTIVE: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Your output must first provide a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed by the complete, executable, and fully defined Python code implementation. "
            "Ensure the algorithm avoids premature convergence, implements robust edge handling, and optimizes execution efficiency."
        )
        message_batch.append(reconstruct_prompt)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9)
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
