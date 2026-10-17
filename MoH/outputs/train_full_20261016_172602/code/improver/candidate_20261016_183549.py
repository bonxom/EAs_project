# Idea: A multi-parent differential ruin-and-recreate metaheuristic that extracts complementary invariant subroutines from elite and diverse reference solutions, performs structured differential diagnosis across parents, and generates novel recombined candidates via batched targeted reconstruction.
#{A multi-parent differential ruin-and-recreate metaheuristic that extracts complementary invariant subroutines from elite and diverse reference solutions, performs structured differential diagnosis across parents, and generates novel recombined candidates via batched targeted reconstruction.}

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

    # Select complementary reference solutions: secondary elite and an explorative random solution
    if pop_size > 1:
        runner_up_idx = min(1, pop_size - 1)
        ref_sol = population.get_solution_by_index(task, runner_up_idx)
    else:
        ref_sol = elite_sol

    random_sol = population.get_random_solution(task) if pop_size > 2 else ref_sol

    # Phase 1: Differential Ruin-and-Recreate Diagnosis across multiple parents
    diagnosis_prompt = (
        f"Perform a Differential Ruin-and-Recreate Diagnosis for optimization task '{task}'.\n\n"
        f"--- Primary Elite Solution (Utility: {current_best_utility}) ---\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n\n"
        f"--- Reference Solution 1 (Utility: {ref_sol.get('utility')}) ---\n"
        f"Code:\n```python\n{ref_sol.get('best_sol')}\n```\n\n"
        f"--- Reference Solution 2 (Utility: {random_sol.get('utility')}) ---\n"
        f"Code:\n```python\n{random_sol.get('best_sol')}\n```\n\n"
        "Instructions:\n"
        "1. Identify the 'Invariable Core': The most potent shared or complementary algorithmic backbone across these solutions.\n"
        "2. Formulate 4 distinct Differential Ruin-and-Recreate operators that tear down stagnant sub-mechanisms in the elite solution "
        "and reconstruct them by cross-breeding or differentially transplanting successful features (e.g., adaptive parameters, hybrid neighborhood shifts, "
        "diversity preservation, and robust exploration) observed across the references.\n\n"
        "Return your analysis strictly in JSON format:\n"
        "```json\n"
        "{\n"
        '  "invariable_core": "<synthesis of robust core algorithmic backbone to preserve>",\n'
        '  "reconstruction_operators": [\n'
        '    {"target_to_ruin": "<specific bottleneck component in elite>", "recombine_source": "<component/idea from references to integrate>", "reconstruct_strategy": "<detailed differential recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific bottleneck component in elite>", "recombine_source": "<component/idea from references to integrate>", "reconstruct_strategy": "<detailed differential recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific bottleneck component in elite>", "recombine_source": "<component/idea from references to integrate>", "reconstruct_strategy": "<detailed differential recreation mechanism>"},\n'
        '    {"target_to_ruin": "<specific bottleneck component in elite>", "recombine_source": "<component/idea from references to integrate>", "reconstruct_strategy": "<detailed differential recreation mechanism>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.7)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core", "Dual-stage core heuristic search pipeline")
        operators = diag_data.get("reconstruction_operators", [])
    except Exception:
        invariable_core = "Core heuristic search progression with balanced exploration-exploitation"
        operators = [
            {"target_to_ruin": "Single-trajectory neighborhood search", "recombine_source": "Reference solution exploratory diversification", "reconstruct_strategy": "Adaptive differential cross-neighborhood perturbation with multi-parent crossover"},
            {"target_to_ruin": "Static parameter cooling", "recombine_source": "Reference solution adaptive mechanisms", "reconstruct_strategy": "Dynamic self-adaptive parameter regulation based on fitness momentum"},
            {"target_to_ruin": "Greedy deterministic acceptance", "recombine_source": "Stochastic reference transition operators", "reconstruct_strategy": "Cross-entropy guided probabilistic acceptance filtering"},
            {"target_to_ruin": "Local stagnation recovery", "recombine_source": "Diverse structural mutations from reference", "reconstruct_strategy": "Multi-parent differential ruin and reconstruct with structured memory"}
        ]

    # Phase 2: Batched Recombination and Reconstruct Prompting
    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Stagnant heuristic components")
        recombine_source = op.get("recombine_source", "Reference solution features")
        rebuild_strat = op.get("reconstruct_strategy", "Differential structural synthesis")

        reconstruct_prompt = (
            f"You are executing Differential Ruin-and-Recreate on optimization task '{task}'.\n\n"
            f"PRIMARY PARENT (Elite):\n```python\n{current_best_code}\n```\n\n"
            f"SECONDARY PARENT (Reference):\n```python\n{ref_sol.get('best_sol')}\n```\n\n"
            f"PRESERVE INVARIABLE CORE: {invariable_core}\n"
            f"TARGET TO RUIN: {ruin_target}\n"
            f"DONOR COMPONENT: {recombine_source}\n"
            f"DIFFERENTIAL RECONSTRUCTION STRATEGY: {rebuild_strat}\n\n"
            f"Format requirements:\n{function_format}\n"
            "Ensure the output begins with a concise one-sentence summary of the new idea enclosed in braces and marked as a comment, "
            "followed immediately by the complete, executable, and self-contained Python function implementation. "
            "Prevent infinite execution loops, ensure valid edge case handling, and optimize algorithmic computational efficiency."
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
