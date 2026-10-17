# Idea: An adaptive Ruin-and-Recreate metaheuristic that extracts structural invariant search logic via diagnostic decomposition and replaces static fallbacks with task-contextual dynamic telemetry memory to sample proven perturbation targets.
# {An adaptive Ruin-and-Recreate metaheuristic that extracts structural invariant search logic via diagnostic decomposition and replaces static fallbacks with task-contextual dynamic telemetry memory to sample proven perturbation targets.}

from utils.utils import extract_code, extract_idea
import json
import numpy as np

# Global dynamic telemetry memory to track perturbation target efficacy across invocations and tasks
_OPERATOR_TELEMETRY = {}

def improve_algorithm(population, utility, language_model, function_format, task):
    global _OPERATOR_TELEMETRY

    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    elite_sol = population.get_solution_by_index(task, 0)
    pop_size = population.get_subtask_size(task)
    reference_sol = population.get_solution_by_index(task, min(1, pop_size - 1)) if pop_size > 1 else elite_sol

    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # Initialize task telemetry pool if not already present
    if task not in _OPERATOR_TELEMETRY:
        _OPERATOR_TELEMETRY[task] = [
            {"target_to_ruin": "Neighborhood perturbation and move acceptance logic",
             "reconstruct_strategy": "Adaptive multi-neighborhood ruin-and-rebuild operator with dynamic shake intensity",
             "weight": 1.0, "successes": 1},
            {"target_to_ruin": "Cooling schedule and stagnation evasion mechanism",
             "reconstruct_strategy": "Non-monotonic adaptive temperature cycling with history-informed memory restarts",
             "weight": 1.0, "successes": 1},
            {"target_to_ruin": "Initial candidate construction and greedy heuristic seeding",
             "reconstruct_strategy": "Guided regret-based randomized adaptive greedy construction with elite feature bias",
             "weight": 1.0, "successes": 1},
            {"target_to_ruin": "Local exploitation vs global diversification balance",
             "reconstruct_strategy": "Dual-phase Pareto-guided decomposition search with path relinking between distant elites",
             "weight": 1.0, "successes": 1},
            {"target_to_ruin": "Stagnation detection and population update rule",
             "reconstruct_strategy": "Information-entropy-guided structural mutation with feasibility repair operators",
             "weight": 1.0, "successes": 1}
        ]

    # Diagnostic Phase: Ruin-and-Recreate structural decomposition
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

    invariable_core = None
    operators = []

    try:
        diag_response = language_model.prompt(expertise, diagnosis_prompt, temperature=0.7)
        diag_data = json.loads(extract_code(diag_response))
        invariable_core = diag_data.get("invariable_core")
        parsed_ops = diag_data.get("reconstruction_operators", [])
        if isinstance(parsed_ops, list) and len(parsed_ops) >= 2:
            operators = parsed_ops
    except Exception:
        pass

    # Dynamic Telemetry Memory Fallback / Augmentation
    if not invariable_core or len(operators) < 2:
        invariable_core = (
            f"Core iterative optimization architecture and empirical solution validation logic for {task}"
        )
        task_pool = _OPERATOR_TELEMETRY[task]
        weights = np.array([max(0.1, op.get("weight", 1.0)) for op in task_pool])
        probabilities = weights / np.sum(weights)
        selected_indices = np.random.choice(len(task_pool), size=min(4, len(task_pool)), replace=False, p=probabilities)
        operators = [task_pool[idx] for idx in selected_indices]

    # Generation Phase: Parallel LLM generation targeting specific ruined sub-components
    message_batch = []
    active_operators = operators[:4]
    for op in active_operators:
        ruin_target = op.get("target_to_ruin", "Local search operator")
        rebuild_strat = op.get("reconstruct_strategy", "Adaptive neighborhood search")
        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution for task '{task}':\n"
            f"```python\n{current_best_code}\n```\n"
            f"Current Utility: {current_best_utility} (lower is better)\n\n"
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

    candidates = []
    for idx, (idea, code) in enumerate(zip(candidate_ideas, candidate_codes)):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            candidates.append((idea, code, score, idx))
        except Exception:
            continue

    # Evaluation & Telemetry Feedback Update
    if candidates:
        iter_best_idea, iter_best_code, iter_best_utility, best_idx = min(candidates, key=lambda x: x[2])
        if iter_best_utility < current_best_utility:
            # Reward successful perturbation target in telemetry memory
            winning_op = active_operators[best_idx]
            winning_target = winning_op.get("target_to_ruin")
            for stored_op in _OPERATOR_TELEMETRY[task]:
                if stored_op.get("target_to_ruin") == winning_target:
                    stored_op["weight"] = stored_op.get("weight", 1.0) * 1.5
                    stored_op["successes"] = stored_op.get("successes", 1) + 1
                    break
            else:
                _OPERATOR_TELEMETRY[task].append({
                    "target_to_ruin": winning_target,
                    "reconstruct_strategy": winning_op.get("reconstruct_strategy", ""),
                    "weight": 2.0,
                    "successes": 2
                })

            current_best_idea = iter_best_idea
            current_best_code = iter_best_code
            current_best_utility = iter_best_utility

    return current_best_idea, current_best_code, current_best_utility
