# Idea: A diagnostic Ruin-and-Recreate meta-optimizer that couples multi-operator parallel perturbation with an active reflexive self-healing loop to salvage promising candidates encountering syntax or runtime exceptions.
#{A diagnostic Ruin-and-Recreate meta-optimizer that couples multi-operator parallel perturbation with an active reflexive self-healing loop to salvage promising candidates encountering syntax or runtime exceptions.}

from utils.utils import extract_code, extract_idea
import json
import traceback

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    # 1. Retrieve Current Elite and Reference Solution
    elite_sol = population.get_solution_by_index(task, 0)
    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    # 2. Diagnostic Ruin-and-Recreate Meta-Decomposition Phase
    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate (Iterated Local Search) philosophy:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components to ruin (e.g., representation, neighborhood exploration, selection pressure, or stagnation handling) and rebuild them with superior algorithmic logic.\n"
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

    # 3. Parallel Batch-Prompted Ruin-and-Recreate Reconstruction
    message_batch = []
    for op in operators[:4]:
        ruin_target = op.get("target_to_ruin", "Inefficient heuristic blocks")
        rebuild_strat = op.get("reconstruct_strategy", "Advanced search intensification")
        reconstruct_prompt = (
            f"You are executing an Iterated Local Search with Ruin-and-Recreate on the following solution:\n"
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

    # 4. Evaluation with Active Compiler-in-the-Loop Self-Healing
    candidates = []
    failed_candidates = []

    for idx, (idea, code) in enumerate(zip(candidate_ideas, candidate_codes)):
        if not code or not code.strip():
            continue
        try:
            score = utility(code, idea, task)
            candidates.append((idea, code, score))
        except Exception as e:
            err_info = str(e).strip()
            if not err_info:
                err_info = traceback.format_exc().strip()
            failed_candidates.append({
                "idea": idea,
                "code": code,
                "error": err_info[:500]
            })

    # Reflexive repair pass for structurally high-potential failed candidates
    if failed_candidates:
        repair_batch = []
        for fail in failed_candidates:
            repair_prompt = (
                f"A candidate optimization heuristic failed during execution with the following error:\n"
                f"Error: {fail['error']}\n\n"
                f"Faulty Code:\n```python\n{fail['code']}\n```\n\n"
                f"Original Idea: {fail['idea']}\n\n"
                f"Please fix the syntax, type, indexing, or runtime bug in the code. "
                f"Preserve the algorithmic intent and idea while ensuring strict syntactic and operational correctness.\n\n"
                f"Format requirements:\n{function_format}\n"
                "Your output must first provide a concise one-sentence summary of the repaired idea enclosed in braces and marked as a comment, "
                "followed by the complete, executable, and fully defined Python code implementation."
            )
            repair_batch.append(repair_prompt)

        try:
            repaired_responses = language_model.prompt_batch(expertise, repair_batch, temperature=0.3)
            repaired_codes = extract_code(repaired_responses)
            repaired_ideas = extract_idea(repaired_responses)

            for rep_idea, rep_code in zip(repaired_ideas, repaired_codes):
                if not rep_code or not rep_code.strip():
                    continue
                try:
                    rep_score = utility(rep_code, rep_idea, task)
                    candidates.append((rep_idea, rep_code, rep_score))
                except Exception:
                    continue
        except Exception:
            pass

    # 5. Elitist Selection
    if candidates:
        iter_best_idea, iter_best_code, iter_best_utility = min(candidates, key=lambda x: x[2])
        if iter_best_utility < current_best_utility:
            current_best_idea = iter_best_idea
            current_best_code = iter_best_code
            current_best_utility = iter_best_utility

    return current_best_idea, current_best_code, current_best_utility
