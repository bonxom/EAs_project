# Idea: An adaptive Ruin-and-Recreate metaheuristic combining LLM-driven diagnosis and targeted batch reconstruction with an AST-based structural diversity filter to prune redundant or degenerate code candidates before costly utility evaluation.
#{An adaptive Ruin-and-Recreate metaheuristic combining LLM-driven diagnosis and targeted batch reconstruction with an AST-based structural diversity filter to prune redundant or degenerate code candidates before costly utility evaluation.}#

from utils.utils import extract_code, extract_idea
import json
import ast
from collections import Counter

def _compute_ast_features(code_str):
    """Extract structural AST n-grams and syntactic complexity metrics."""
    try:
        tree = ast.parse(code_str)
    except Exception:
        return None
    node_types = [type(node).__name__ for node in ast.walk(tree)]
    if not node_types or len(node_types) < 10:
        return None
    type_counts = Counter(node_types)
    return type_counts

def _ast_similarity(feat1, feat2):
    """Compute cosine similarity between two AST node frequency profiles."""
    all_keys = set(feat1.keys()).union(set(feat2.keys()))
    dot = sum(feat1.get(k, 0) * feat2.get(k, 0) for k in all_keys)
    norm1 = sum(v ** 2 for v in feat1.values()) ** 0.5
    norm2 = sum(v ** 2 for v in feat2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return dot / (norm1 * norm2)

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in metaheuristics and combinatorial optimization. "
        "Your goal is to design highly competitive, novel, and robust optimization algorithms."
    )

    elite_sol = population.get_solution_by_index(task, 0)
    current_best_code = elite_sol['best_sol']
    current_best_idea = elite_sol.get('idea', 'Elite baseline heuristic')
    current_best_utility = elite_sol['utility']

    elite_ast = _compute_ast_features(current_best_code)

    diagnosis_prompt = (
        f"Analyze the current best algorithm for the task '{task}':\n"
        f"Idea: {current_best_idea}\n"
        f"Code:\n```python\n{current_best_code}\n```\n"
        f"Utility Score (lower is better): {current_best_utility}\n\n"
        "Apply an Adaptive Ruin-and-Recreate (Iterated Local Search) philosophy:\n"
        "1. Identify the 'Invariable Core': The critical high-utility algorithmic backbone that MUST be preserved.\n"
        "2. Formulate 4 distinct, orthogonal 'Ruin-and-Recreate' perturbation strategies targeting distinct sub-components to ruin "
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

    # Filter candidates by AST structural novelty and validity
    filtered_candidates = []
    accepted_ast_features = [elite_ast] if elite_ast is not None else []

    for idea, code in zip(candidate_ideas, candidate_codes):
        if not code or not code.strip():
            continue

        cand_ast = _compute_ast_features(code)
        if cand_ast is None:
            continue

        # Check AST structural novelty against the elite and already accepted candidates
        is_novel = True
        for prev_ast in accepted_ast_features:
            if _ast_similarity(cand_ast, prev_ast) > 0.985:  # Degenerate or near-duplicate AST
                is_novel = False
                break

        if is_novel:
            accepted_ast_features.append(cand_ast)
            filtered_candidates.append((idea, code))

    # Fallback to unfiltered parsed candidates if AST filtering was overly strict
    if not filtered_candidates:
        filtered_candidates = [(idea, code) for idea, code in zip(candidate_ideas, candidate_codes) if code and code.strip()]

    evaluated_candidates = []
    for idea, code in filtered_candidates:
        try:
            score = utility(code, idea, task)
            evaluated_candidates.append((idea, code, score))
        except Exception:
            continue

    if evaluated_candidates:
        iter_best_idea, iter_best_code, iter_best_utility = min(evaluated_candidates, key=lambda x: x[2])
        if iter_best_utility < current_best_utility:
            current_best_idea = iter_best_idea
            current_best_code = iter_best_code
            current_best_utility = iter_best_utility

    return current_best_idea, current_best_code, current_best_utility
