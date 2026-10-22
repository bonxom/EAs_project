# {Seed meta-improver: samples a parent heuristic, asks the LLM for a structurally improved TSP heuristic, and scores the result with the supplied utility.}
def improve_algorithm(population, utility, language_model, function_format, task):
    parent = population.get_random_solution(task)

    system = (
        "You are an expert in combinatorial optimization and in designing TSP node-selection heuristics. "
        "Return one Python function with the required signature."
    )
    prompt = (
        "[TSP_HEURISTIC] Improve the following TSP node-selection heuristic.\n\n"
        "Current algorithm description:\n"
        f"{parent.get('idea', '')}\n\n"
        "Current code:\n"
        f"```python\n{parent.get('best_sol', '')}\n```\n\n"
        "Create a new heuristic that is meaningfully different from the current one. "
        "Consider candidate sets, one-step look-ahead, clusters, MST/connectivity, or a different "
        "score normalisation. The function must keep the same name, inputs and outputs.\n\n"
        f"{function_format}\n\n"
        "First put a one-sentence description inside a brace, then provide exactly one Python code block."
    )
    response = language_model.prompt(system, prompt, temperature=0.9)
    code = extract_code(response)
    idea = extract_idea(response) or "improved TSP heuristic"

    if not code:
        return parent.get("idea", "seed"), parent["best_sol"], parent["utility"]

    score = utility(code, idea, task)
    return idea, code, score
