# Idea: This metaheuristic applies genetic crossover by synthesizing complementary structural mechanisms from two high-performing parent heuristics, followed by directed mutation across multiple temperature regimes to discover superior combinatorial algorithms.
from utils.utils import extract_code, extract_idea
import json
import random

# {This metaheuristic applies genetic crossover by synthesizing complementary structural mechanisms from two high-performing parent heuristics, followed by directed mutation across multiple temperature regimes to discover superior combinatorial algorithms.}

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in the domain of designing meta optimization strategy and "
        "combinatorial optimization problems. Your task is to design heuristics that "
        "can effectively solve optimization problems."
    )

    pop_size = population.get_subtask_size(task)

    # Select parent 1 from the elite (rank 0)
    parent_a = population.get_solution_by_index(task, 0)

    # Select parent 2: distinct elite or random high-performing candidate to ensure diversity
    if pop_size > 1:
        parent_b_idx = random.randint(1, min(pop_size - 1, 3))
        parent_b = population.get_solution_by_index(task, parent_b_idx)
    else:
        parent_b = population.get_random_solution(task)

    parent_a_idea = parent_a.get('idea', 'Elite solution idea')
    parent_b_idea = parent_b.get('idea', 'Diverse candidate idea')

    # Crossover prompt: elicit structural synthesis and novel recombination directions
    crossover_analysis_prompt = (
        f"Parent Algorithm 1 (Utility: {parent_a['utility']}):\n"
        f"Idea: {parent_a_idea}\n"
        f"Code:\n```python\n{parent_a['best_sol']}\n```\n\n"
        f"Parent Algorithm 2 (Utility: {parent_b['utility']}):\n"
        f"Idea: {parent_b_idea}\n"
        f"Code:\n```python\n{parent_b['best_sol']}\n```\n\n"
        "Perform a genetic crossover by identifying the core structural strengths and orthogonal search mechanics of both parent algorithms. "
        "Propose 3 distinct hybrid offspring concepts that synthesize complementary logic from both parents (e.g., combining exploration operators, "
        "acceptance criteria, or adaptive mechanisms) to achieve lower utility scores.\n"
        "Your response must be formatted as a valid JSON object:\n"
        "```json\n{\n  \"insights\": [\"Hybrid Concept 1\", \"Hybrid Concept 2\", \"Hybrid Concept 3\"]\n}\n```"
    )

    crossover_response = language_model.prompt(expertise, crossover_analysis_prompt, temperature=0.7)
    try:
        crossover_directions = json.loads(extract_code(crossover_response))["insights"]
    except Exception:
        crossover_directions = [
            "Hybridize Parent 1 exploitation with Parent 2 exploration mechanisms.",
            "Integrate dynamic parameter adaptation from Parent 1 into Parent 2 structure.",
            "Combine the neighborhood search of Parent 2 with the selection strategy of Parent 1."
        ]

    # Batch prompt generation incorporating crossover directions and parents' contexts
    message_batch = []
    for direction in crossover_directions:
        message = (
            f"You are tasked with executing a genetic crossover between two heuristics to create an improved offspring algorithm.\n"
            f"Parent 1 Code:\n```python\n{parent_a['best_sol']}\n```\n"
            f"Parent 2 Code:\n```python\n{parent_b['best_sol']}\n```\n\n"
            f"Crossover Guidance:\n{direction}\n\n"
            f"Synthesize the structural strengths of both parents into a unified, coherent, and highly effective heuristic. "
            f"Avoid simply concatenating code; cleanly integrate their logic.\n"
            f"You must return the improved solution formatted strictly as:\n{function_format}\n"
            f"Remember, lower utility indicates a superior solution.\n"
            f"Your response must first provide a one-sentence summary of the key hybrid idea inside braces marked as a comment: "
            f"# {{Your hybrid idea here}}, followed immediately by the executable Python code."
        )
        message_batch.append(message)

    # Prompt LLM batch to generate offspring candidate implementations
    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.7)
    new_solutions = extract_code(responses)
    new_ideas = extract_idea(responses)

    evaluated_candidates = []
    for idea, sol in zip(new_ideas, new_solutions):
        if sol and len(sol.strip()) > 0:
            try:
                score = utility(sol, idea, task)
                evaluated_candidates.append((idea, sol, score))
            except Exception:
                continue

    # Fallback to parent_a if all offspring evaluations fail
    if not evaluated_candidates:
        return parent_a_idea, parent_a['best_sol'], parent_a['utility']

    best_idea, best_solution, best_utility = min(evaluated_candidates, key=lambda x: x[2])
    return best_idea, best_solution, best_utility
