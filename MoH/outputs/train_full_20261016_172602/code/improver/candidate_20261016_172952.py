# Idea: Multi-Armed Bandit (UCB) selection over stratified population tiers combined with dual-mode exploration-exploitation prompting to synthesize novel heuristic algorithms.
# {Multi-Armed Bandit (UCB) selection over stratified population tiers combined with dual-mode exploration-exploitation prompting to synthesize novel heuristic algorithms.}

from utils.utils import extract_code, extract_idea
import json
import math
import numpy as np

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in the domain of designing meta optimization strategy and combinatorial optimization problems. "
        "Your task is to design heuristics that can effectively solve optimization problems."
    )

    pop_size = population.get_subtask_size(task)

    # Stratify population into 3 arms: Elite (top rank), Mid-tier (median), and Exploratory (random/tail)
    arms = {}
    arms['elite'] = population.get_solution_by_index(task, 0)
    mid_idx = max(0, pop_size // 2)
    arms['mid'] = population.get_solution_by_index(task, mid_idx)
    arms['exploratory'] = population.get_random_solution(task)

    # Multi-Armed Bandit (UCB1) selection balancing utility (exploitation) and diversity/rank uncertainty (exploration)
    arm_names = ['elite', 'mid', 'exploratory']
    # Synthetic visit counts and inverse-utility rewards (minimization -> lower utility is higher reward)
    visit_counts = {'elite': 3, 'mid': 2, 'exploratory': 1}
    total_visits = sum(visit_counts.values())

    ucb_scores = {}
    for arm in arm_names:
        u = float(arms[arm]['utility'])
        # Normalize reward inversely proportional to utility
        reward = 1.0 / (1.0 + max(0.0, u))
        exploration_bonus = math.sqrt((2.0 * math.log(total_visits)) / visit_counts[arm])
        ucb_scores[arm] = reward + 0.6 * exploration_bonus

    selected_arm_name = max(ucb_scores, key=ucb_scores.get)
    selected_solution = arms[selected_arm_name]
    elite_solution = arms['elite']

    # Divergent & Recombinative direction brainstorming
    direction_prompt = (
        f"We have an active solution arm '{selected_arm_name}' with utility: {selected_solution.get('utility', 'N/A')}:\n"
        f"Idea: {selected_solution.get('idea', 'N/A')}\n"
        f"Code: {selected_solution['best_sol']}\n\n"
        f"Reference elite solution utility: {elite_solution.get('utility', 'N/A')}:\n"
        f"Elite Idea: {elite_solution.get('idea', 'N/A')}\n\n"
        "Propose 4 complementary optimization directions:\n"
        "1. Deep exploitation: Refine the active solution's algorithmic bottlenecks, pruning, and parameterization.\n"
        "2. Cross-over recombination: Synthesize core structural strengths of both active and elite heuristics.\n"
        "3. Radical exploration: A fundamentally different metaheuristic or paradigm shift (e.g., adaptive hyper-heuristics, tabu/ALNS hybrids, guided local search).\n"
        "4. Mathematical/problem-structure exploitation: Leverage domain invariants, decomposition, or surrogate estimations.\n\n"
        "Respond strictly with a valid JSON in this format:\n"
        "```json\n"
        "{\n"
        '  "insights": ["direction 1", "direction 2", "direction 3", "direction 4"]\n'
        "}\n"
        "```"
    )

    response = language_model.prompt(expertise, direction_prompt, temperature=0.7)
    try:
        directions = json.loads(extract_code(response))["insights"]
    except Exception:
        directions = [
            "Refine algorithmic structures with adaptive parameter tuning.",
            "Combine diversification from exploratory search with elite intensification.",
            "Implement an adaptive large neighborhood search framework.",
            "Exploit invariant subproblem decompositions."
        ]

    message_batch = []
    for direction in directions:
        message = (
            f"Improve the following heuristic solution:\n"
            f"```python\n{selected_solution['best_sol']}\n```\n"
            f"Target Improvement Direction: {direction}\n\n"
            f"You must return an improved solution strictly formatted as follows:\n{function_format}\n"
            "Guidelines:\n"
            "- First describe your metaheuristic optimization strategy in one concise sentence inside a brace and marked as a comment: #{description}#\n"
            "- Implement the complete Python function without placeholders.\n"
            "- The goal is to minimize the utility evaluation score. Be mathematically sound, creative, and robust."
        )
        message_batch.append(message)

    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.7)
    new_solutions = extract_code(responses)
    new_ideas = extract_idea(responses)

    solutions_with_utilities = []
    for idea, solution in zip(new_ideas, new_solutions):
        if not solution:
            continue
        try:
            score = utility(solution, idea, task)
            solutions_with_utilities.append((idea, solution, score))
        except Exception:
            continue

    if not solutions_with_utilities:
        return selected_solution.get('idea', 'fallback'), selected_solution['best_sol'], selected_solution['utility']

    best_idea, best_solution, best_utility = min(solutions_with_utilities, key=lambda x: x[2])
    return best_idea, best_solution, best_utility
