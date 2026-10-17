# Idea: A bandit-driven metaheuristic that leverages an Upper Confidence Bound (UCB1) allocation scheme over dynamically generated strategic directions to iteratively exploit top-performing optimization paradigms while exploring novel algorithmic variants.
from utils.utils import extract_code, extract_idea
import json
import math
import numpy as np

# {A bandit-driven metaheuristic that leverages an Upper Confidence Bound (UCB1) allocation scheme over dynamically generated strategic directions to iteratively exploit top-performing optimization paradigms while exploring novel algorithmic variants.}

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in the domain of designing meta optimization strategies and combinatorial optimization problems. "
        "Your task is to design heuristics that can effectively solve optimization problems."
    )

    # Select the current best elite and an exploratory solution from the population
    pop_size = population.get_subtask_size(task)
    elite_solution = population.get_solution_by_index(task, 0)
    rand_solution = population.get_random_solution(task) if pop_size > 1 else elite_solution

    # Synthesize distinct algorithmic search directions (arms) combining elite performance and exploratory diversity
    direction_prompt = (
        f"We have an elite solution with utility {elite_solution.get('utility', 'N/A')}:\n"
        f"```python\n{elite_solution['best_sol']}\n```\n"
        f"and an exploratory reference solution with utility {rand_solution.get('utility', 'N/A')}:\n"
        f"```python\n{rand_solution['best_sol']}\n```\n"
        "Propose 4 radically diverse, novel, and concrete heuristic paradigms or algorithmic directions to improve performance and lower the utility score. "
        "Return your response strictly in the following JSON format:\n"
        "```json\n"
        "{\"insights\": [\"direction 1\", \"direction 2\", \"direction 3\", \"direction 4\"]}\n"
        "```"
    )

    dir_response = language_model.prompt(expertise, direction_prompt, temperature=1.0)
    try:
        directions = json.loads(extract_code(dir_response))["insights"]
    except Exception:
        directions = [
            "Adaptive local search with dynamic tabu tenure and perturbation",
            "Population-based differential evolution tailored for discrete structures",
            "Iterated greedy heuristic with destruction and deep reconstruction phases",
            "Hyper-heuristic selecting low-level perturbation heuristics adaptively"
        ]

    # Initialize Multi-Armed Bandit (MAB) records for each direction (arm)
    num_arms = len(directions)
    arm_counts = np.zeros(num_arms)
    arm_rewards = np.zeros(num_arms)  # Tracking reward: higher is better (reward = -utility)

    eval_cache = []

    # Phase 1: Initialize each arm with 1 evaluation
    init_batch = []
    for d in directions:
        prompt_msg = (
            f"Improve the following baseline solution:\n"
            f"```python\n{elite_solution['best_sol']}\n```\n"
            f"You must return an improved solution formatted strictly as:\n{function_format}\n"
            f"Incorporate the following optimization direction: {d}.\n"
            f"The evaluation assesses quality where LOWER score is BETTER.\n"
            f"Your response must first provide a summary of the key idea inside braces and marked as a comment, followed by the code implementation."
        )
        init_batch.append(prompt_msg)

    init_responses = language_model.prompt_batch(expertise, init_batch, temperature=1.0)
    init_codes = extract_code(init_responses)
    init_ideas = extract_idea(init_responses)

    for i in range(num_arms):
        code_i = init_codes[i]
        idea_i = init_ideas[i]
        u = utility(code_i, idea_i, task)
        eval_cache.append((idea_i, code_i, u))

        arm_counts[i] += 1
        arm_rewards[i] = -float(u)

    # Phase 2: Bandit iterations using UCB1 to allocate queries to the most promising direction
    total_rounds = 4
    for r in range(total_rounds):
        total_pulls = np.sum(arm_counts)
        # Compute UCB scores (standard UCB1 formulation for reward maximization)
        ucb_values = [
            (arm_rewards[i] / arm_counts[i]) + math.sqrt(2.0 * math.log(total_rounds + total_pulls) / arm_counts[i])
            for i in range(num_arms)
        ]
        chosen_arm = int(np.argmax(ucb_values))
        chosen_direction = directions[chosen_arm]

        # Pick the best discovered solution so far as the base for refinement
        current_best = min(eval_cache, key=lambda x: x[2])
        refine_prompt = (
            f"Current best discovered solution has utility {current_best[2]}:\n"
            f"```python\n{current_best[1]}\n```\n"
            f"Refine and aggressively optimize this solution further, advancing the strategic direction: '{chosen_direction}'.\n"
            f"Focus on algorithmic efficiency, fine-tuning meta-parameters, and breaking local optima.\n"
            f"Return an improved solution formatted as:\n{function_format}\n"
            f"Start with the summary of the key idea inside braces and marked as a comment, followed by the full code implementation."
        )

        refine_resp = language_model.prompt(expertise, refine_prompt, temperature=0.8)
        new_code = extract_code(refine_resp)
        new_idea = extract_idea(refine_resp)
        new_u = utility(new_code, new_idea, task)
        eval_cache.append((new_idea, new_code, new_u))

        # Update bandit statistics
        arm_counts[chosen_arm] += 1
        arm_rewards[chosen_arm] += -float(new_u)

    best_idea, best_solution, best_utility = min(eval_cache, key=lambda x: x[2])
    return best_idea, best_solution, best_utility
