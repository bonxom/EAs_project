"""Prompts for the inner (TSP) and outer (meta) loops."""

from __future__ import annotations

from typing import List, Optional

INNER_TASK = """[TSP_HEURISTIC]
Given a TSP instance, choose the next node while constructing a tour step by step.
The heuristic should consider, where useful:
1) candidate sets of nearby unvisited nodes;
2) one-step look-ahead or cheap simulation of the partial tour;
3) connectivity and clustering (avoid isolating unvisited regions);
4) normalisation of edge costs by the local or global distance scale.
"""

INNER_FUNC_NAME = "select_next_node"
INNER_INPUTS = ["current_node", "destination_node", "unvisited_nodes", "distance_matrix"]
INNER_OUTPUTS = ["next_node"]
INNER_INOUT = (
    "'current_node', 'destination_node', 'next_node' and 'unvisited_nodes' are node IDs. "
    "'distance_matrix' is the full TSP distance matrix. "
    "'unvisited_nodes' is a Numpy array of candidate unvisited node IDs."
)
INNER_OTHER = "Return one integer node ID."

OUTER_TASK = """[META_IMPROVER]
Design a Python function `improve_algorithm` that improves a population of TSP heuristics.

Signature:
    def improve_algorithm(population, utility, language_model, function_format, task):
        ...
        return idea, code, score

Contract:
- population.get_random_solution(task), population.get_solution_by_index(task, i),
  population.get_best_solution(), population.get_subtask_size(task) are available.
  A solution dict has keys 'idea', 'best_sol', 'utility'.
- utility(code, idea, task) scores a generated TSP heuristic; lower is better.
- language_model.prompt(system, user) returns text; prompt_batch(system, prompts) returns a list.
- function_format is the required TSP heuristic format.
- task is the subtask name, e.g. 'tsp-100'.
- extract_code(text) and extract_idea(text) are available.
- Return `idea`, full `code`, and its `score`. Use at most a few LLM calls and do not
  hard-code a specific tour or instance.
"""

OUTER_FUNC_NAME = "improve_algorithm"
OUTER_INPUTS = ["population", "utility", "language_model", "function_format", "task"]
OUTER_OUTPUTS = ["idea", "code", "score"]
OUTER_INOUT = (
    "'population' holds candidate heuristics, 'utility' scores them, "
    "'language_model' is an LLM client, 'function_format' describes the TSP function, "
    "and 'task' is the subtask name."
)
OUTER_OTHER = "Return a tuple (idea, code, score); code must define select_next_node."


class PromptBuilder:
    def __init__(self, *, task: str, func_name: str, inputs: List[str],
                 outputs: List[str], inout_info: str, other_info: str,
                 seed_code: Optional[str] = None):
        self.task = task
        self.func_name = func_name
        self.inputs = inputs
        self.outputs = outputs
        self.inout_info = inout_info
        self.other_info = other_info
        self.seed_code = seed_code
        self.joined_inputs = ", ".join(f"'{name}'" for name in inputs)
        self.joined_outputs = ", ".join(f"'{name}'" for name in outputs)

    def _tail(self, insights, directive, regime) -> str:
        text = ""
        if insights:
            text += "\nConsider these design principles:\n" + "\n".join(
                f"- {tip}" for tip in insights) + "\n"
        if directive:
            text += f"\nFocus: {directive}\n"
        if regime == "explore":
            text += "\nPrefer a substantially different approach.\n"
        elif regime == "exploit":
            text += "\nRefine and optimise the strongest existing patterns.\n"

        text += (
            "\nFirst describe the algorithm in one sentence inside braces {like this}. "
            f"Then implement it as a function named {self.func_name} with "
            f"{len(self.inputs)} input(s): {self.joined_inputs}, returning "
            f"{len(self.outputs)} output(s): {self.joined_outputs}. "
            f"{self.inout_info} {self.other_info}\n"
            "Return exactly one Python code block and no extra explanation.\n"
        )
        return text

    def _format_parents(self, parents: List[dict]) -> str:
        return "".join(
            f"No.{i + 1} description: {p.get('idea', '')}\nCode:\n{p.get('code', '')}\n\n"
            for i, p in enumerate(parents)
        )

    def i1(self, insights=None, directive=None, regime="balanced") -> str:
        prompt = self.task + "\n\nWrite a new algorithm from scratch.\n"
        if self.seed_code:
            prompt += (
                "Seed implementation and required interface:\n"
                f"```python\n{self.seed_code}\n```\n"
            )
        return prompt + self._tail(insights, directive, regime)

    def e1(self, parents: List[dict], insights=None, directive=None,
           regime="balanced") -> str:
        prompt = self.task + f"\n\nExisting algorithms:\n\n{self._format_parents(parents)}"
        prompt += "Create a new algorithm with a substantially different form.\n"
        return prompt + self._tail(insights, directive, regime)

    def e2(self, parents: List[dict], insights=None, directive=None,
           regime="balanced") -> str:
        prompt = self.task + f"\n\nExisting algorithms:\n\n{self._format_parents(parents)}"
        prompt += (
            "Identify their common backbone, then create a different algorithm motivated by it.\n"
        )
        return prompt + self._tail(insights, directive, regime)

    def m1(self, parent: dict, insights=None, directive=None,
           regime="balanced") -> str:
        prompt = (
            self.task + "\n\nCurrent algorithm:\n"
            f"{parent.get('idea', '')}\n{parent.get('code', '')}\n\n"
            "Create a modified version with a different form.\n"
        )
        return prompt + self._tail(insights, directive, regime)

    def m2(self, parent: dict, insights=None, directive=None,
           regime="balanced") -> str:
        prompt = (
            self.task + "\n\nCurrent algorithm:\n"
            f"{parent.get('idea', '')}\n{parent.get('code', '')}\n\n"
            "Identify its main parameters and create a version with different parameter settings.\n"
        )
        return prompt + self._tail(insights, directive, regime)

    def m3(self, parent: dict, insights=None, directive=None,
           regime="balanced") -> str:
        prompt = (
            self.task + "\n\nSimplify the following implementation to reduce overfitting "
            "while keeping the name, inputs and outputs unchanged:\n\n"
            f"{parent.get('code', '')}\n\n{self.inout_info}\n"
        )
        return prompt + self._tail(insights, directive, regime)

    def build(self, operator: str, parents: Optional[List[dict]] = None,
              insights=None, directive=None, regime="balanced") -> str:
        parents = parents or []
        if operator == "i1":
            return self.i1(insights, directive, regime)
        if operator == "e1":
            return self.e1(parents, insights, directive, regime)
        if operator == "e2":
            return self.e2(parents, insights, directive, regime)
        if operator == "m1":
            return self.m1(parents[0], insights, directive, regime)
        if operator == "m2":
            return self.m2(parents[0], insights, directive, regime)
        if operator == "m3":
            return self.m3(parents[0], insights, directive, regime)
        raise ValueError(f"unknown operator: {operator}")


def build_inner_prompts(seed_code: str) -> PromptBuilder:
    return PromptBuilder(
        task=INNER_TASK, func_name=INNER_FUNC_NAME, inputs=INNER_INPUTS,
        outputs=INNER_OUTPUTS, inout_info=INNER_INOUT, other_info=INNER_OTHER,
        seed_code=seed_code,
    )


def build_outer_prompts(seed_improver: str) -> PromptBuilder:
    return PromptBuilder(
        task=OUTER_TASK, func_name=OUTER_FUNC_NAME, inputs=OUTER_INPUTS,
        outputs=OUTER_OUTPUTS, inout_info=OUTER_INOUT, other_info=OUTER_OTHER,
        seed_code=seed_improver,
    )
