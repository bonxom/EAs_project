"""One HiFo-style evolutionary layer, used by both the outer and inner loops."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

from .insight import INNER_TIPS, OUTER_TIPS, InsightPool, Navigator
from .llm import LoggedLLM
from .population import Population
from .prompts import PromptBuilder
from .run_logger import RunLogger
from .utils import extract_code, extract_idea

EvaluateFn = Callable[..., Dict[str, Any]]


class HiFoLoop:
    def __init__(self, *, name: str, cfg: dict, prompts: PromptBuilder,
                 evaluate: EvaluateFn, llm: LoggedLLM, logger: RunLogger,
                 seed_items: Optional[List[dict]] = None,
                 extra_offspring: Optional[Callable[[Population, int], List[dict]]] = None):
        self.name = name
        self.prompts = prompts
        self.evaluate = evaluate
        self.llm = llm
        self.logger = logger
        self.n_pop = int(cfg["n_pop"])
        self.m = int(cfg["m"])
        self.operators = list(cfg["operators"])
        self.operator_weights = list(cfg["operator_weights"])
        self.seed_items = seed_items or []
        self.extra_offspring = extra_offspring

        self.pop = Population(max_size=int(cfg["pop_size"]))
        self.insights = InsightPool(initial_tips=OUTER_TIPS if name == "outer" else INNER_TIPS)
        self.navigator = Navigator()
        self.best_history: List[float] = []
        self.avg_history: List[float] = []
        self.diversity_history: List[float] = []

    def _seed_population(self) -> None:
        for item in self.seed_items:
            result = self.evaluate(item["code"], source=f"{self.name}_seed",
                                   idea=item.get("idea", ""))
            if result["valid"]:
                self.pop.add(item.get("idea", ""), item["code"],
                             result["objective"], result)

    def _select_parents(self, n: int) -> List[dict]:
        items = self.pop.items()
        if not items:
            return []
        weights = [1.0 / (rank + 1 + len(items) / 2.0) for rank in range(len(items))]
        return random.choices(items, weights=weights, k=min(n, len(items)))

    def _effectiveness(self, objective: float) -> float:
        objectives = [item["objective"] for item in self.pop.items()]
        if not objectives:
            return 0.0
        best, worst = min(objectives), max(objectives)
        if worst == best:
            return 0.1
        normalized = (worst - objective) / (worst - best)
        if objective <= best:
            return max(-1.0, min(1.0, 0.8 + 0.2 * normalized))
        if objective <= sum(objectives) / len(objectives):
            return max(-1.0, min(1.0, 0.2 + 0.6 * normalized))
        return max(-1.0, min(1.0, -0.3 + 0.5 * normalized))

    def _generate_one(self, operator: str, parents: Optional[List[dict]],
                      *, source: str, regime: str, directive: str,
                      insights: List[str]) -> Optional[dict]:
        prompt = self.prompts.build(operator, parents, insights, directive, regime)
        try:
            response = self.llm.prompt(
                "You are an expert in combinatorial optimization and heuristic design.",
                prompt, tag=f"{self.name}_{operator}",
            )
        except Exception as exc:
            self.logger.info(f"[{self.name}/{operator}] LLM failed: {exc}")
            return None

        code = extract_code(response)
        if not code:
            self.logger.info(f"[{self.name}/{operator}] no code block")
            return None
        idea = extract_idea(response) or f"{operator} candidate"
        result = self.evaluate(code, source=source, idea=idea)
        if not result["valid"]:
            self.logger.info(f"[{self.name}/{operator}] invalid code: {result['error']}")
            return None

        reward = self._effectiveness(result["objective"])
        for tip in insights:
            self.insights.update(tip, reward)
        return {"idea": idea, "code": code, "objective": result["objective"],
                "result": result}

    def _update_metrics(self) -> None:
        objectives = [item["objective"] for item in self.pop.items()]
        if objectives:
            self.best_history.append(min(objectives))
            self.avg_history.append(sum(objectives) / len(objectives))
        items = self.pop.items()
        if len(items) >= 2:
            total = len(items) * (len(items) - 1) / 2
            diverse = sum(1 for i in range(len(items))
                          for j in range(i + 1, len(items))
                          if items[i]["code"] != items[j]["code"])
            self.diversity_history.append(diverse / total)

    def _extract_insights(self) -> None:
        items = sorted(self.pop.items(), key=lambda item: item["objective"])
        if not items:
            return
        subject = "meta-improvers" if self.name == "outer" else "TSP heuristics"
        prompt = (
            f"The following are high-performing {subject}:\n"
            + "\n".join(f"{i + 1}. {item.get('idea', '')}\n{item['code'][:800]}"
                        for i, item in enumerate(items[:max(1, len(items) // 2)]))
            + "\nReturn 1-2 general design principles, one per line, each starting with '- '."
        )
        try:
            response = self.llm.prompt(
                "You are an expert in metaheuristic design.", prompt,
                tag=f"{self.name}_insights",
            )
        except Exception:
            return
        for line in response.splitlines():
            if line.strip().startswith("-"):
                self.insights.add(line.strip()[1:].strip())

    def run(self) -> Dict[str, Any]:
        self._seed_population()
        self.logger.log_generation(loop=self.name, generation=0,
                                   population=self.pop.items())

        for generation in range(1, self.n_pop + 1):
            regime, directive = self.navigator.get_guidance(
                self.best_history, self.diversity_history)
            insights = self.insights.get(k=3)

            for operator, weight in zip(self.operators, self.operator_weights):
                if random.random() > weight / max(self.operator_weights):
                    continue
                n_parents = 1 if operator.startswith("m") else self.m
                parents = self._select_parents(n_parents)
                if not parents:
                    continue
                offspring = self._generate_one(
                    operator, parents, source=f"{self.name}_gen{generation}_{operator}",
                    regime=regime, directive=directive, insights=insights,
                )
                if offspring:
                    self.pop.add(offspring["idea"], offspring["code"],
                                 offspring["objective"], offspring["result"])

            if self.extra_offspring:
                for item in self.extra_offspring(self.pop, generation):
                    self.pop.add(item["idea"], item["code"], item["objective"],
                                 item.get("result"))

            self._update_metrics()
            if random.random() < 0.8:
                self._extract_insights()

            self.logger.log_generation(loop=self.name, generation=generation,
                                       population=self.pop.items())
            self.logger.info(
                f"[{self.name}] gen {generation}: best={self.pop.get_best_objective():.4f}"
            )

        return {
            "best": self.pop.get_best_solution(),
            "population": self.pop.items(),
            "insights": self.insights.summary(),
        }
