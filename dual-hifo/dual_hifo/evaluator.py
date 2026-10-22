"""Inner-loop TSP evaluation and execution of outer-loop improvers."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .population import Population
from .run_logger import RunLogger
from .tsp import FUNCTION_FORMAT, TSPInstance, evaluate_heuristic
from .utils import exec_code, extract_code, extract_idea, time_limit


class TSPEvaluator:
    def __init__(self, *, instances: List[TSPInstance], size: int, logger: RunLogger,
                 neighbor_size: Optional[int] = None, timeout: int = 45):
        self.instances = instances
        self.size = size
        self.logger = logger
        self.neighbor_size = neighbor_size
        self.timeout = timeout

    def evaluate(self, code: Optional[str], *, source: str, idea: str = "") -> Dict[str, Any]:
        result = evaluate_heuristic(
            code or "", self.instances,
            neighbor_size=self.neighbor_size,
            timeout=self.timeout,
        )
        self.logger.log_tsp(
            source=source,
            objective=result["objective"],
            valid=result["valid"],
            mean_cost=result["mean_cost"],
            mean_optimal_cost=result["mean_optimal_cost"],
            per_instance=result["per_instance_cost"],
            code=code or "",
            idea=idea,
            error=result.get("error"),
        )
        return result

    def utility(self, code: str, idea: str = "", task: Optional[str] = None) -> float:
        return self.evaluate(code, source="utility", idea=idea)["objective"]


def apply_improver(*, improver_code: str, population: Population, evaluator: TSPEvaluator,
                   language_model, logger: RunLogger, size: int, source: str,
                   timeout: int = 240) -> Dict[str, Any]:
    failure = {
        "ok": False,
        "idea": "",
        "code": "",
        "reported_score": 1e6,
        "result": {"objective": 1e6, "valid": False},
        "error": None,
    }
    if not improver_code or "def improve_algorithm" not in improver_code:
        failure["error"] = "improve_algorithm is missing"
        return failure

    try:
        with time_limit(timeout):
            env = exec_code(improver_code)
            improve = env.get("improve_algorithm")
            if not callable(improve):
                failure["error"] = "improve_algorithm is not callable"
                return failure
            output = improve(copy.deepcopy(population), evaluator.utility,
                             language_model, FUNCTION_FORMAT, f"tsp-{size}")
    except Exception as exc:
        failure["error"] = f"{type(exc).__name__}: {exc}"
        logger.info(f"[{source}] improver failed: {failure['error']}")
        return failure

    if not isinstance(output, (tuple, list)) or len(output) < 3:
        failure["error"] = "improver must return (idea, code, score)"
        return failure

    idea, code, score = output[0], output[1], output[2]
    if isinstance(code, list):
        code = max(code, key=len) if code else ""
    if not isinstance(code, str):
        failure["error"] = f"returned code has type {type(code).__name__}"
        return failure
    code = extract_code(code) or code
    idea = str(idea or extract_idea(code) or "improver candidate")

    result = evaluator.evaluate(code, source=f"{source}_returned", idea=idea)
    try:
        reported_score = float(score)
    except (TypeError, ValueError):
        reported_score = 1e6

    return {
        "ok": result["valid"],
        "idea": idea,
        "code": code,
        "reported_score": reported_score,
        "result": result,
        "error": result.get("error"),
    }
