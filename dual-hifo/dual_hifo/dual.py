"""Run the outer (improver) and inner (TSP heuristic) loops for one size."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .evaluator import TSPEvaluator, apply_improver
from .hifo_loop import HiFoLoop
from .llm import ChatClient, LoggedLLM
from .population import Population
from .prompts import build_inner_prompts, build_outer_prompts
from .run_logger import RunLogger
from .tsp import load_instances


def run_one_size(*, size: int, cfg: dict, client: ChatClient, output_dir: Path,
                 seed_heuristic: str, seed_improver: str) -> Dict[str, Any]:
    run_dir = output_dir / f"size_{size}"
    logger = RunLogger(run_dir, size)
    random.seed(cfg.get("random_seed", 2024) + size)
    np.random.seed(cfg.get("random_seed", 2024) + size)

    instances, dataset_info = load_instances(
        size=size,
        n_instances=int(cfg.get("n_instances", 100)),
    )
    logger.info(f"[dataset] {dataset_info['file']} instances=0..{dataset_info['n_instances'] - 1}")

    evaluator = TSPEvaluator(
        instances=instances,
        size=size,
        logger=logger,
        neighbor_size=cfg.get("neighbor_size"),
        timeout=int(cfg.get("eval_timeout", 45)),
    )
    seed_result = evaluator.evaluate(seed_heuristic, source="seed_heuristic",
                                     idea="seed heuristic")
    if not seed_result["valid"]:
        raise RuntimeError(f"invalid seed heuristic: {seed_result['error']}")

    seed_population = Population(max_size=10)
    seed_population.add("seed heuristic", seed_heuristic,
                        seed_result["objective"], seed_result)

    # Outer loop: score each improver by the heuristic it generates from the seed population.
    outer_llm = LoggedLLM(client, logger, loop="outer", tag_prefix="outer")
    outer_eval_llm = LoggedLLM(client, logger, loop="outer", tag_prefix="improver_call")

    def evaluate_outer(code: str, *, source: str, **_) -> Dict[str, Any]:
        result = apply_improver(
            improver_code=code,
            population=seed_population,
            evaluator=evaluator,
            language_model=outer_eval_llm,
            logger=logger,
            size=size,
            source=source,
            timeout=int(cfg.get("improver_timeout", 300)),
        )
        objective = result["result"]["objective"] if result["ok"] else 1e6
        return {"objective": objective, "valid": result["ok"], "error": result["error"]}

    outer_loop = HiFoLoop(
        name="outer",
        cfg=cfg["outer"],
        prompts=build_outer_prompts(seed_improver),
        evaluate=evaluate_outer,
        llm=outer_llm,
        logger=logger,
        seed_items=[{"idea": "seed improver", "code": seed_improver}],
    )
    outer_result = outer_loop.run()
    best_improver = outer_result["best"]
    if not best_improver:
        raise RuntimeError("outer loop produced no valid improver")
    improver_path = logger.save_code(best_improver["code"], "best_improver")

    # Inner loop: evolve TSP heuristics and add meta-guided offspring each generation.
    inner_llm = LoggedLLM(client, logger, loop="inner", tag_prefix="inner")
    meta_llm = LoggedLLM(client, logger, loop="inner", tag_prefix="meta_improver")
    n_meta = int(cfg["inner"].get("meta_offspring", 1))

    def extra_offspring(population: Population, generation: int) -> List[Dict[str, Any]]:
        offspring = []
        for index in range(n_meta):
            result = apply_improver(
                improver_code=best_improver["code"],
                population=population,
                evaluator=evaluator,
                language_model=meta_llm,
                logger=logger,
                size=size,
                source=f"inner_gen{generation}_meta{index + 1}",
                timeout=int(cfg.get("improver_timeout", 300)),
            )
            if result["ok"]:
                offspring.append({
                    "idea": result["idea"],
                    "code": result["code"],
                    "objective": result["result"]["objective"],
                    "result": result["result"],
                })
        return offspring

    inner_loop = HiFoLoop(
        name="inner",
        cfg=cfg["inner"],
        prompts=build_inner_prompts(seed_heuristic),
        evaluate=evaluator.evaluate,
        llm=inner_llm,
        logger=logger,
        seed_items=[{"idea": "seed heuristic", "code": seed_heuristic}],
        extra_offspring=extra_offspring,
    )
    inner_result = inner_loop.run()
    best_heuristic = inner_result["best"]
    if not best_heuristic:
        raise RuntimeError("inner loop produced no valid heuristic")
    heuristic_path = logger.save_code(best_heuristic["code"], "best_heuristic")

    (run_dir / "outer_population.json").write_text(
        json.dumps(outer_result["population"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "inner_population.json").write_text(
        json.dumps(inner_result["population"], ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "size": size,
        "dataset": dataset_info,
        "seed_objective": seed_result["objective"],
        "best_improver": {
            "objective": best_improver["objective"],
            "idea": best_improver.get("idea", ""),
            "code_file": str(improver_path),
        },
        "best_heuristic": {
            "objective": best_heuristic["objective"],
            "idea": best_heuristic.get("idea", ""),
            "mean_cost": best_heuristic.get("result", {}).get("mean_cost"),
            "mean_optimal_cost": best_heuristic.get("result", {}).get("mean_optimal_cost"),
            "code_file": str(heuristic_path),
        },
        "outer": {
            "objectives": [item["objective"] for item in outer_result["population"]],
            "insights": outer_result["insights"],
        },
        "inner": {
            "objectives": [item["objective"] for item in inner_result["population"]],
            "best_history": inner_loop.best_history,
            "insights": inner_result["insights"],
        },
        "llm_calls": logger.llm_count,
        "tsp_evaluations": logger.tsp_count,
    }
    logger.save_summary(summary)
    logger.info(f"[done] size {size}: gap={best_heuristic['objective']:.4f}%")
    return summary
