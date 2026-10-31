"""Hydra entrypoint for checkpoint-driven MoH deployment."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import hydra
import numpy as np
from inference import InferenceRunner, TargetInitializationError
from utils.behavior_population import validate_fraction
from utils.checkpoint import load_checkpoint
from utils.deployment_report import evaluate_selected_heuristics, write_report
from utils.generated_improver import GeneratedImprover
from utils.heuristic_evaluator import HeuristicEvaluator

ROOT_DIR = Path(__file__).resolve().parent
SUPPORTED_TARGET_SIZES = {500, 1000}
logger = logging.getLogger(__name__)


def _validate_config(cfg) -> list[int]:
    validate_fraction(cfg.get("population", {}).get("diversity_fraction", 0.3))
    if not cfg.get("checkpoint"):
        raise ValueError("deployment checkpoint is required")
    targets = [int(size) for size in cfg.target_sizes]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_sizes must be a nonempty unique list")
    unsupported = set(targets) - SUPPORTED_TARGET_SIZES
    if unsupported:
        raise ValueError(f"Unsupported target sizes: {sorted(unsupported)}")
    positive_fields = (
        "population_size",
        "seed_count",
        "initialization_attempts",
        "reference_count",
        "val_n_instances",
        "test_n_instances",
        "max_evaluations_per_target",
        "optimizer_timeout",
        "instance_timeout",
        "workers",
    )
    invalid = [name for name in positive_fields if cfg.get(name) is None or cfg[name] <= 0]
    if invalid:
        raise ValueError(f"Positive values required for: {', '.join(invalid)}")
    if cfg.inference_iterations < 0:
        raise ValueError("inference_iterations cannot be negative")
    if cfg.population_size < cfg.seed_count:
        raise ValueError("population_size must be at least seed_count")
    if cfg.evaluation_timeout is not None and cfg.evaluation_timeout <= 0:
        raise ValueError("evaluation_timeout must be positive when configured")
    return targets


def _checkpoint_path(cfg) -> Path:
    return Path(hydra.utils.to_absolute_path(str(cfg.checkpoint)))


def preflight_deployment(cfg, evaluator=None, checkpoint=None):
    targets = _validate_config(cfg)
    loaded_checkpoint = checkpoint or load_checkpoint(_checkpoint_path(cfg))
    output_dir = Path.cwd()
    if evaluator is None:
        evaluator = HeuristicEvaluator(
            ROOT_DIR,
            output_dir,
            seed=cfg.seed,
            val_count=cfg.val_n_instances,
            test_count=cfg.test_n_instances,
            workers=cfg.workers,
            instance_timeout=cfg.instance_timeout,
            evaluation_timeout=cfg.evaluation_timeout,
        )
    elif (
        evaluator.val_count != cfg.val_n_instances
        or evaluator.test_count != cfg.test_n_instances
    ):
        raise ValueError("Injected evaluator counts do not match deployment config")
    evaluator.preflight(targets, include_test=bool(cfg.evaluate_test))
    GeneratedImprover(
        loaded_checkpoint.code,
        timeout=cfg.optimizer_timeout,
        seed=cfg.seed,
    ).validate()
    return loaded_checkpoint, evaluator, targets


def _execute_deployment(cfg, llm, checkpoint, evaluator, targets):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    output_dir = Path.cwd()
    heuristic_format = (ROOT_DIR / "prompts" / "tsp_gls" / "desc.txt").read_text(
        encoding="utf-8"
    )
    runner = InferenceRunner(
        checkpoint,
        llm,
        evaluator,
        output_dir,
        heuristic_format,
        diversity_fraction=cfg.get("population", {}).get("diversity_fraction", 0.3),
        iterations=cfg.inference_iterations,
        population_size=cfg.population_size,
        seed_count=cfg.seed_count,
        initialization_attempts=cfg.initialization_attempts,
        reference_count=cfg.reference_count,
        max_evaluations=cfg.max_evaluations_per_target,
        optimizer_timeout=cfg.optimizer_timeout,
        seed=cfg.seed,
    )
    selected = []
    target_errors = {}
    for size in targets:
        try:
            selected.append(runner.run_target(size))
        except TargetInitializationError as error:
            target_errors[size] = str(error)
            logger.warning("Target %s inference failed: %s", size, error)
        except Exception as error:
            target_errors[size] = f"{type(error).__name__}: {error}"
            logger.exception("Target %s inference failed", size)
    rows = evaluate_selected_heuristics(
        selected,
        evaluator,
        output_dir,
        checkpoint,
        target_errors=target_errors,
        evaluate_test=bool(cfg.evaluate_test),
    )
    write_report(rows, output_dir)
    return rows


def run_deployment(cfg, llm, evaluator=None, checkpoint=None) -> list[dict]:
    checkpoint, evaluator, targets = preflight_deployment(
        cfg, evaluator=evaluator, checkpoint=checkpoint
    )
    return _execute_deployment(cfg, llm, checkpoint, evaluator, targets)


@hydra.main(version_base=None, config_path="cfg", config_name="deployment")
def main(cfg):
    checkpoint, evaluator, targets = preflight_deployment(cfg)
    cache_dir = Path.cwd() / "logs" / "llm" / "heu"
    llm = hydra.utils.instantiate(cfg.heu, cache_dir=str(cache_dir))
    logger.info("Using heuristic LLM %s; raw responses: %s", cfg.heu.model, cache_dir)
    return _execute_deployment(cfg, llm, checkpoint, evaluator, targets)


if __name__ == "__main__":
    main()
