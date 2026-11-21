"""Run deployment and evaluation on all 7 TSP datasets for MoH vs R-MoH checkpoints."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parent
ALL_TARGET_SIZES = [10, 20, 50, 100, 200, 500, 1000]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_suite")


def evaluate_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    target_sizes: list[int] | None = None,
    test_n_instances: int = 100,
    val_n_instances: int = 8,
    inference_iterations: int = 10,
    seed: int = 0,
) -> dict:
    import sys
    sys.path.insert(0, str(ROOT_DIR))

    from deploy import run_deployment
    from utils.checkpoint import load_checkpoint
    from utils.heuristic_evaluator import HeuristicEvaluator
    from utils.llm_client.openai import OpenAIClient

    sizes = target_sizes or ALL_TARGET_SIZES
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_checkpoint = load_checkpoint(checkpoint_dir)
    logger.info("Loaded checkpoint from %s with training sizes: %s", checkpoint_dir, loaded_checkpoint.metadata.get("training_sizes"))

    # Initialize local LLM client for heuristic generation
    llm = OpenAIClient(
        model="ag/gemini-3.8-flash-low",
        temperature=1.0,
        batch_size=5,
        base_url="http://127.0.0.1:20128/v1",
        api_key="sk-74b3a2edd5c5d392-7ns8oo-0dd78a12",
        cache_dir=str(output_dir / "logs" / "llm" / "heu"),
    )

    evaluator = HeuristicEvaluator(
        ROOT_DIR,
        output_dir,
        seed=seed,
        val_count=val_n_instances,
        test_count=test_n_instances,
        workers=8,
        instance_timeout=90,
    )

    deploy_cfg = OmegaConf.create({
        "checkpoint": str(checkpoint_dir),
        "target_sizes": sizes,
        "inference_iterations": inference_iterations,
        "population_size": 10,
        "seed_count": 5,
        "initialization_attempts": 25,
        "reference_count": 2,
        "val_n_instances": val_n_instances,
        "test_n_instances": test_n_instances,
        "max_evaluations_per_target": 1000,
        "seed": seed,
        "optimizer_timeout": 3600,
        "instance_timeout": 90,
        "evaluation_timeout": None,
        "workers": 8,
        "evaluate_test": True,
    })

    rows = run_deployment(deploy_cfg, llm, evaluator=evaluator, checkpoint=loaded_checkpoint, output_dir=output_dir)
    report_file = output_dir / "report.json"
    logger.info("Saved deployment report to %s", report_file)
    return {"rows": rows, "report_file": str(report_file)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint across all TSP datasets")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--sizes", type=int, nargs="+", default=ALL_TARGET_SIZES)
    parser.add_argument("--test-instances", type=int, default=100)
    parser.add_argument("--val-instances", type=int, default=8)
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        args.output,
        target_sizes=args.sizes,
        test_n_instances=args.test_instances,
        val_n_instances=args.val_instances,
        inference_iterations=args.inference_iterations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
