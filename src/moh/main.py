import argparse
import json
import sys

import yaml

from moh.config import load_config
from moh.core.models import EvaluationContext, Heuristic
from moh.core.seeds import derive_seed
from moh.execution.heuristic_runner import HeuristicRunner
from moh.execution.protocol import ExecutionLimits
from moh.experiment import run_experiment
from moh.problems.baselines import NEAREST_NEIGHBOR_SOURCE, RANDOM_CHOICE_SOURCE
from moh.problems.tsp import TSPTask


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mini-MoH two-level TSP search")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument(
        "--mode", choices=["experiment", "tsp-demo"], default="experiment"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.mode == "tsp-demo":
            task = TSPTask.create(20, config.instances_per_task, config.seed)
            context = EvaluationContext(
                task.id,
                task.instance_seeds,
                tuple(
                    derive_seed(
                        config.seed, "task", task.id, "instance", i, "evaluation", 0
                    )
                    for i in range(config.instances_per_task)
                ),
            )
            runner = HeuristicRunner(ExecutionLimits(**config.execution.model_dump()))
            successful = True
            for name, source in [
                ("nearest", NEAREST_NEIGHBOR_SOURCE),
                ("random", RANDOM_CHOICE_SOURCE),
            ]:
                result = runner.evaluate(Heuristic(name, source), task, context)
                successful &= result.status == "success"
                print(
                    json.dumps(
                        {
                            "baseline": name,
                            "status": result.status,
                            "utility": result.utility,
                            "mean_length": -result.utility
                            if result.utility is not None
                            else None,
                            "error": result.error,
                        },
                        allow_nan=False,
                    )
                )
            return 0 if successful else 1
        result, path = run_experiment(config)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "winner": result.winner.candidate.id if result.winner else None,
                    "utility": result.winner.evaluation.utility
                    if result.winner
                    else None,
                    "run_dir": str(path),
                },
                allow_nan=False,
            )
        )
        return 0 if result.status == "success" else 1
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
