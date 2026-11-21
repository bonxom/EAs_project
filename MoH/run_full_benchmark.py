"""Orchestration script for Full MoH Baseline vs R-MoH benchmark on all TSP datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
ALL_SIZES = [10, 20, 50, 100, 200, 500, 1000]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("FullBenchmark")


def restore_initial_solutions():
    src_dir = ROOT_DIR / "problems" / "tsp_gls" / "initial_solutions"
    dst_dir = ROOT_DIR / "problems" / "tsp_gls"
    if not src_dir.exists():
        logger.warning("No initial_solutions directory found at %s", src_dir)
        return
    for json_file in src_dir.glob("solution_*.json"):
        shutil.copy2(json_file, dst_dir / json_file.name)
    logger.info("Restored all 7 initial solution files from %s", src_dir)


def run_training_experiment(
    rmoh_enabled: bool,
    output_dir: Path,
    n_iterations: int = 10,
    seed: int = 0,
    sizes: list[int] | None = None,
) -> Path:
    sizes = sizes or ALL_SIZES
    sizes_str = "[" + ", ".join(str(s) for s in sizes) + "]"
    method_name = "R-MoH" if rmoh_enabled else "Baseline-MoH"

    checkpoint_dir = output_dir / "checkpoint"
    if (checkpoint_dir / "metadata.json").exists():
        logger.info("[%s] Checkpoint already exists at %s. Skipping training.", method_name, checkpoint_dir)
        return checkpoint_dir

    logger.info("=" * 80)
    logger.info("STARTING TRAINING: %s (n_iterations=%d, seed=%d)", method_name, n_iterations, seed)
    logger.info("Sizes: %s", sizes_str)
    logger.info("Output dir: %s", output_dir)
    logger.info("=" * 80)

    # Restore initial population to guarantee exact identical starting conditions
    restore_initial_solutions()

    cmd = [
        sys.executable,
        str(ROOT_DIR / "main.py"),
        f"rmoh.enabled={str(rmoh_enabled).lower()}",
        "llm_client@heu=local",
        "llm_client@meta=local",
        f"problem.problem_size={sizes_str}",
        f"n_iterations={n_iterations}",
        f"seed={seed}",
        f"hydra.run.dir={output_dir.resolve()}",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    started = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(ROOT_DIR), check=False)
    elapsed = time.time() - started

    if result.returncode != 0:
        logger.error("[%s] Training failed with return code %d (elapsed %.1fs)", method_name, result.returncode, elapsed)
        raise RuntimeError(f"Training failed for {method_name}")

    logger.info("[%s] Training completed successfully in %.1f seconds!", method_name, elapsed)
    if not (checkpoint_dir / "metadata.json").exists():
        raise RuntimeError(f"Expected checkpoint at {checkpoint_dir} was not created!")
    return checkpoint_dir


def run_test_evaluation(
    checkpoint_dir: Path,
    output_dir: Path,
    method_name: str,
    sizes: list[int] | None = None,
    test_n_instances: int = 100,
    inference_iterations: int = 10,
    seed: int = 0,
) -> dict:
    from run_evaluation_suite import evaluate_checkpoint

    sizes = sizes or ALL_SIZES
    logger.info("=" * 80)
    logger.info("STARTING DEPLOYMENT & EVALUATION: %s", method_name)
    logger.info("Target sizes: %s, Test instances per size: %d", sizes, test_n_instances)
    logger.info("Checkpoint: %s", checkpoint_dir)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 80)

    res = evaluate_checkpoint(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        target_sizes=sizes,
        test_n_instances=test_n_instances,
        val_n_instances=8,
        inference_iterations=inference_iterations,
        seed=seed,
    )
    return res


def compare_reports(baseline_report_path: Path, rmoh_report_path: Path, summary_output: Path) -> dict:
    with baseline_report_path.open(encoding="utf-8") as f:
        baseline_rows = json.load(f)
    with rmoh_report_path.open(encoding="utf-8") as f:
        rmoh_rows = json.load(f)

    baseline_by_size = {int(r["size"]): r for r in baseline_rows}
    rmoh_by_size = {int(r["size"]): r for r in rmoh_rows}

    comparison = []
    print("\n" + "=" * 90)
    print(f"{'Size':<8} | {'MoH Base Val Gap':<16} | {'R-MoH Val Gap':<16} | {'MoH Base Test Gap':<17} | {'R-MoH Test Gap':<17} | {'Improvement':<12}")
    print("-" * 90)

    for sz in sorted(set(baseline_by_size.keys()) | set(rmoh_by_size.keys())):
        b = baseline_by_size.get(sz, {})
        r = rmoh_by_size.get(sz, {})

        b_val = b.get("validation_gap_pct")
        r_val = r.get("validation_gap_pct")
        b_test = b.get("test_gap_pct")
        r_test = r.get("test_gap_pct")

        diff = None
        if b_test is not None and r_test is not None:
            diff = b_test - r_test  # positive means R-MoH has lower gap (better)

        b_val_s = f"{b_val:.4f}%" if b_val is not None else "N/A"
        r_val_s = f"{r_val:.4f}%" if r_val is not None else "N/A"
        b_test_s = f"{b_test:.4f}%" if b_test is not None else "N/A"
        r_test_s = f"{r_test:.4f}%" if r_test is not None else "N/A"
        diff_s = f"{diff:+.4f}%" if diff is not None else "N/A"

        print(f"TSP{sz:<5} | {b_val_s:<16} | {r_val_s:<16} | {b_test_s:<17} | {r_test_s:<17} | {diff_s:<12}")

        comparison.append({
            "size": sz,
            "baseline": b,
            "rmoh": r,
            "test_gap_diff_pct": diff,
        })

    print("=" * 90 + "\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparison": comparison,
    }
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved final comparison summary to %s", summary_output)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full Benchmark: MoH Baseline vs R-MoH on all TSP datasets")
    parser.add_argument("--mode", choices=["all", "train_baseline", "train_rmoh", "evaluate", "compare"], default="all")
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--test-instances", type=int, default=100)
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=ALL_SIZES)
    args = parser.parse_args()

    benchmark_root = PROJECT_ROOT / "outputs" / "full_comparison_n10"
    baseline_dir = benchmark_root / "baseline_train"
    rmoh_dir = benchmark_root / "rmoh_train"
    baseline_deploy = benchmark_root / "baseline_deploy"
    rmoh_deploy = benchmark_root / "rmoh_deploy"
    summary_file = benchmark_root / "comparison_summary.json"

    benchmark_root.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "train_baseline"):
        run_training_experiment(
            rmoh_enabled=False,
            output_dir=baseline_dir,
            n_iterations=args.n_iterations,
            seed=args.seed,
            sizes=args.sizes,
        )

    if args.mode in ("all", "train_rmoh"):
        run_training_experiment(
            rmoh_enabled=True,
            output_dir=rmoh_dir,
            n_iterations=args.n_iterations,
            seed=args.seed,
            sizes=args.sizes,
        )

    if args.mode in ("all", "evaluate"):
        if not (baseline_deploy / "report.json").exists():
            run_test_evaluation(
                checkpoint_dir=baseline_dir / "checkpoint",
                output_dir=baseline_deploy,
                method_name="Baseline-MoH",
                sizes=args.sizes,
                test_n_instances=args.test_instances,
                inference_iterations=args.inference_iterations,
                seed=args.seed,
            )
        else:
            logger.info("Baseline-MoH deployment report already exists at %s. Skipping.", baseline_deploy / "report.json")

        if not (rmoh_deploy / "report.json").exists():
            run_test_evaluation(
                checkpoint_dir=rmoh_dir / "checkpoint",
                output_dir=rmoh_deploy,
                method_name="R-MoH",
                sizes=args.sizes,
                test_n_instances=args.test_instances,
                inference_iterations=args.inference_iterations,
                seed=args.seed,
            )
        else:
            logger.info("R-MoH deployment report already exists at %s. Skipping.", rmoh_deploy / "report.json")

    if args.mode in ("all", "evaluate", "compare"):
        compare_reports(
            baseline_report_path=baseline_deploy / "report.json",
            rmoh_report_path=rmoh_deploy / "report.json",
            summary_output=summary_file,
        )


if __name__ == "__main__":
    main()
