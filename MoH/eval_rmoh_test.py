"""Direct evaluation of trained R-MoH heuristics on the 100 test instances per dataset."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
ALL_SIZES = [10, 20, 50, 100, 200, 500, 1000]

sys.path.insert(0, str(ROOT_DIR))

from utils.heuristic_evaluator import HeuristicEvaluator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_rmoh_test")


def main():
    rmoh_subtask_path = PROJECT_ROOT / "outputs" / "full_comparison_n10" / "rmoh_train" / "pop" / "subtask" / "iter_9.json"
    baseline_report_path = PROJECT_ROOT / "outputs" / "full_comparison_n10" / "baseline_deploy" / "report.json"
    output_dir = PROJECT_ROOT / "outputs" / "full_comparison_n10" / "rmoh_direct_test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with rmoh_subtask_path.open(encoding="utf-8") as f:
        rmoh_subtasks = json.load(f)

    with baseline_report_path.open(encoding="utf-8") as f:
        baseline_rows = json.load(f)
    baseline_by_size = {r["size"]: r for r in baseline_rows}

    evaluator = HeuristicEvaluator(
        ROOT_DIR,
        output_dir,
        seed=0,
        val_count=8,
        test_count=100,
        workers=8,
        instance_timeout=90,
    )

    rmoh_results = []
    print("\n" + "=" * 95)
    print("EVALUATING R-MOH HEURISTICS ON FULL TEST SET (100 instances per size)")
    print("=" * 95)

    for size in ALL_SIZES:
        task_name = f"tsp_gls-{size}"
        candidates = rmoh_subtasks.get(task_name, [])
        if not candidates:
            logger.warning("No candidates found for %s", task_name)
            continue

        best_cand = min(candidates, key=lambda x: x["utility"])
        code = best_cand["best_sol"]
        idea = best_cand["idea"]
        train_val_gap = best_cand["utility"]

        logger.info("Evaluating TSP%d (train val gap: %.6f%%) on 100 test instances...", size, train_val_gap)
        started = time.time()
        test_res = evaluator.evaluate(code, size, "test")
        elapsed = time.time() - started

        test_gap = test_res.mean_gap_pct
        logger.info("TSP%d Test Completed in %.1fs: Test Gap = %.6f%% (status=%s)", size, elapsed, test_gap, test_res.status)

        rmoh_results.append({
            "size": size,
            "idea": idea,
            "train_val_gap_pct": train_val_gap,
            "test_gap_pct": test_gap,
            "test_failures": test_res.failures,
            "status": test_res.status,
            "elapsed_seconds": elapsed,
        })

    # Save R-MoH test results
    rmoh_report_path = output_dir / "report.json"
    with rmoh_report_path.open("w", encoding="utf-8") as f:
        json.dump(rmoh_results, f, indent=2)
    logger.info("Saved R-MoH test results to %s", rmoh_report_path)

    # Print Side-by-Side Comparison
    print("\n" + "=" * 105)
    print(f"{'Size':<8} | {'MoH Base Test Gap':<18} | {'R-MoH Test Gap':<18} | {'Delta (Baseline - R-MoH)':<26} | {'Winner':<10}")
    print("-" * 105)

    comparison = []
    for r in rmoh_results:
        sz = r["size"]
        b = baseline_by_size.get(sz, {})
        b_test = b.get("test_gap_pct")
        r_test = r.get("test_gap_pct")

        if b_test is not None and r_test is not None:
            delta = b_test - r_test
            winner = "R-MoH" if delta > 1e-6 else ("Baseline" if delta < -1e-6 else "Tie")
            delta_s = f"{delta:+.6f}%"
        else:
            delta = None
            winner = "N/A"
            delta_s = "N/A"

        b_test_s = f"{b_test:.6f}%" if b_test is not None else "N/A"
        r_test_s = f"{r_test:.6f}%" if r_test is not None else "N/A"

        print(f"TSP{sz:<5} | {b_test_s:<18} | {r_test_s:<18} | {delta_s:<26} | {winner:<10}")

        comparison.append({
            "size": sz,
            "baseline_test_gap_pct": b_test,
            "rmoh_test_gap_pct": r_test,
            "improvement_pct": delta,
            "winner": winner,
        })

    print("=" * 105 + "\n")

    summary_file = PROJECT_ROOT / "outputs" / "full_comparison_n10" / "final_comparison_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "ag/gemini-3.8-flash-low",
            "n_iterations": 10,
            "test_instances_per_dataset": 100,
            "comparison": comparison,
            "baseline_results": baseline_rows,
            "rmoh_results": rmoh_results,
        }, f, indent=2)
    logger.info("Saved final comparison summary to %s", summary_file)


if __name__ == "__main__":
    main()
