import json
from pathlib import Path

baseline_path = Path("outputs/full_comparison_n10/baseline_deploy/report.json")
rmoh_path = Path("outputs/full_comparison_n10/rmoh_deploy/report.json")

if baseline_path.exists():
    print("=== BASELINE-MOH DEPLOYMENT RESULTS (All 100 test instances per size) ===")
    with baseline_path.open() as f:
        rows = json.load(f)
    for r in rows:
        print(f"TSP{r['size']:<5} | Val Gap: {r['validation_gap_pct']:.6f}% | Test Gap: {r['test_gap_pct']:.6f}% | Status: {r['status']}")

if rmoh_path.exists():
    print("\n=== R-MOH DEPLOYMENT RESULTS (All 100 test instances per size) ===")
    with rmoh_path.open() as f:
        rows = json.load(f)
    for r in rows:
        print(f"TSP{r['size']:<5} | Val Gap: {r['validation_gap_pct']:.6f}% | Test Gap: {r['test_gap_pct']:.6f}% | Status: {r['status']}")
else:
    print("\nR-MoH deploy report not yet finalized (in progress).")
