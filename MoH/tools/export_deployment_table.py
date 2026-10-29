#!/usr/bin/env python3
"""Export deployment iteration progress and test results into formatted table.

Usage:
    python tools/export_deployment_table.py outputs/deployment/tsp500-1000-fb_full_moh_seed0
    python tools/export_deployment_table.py outputs/deployment/full_budget_seed0/deploy_*
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_single_table(deploy_dir: Path, decimal_comma: bool = True) -> str:
    deploy_dir = Path(deploy_dir)
    p500 = deploy_dir / "populations" / "tsp500"
    p1000 = deploy_dir / "populations" / "tsp1000"
    report_file = deploy_dir / "report.json"

    if not p500.exists() or not p1000.exists():
        raise FileNotFoundError(f"Population directories not found in {deploy_dir}")

    tsp500_iters = []
    tsp1000_iters = []
    for i in range(10):
        f500 = p500 / f"iter_{i:03d}.json"
        f1000 = p1000 / f"iter_{i:03d}.json"
        if not f500.exists() or not f1000.exists():
            break
        tsp500_iters.append(json.loads(f500.read_text(encoding="utf-8"))["best_validation_gap_pct"])
        tsp1000_iters.append(json.loads(f1000.read_text(encoding="utf-8"))["best_validation_gap_pct"])

    n_iters = len(tsp500_iters)
    lines = []
    header = "Iteration\tTSP-500 Best Gap (%)\tTSP-1000 Best Gap (%)\tAverage Gap (%)\tΔ Avg Gap (điểm %)"
    lines.append(header)

    def fmt(val: float, is_delta: bool = False) -> str:
        s = f"{val:+.6f}" if is_delta else f"{val:.6f}"
        if decimal_comma:
            s = s.replace(".", ",")
        return s

    prev_avg = None
    for i in range(n_iters):
        label = "Iter 0 (seed)" if i == 0 else f"Iter {i}"
        g500 = tsp500_iters[i]
        g1000 = tsp1000_iters[i]
        avg = (g500 + g1000) / 2
        if prev_avg is None:
            delta_str = "–"
        else:
            delta = prev_avg - avg
            delta_str = fmt(delta, is_delta=True)
        prev_avg = avg
        lines.append(f"{label}\t{fmt(g500)}\t{fmt(g1000)}\t{fmt(avg)}\t{delta_str}")

    if report_file.exists():
        report_data = json.loads(report_file.read_text(encoding="utf-8"))
        test_gaps = {row["size"]: row.get("test_gap_pct") for row in report_data}
        if 500 in test_gaps and 1000 in test_gaps and test_gaps[500] is not None:
            t500 = test_gaps[500]
            t1000 = test_gaps[1000]
            t_avg = (t500 + t1000) / 2
            lines.append("")
            lines.append(f"test 100 instances\t{fmt(t500)}\t{fmt(t1000)}\t{fmt(t_avg)}\t-")

    return "\n".join(lines)


def generate_multi_summary(deploy_dirs: list[Path], decimal_comma: bool = True) -> str:
    lines = []
    lines.append("# BẢNG TỔNG HỢP SO SÁNH KẾT QUẢ TEST (100 INSTANCES)")
    lines.append("")
    lines.append("Mode / Thử nghiệm\tTSP-500 Test Gap (%)\tTSP-1000 Test Gap (%)\tAverage Test Gap (%)")

    def fmt(val: float | None) -> str:
        if val is None:
            return "N/A"
        s = f"{val:.6f}"
        if decimal_comma:
            s = s.replace(".", ",")
        return s

    for d in deploy_dirs:
        d = Path(d)
        report_file = d / "report.json"
        if not report_file.exists():
            continue
        try:
            report_data = json.loads(report_file.read_text(encoding="utf-8"))
            test_gaps = {row["size"]: row.get("test_gap_pct") for row in report_data}
            t500 = test_gaps.get(500)
            t1000 = test_gaps.get(1000)
            t_avg = (t500 + t1000) / 2 if (t500 is not None and t1000 is not None) else None
            lines.append(f"{d.name}\t{fmt(t500)}\t{fmt(t1000)}\t{fmt(t_avg)}")
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            lines.append(f"{d.name}\tError: {e}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("# CHI TIẾT TỪNG MODE")
    lines.append("=" * 80)

    for d in deploy_dirs:
        d = Path(d)
        if (d / "populations").exists():
            lines.append(f"\n### Mode: {d.name}\n")
            try:
                lines.append(generate_single_table(d, decimal_comma=decimal_comma))
            except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                lines.append(f"Failed to generate table: {e}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export deployment progress table")
    parser.add_argument("deploy_dirs", type=Path, nargs="+", help="Path(s) to deployment output folder(s)")
    parser.add_argument("--dot", action="store_true", help="Use dot instead of comma for decimal separator")
    args = parser.parse_args()

    dirs = [p for p in args.deploy_dirs if p.is_dir()]
    if not dirs:
        print("No valid deployment directories provided.")
        return

    if len(dirs) == 1:
        print(generate_single_table(dirs[0], decimal_comma=not args.dot))
    else:
        print(generate_multi_summary(dirs, decimal_comma=not args.dot))


if __name__ == "__main__":
    main()
