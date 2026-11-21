"""Aggregate controller-arm zero-shot TSP-500/TSP-1000 deployment results.

Offline only: reads ``report.json`` artifacts written by ``deploy.py``; it never
calls an LLM and never executes generated code.

Run directories are grouped by an optional ``<mode>_seed<N>`` suffix, matching
the output layout of ``run_controller_experiment.sh``. Test gaps are reported
per run and per arm as mean ± std across seeds; the average test gap is only
computed when both target sizes have a finite test result.

Usage (from ``MoH/``)::

    uv run python -m tools.controller_deployment_report \
        outputs/controller_exp/deployment/*/ \
        --md-out deployment_report.md --json-out deployment_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

TARGET_SIZES = (500, 1000)
_RUN_NAME = re.compile(r"^(?P<mode>.+)_seed(?P<seed>\d+)$")
_METRICS = (
    "tsp500_test_gap_pct",
    "tsp1000_test_gap_pct",
    "avg_test_gap_pct",
    "tsp500_val_gap_pct",
    "tsp1000_val_gap_pct",
)


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def parse_run_name(name: str) -> tuple[str, int | None]:
    """Split ``<mode>_seed<N>``; leave other names as a single anonymous run."""
    match = _RUN_NAME.match(name)
    if match is None:
        return name, None
    return match.group("mode"), int(match.group("seed"))


def _gap(rows: dict[int, dict], size: int, key: str) -> float | None:
    row = rows.get(size)
    if row is None:
        return None
    value = row.get(key)
    return float(value) if _finite(value) else None


def _test_counts(rows: dict[int, dict]) -> dict[int, int]:
    counts = {}
    for size in TARGET_SIZES:
        row = rows.get(size)
        if row is not None and _finite(row.get("test_n_instances")):
            counts[size] = int(row["test_n_instances"])
    return counts


def load_deployment(run_dir: str | Path) -> dict[str, Any]:
    """Read one deployment output directory into a flat, reportable run row."""
    run_dir = Path(run_dir)
    mode, seed = parse_run_name(run_dir.name)
    result: dict[str, Any] = {
        "run": str(run_dir),
        "mode": mode,
        "seed": seed,
        "status": "invalid",
        "tsp500_test_gap_pct": None,
        "tsp1000_test_gap_pct": None,
        "avg_test_gap_pct": None,
        "tsp500_val_gap_pct": None,
        "tsp1000_val_gap_pct": None,
        "test_n_instances": {},
        "error": None,
    }
    try:
        rows = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("report.json must contain a list of rows")
    except (OSError, ValueError, TypeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result

    rows_by_size = {
        row["size"]: row
        for row in rows
        if isinstance(row, dict) and row.get("size") in TARGET_SIZES
    }
    result["test_n_instances"] = _test_counts(rows_by_size)
    for size in TARGET_SIZES:
        result[f"tsp{size}_test_gap_pct"] = _gap(rows_by_size, size, "test_gap_pct")
        result[f"tsp{size}_val_gap_pct"] = _gap(rows_by_size, size, "validation_gap_pct")

    tested = [
        size
        for size in TARGET_SIZES
        if rows_by_size.get(size, {}).get("status") == "ok"
        and _finite(rows_by_size[size].get("test_gap_pct"))
    ]
    if len(tested) == len(TARGET_SIZES):
        result["status"] = "ok"
        result["avg_test_gap_pct"] = (
            result["tsp500_test_gap_pct"] + result["tsp1000_test_gap_pct"]
        ) / 2
    elif tested:
        result["status"] = "partial"
    else:
        result["status"] = "failed"
    return result


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Group runs by arm and compute mean/std test gaps across seeds."""
    modes: dict[str, Any] = {}
    for mode in sorted({run["mode"] for run in runs}):
        group = [run for run in runs if run["mode"] == mode]
        statuses: dict[str, int] = {}
        for run in group:
            statuses[run["status"]] = statuses.get(run["status"], 0) + 1
        entry: dict[str, Any] = {
            "run_count": len(group),
            "seeds": sorted(run["seed"] for run in group if run["seed"] is not None),
            "statuses": statuses,
        }
        for key in _METRICS:
            entry[key] = _stats([run[key] for run in group if _finite(run[key])])
        modes[mode] = entry
    return {"runs": runs, "modes": modes}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_stat(stats: dict[str, Any]) -> str:
    if stats["n"] == 0:
        return "N/A"
    if stats["std"] is None:
        return _fmt(stats["mean"])
    return f"{_fmt(stats['mean'])} ± {_fmt(stats['std'])}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Controller Zero-Shot Deployment Test — TSP-500 / TSP-1000",
        "",
        (
            "Lower optimality gap is better. Values are mean ± std across "
            "completed runs; `n` is the number of runs contributing a finite "
            "test result."
        ),
        "",
        "| Mode | Runs | n | TSP-500 test gap (%) | TSP-1000 test gap (%) | Average test gap (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, entry in summary["modes"].items():
        lines.append(
            f"| `{mode}` | {entry['run_count']} | {entry['avg_test_gap_pct']['n']} "
            f"| {_fmt_stat(entry['tsp500_test_gap_pct'])} "
            f"| {_fmt_stat(entry['tsp1000_test_gap_pct'])} "
            f"| {_fmt_stat(entry['avg_test_gap_pct'])} |"
        )
    lines += [
        "",
        "## Per-run detail",
        "",
        "| Run | Seed | Status | TSP-500 test gap (%) | TSP-1000 test gap (%) | Average test gap (%) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['run']} | {_fmt(run['seed'])} | {run['status']} "
            f"| {_fmt(run['tsp500_test_gap_pct'])} "
            f"| {_fmt(run['tsp1000_test_gap_pct'])} "
            f"| {_fmt(run['avg_test_gap_pct'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "runs", nargs="+", type=Path, help="Deployment directories with report.json"
    )
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    runs = [load_deployment(run) for run in args.runs]
    summary = summarize(runs)
    text = render_markdown(summary)
    if args.md_out:
        Path(args.md_out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if not args.md_out:
        print(text, end="")
    for run in summary["runs"]:
        if run["error"]:
            print(f"skip {run['run']}: {run['error']}")
    return 1 if all(run["status"] == "invalid" for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
