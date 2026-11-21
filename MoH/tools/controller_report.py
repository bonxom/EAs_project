"""Aggregate Dual-HiFo controller experiments from ``dual_hifo_events.jsonl``.

Offline only: reads telemetry, never calls an LLM or executes generated code.

Usage (from ``MoH/``):
    uv run python -m tools.controller_report outputs/controller/*/ \
        --md-out report.md --json-out report.json [--target 0.05]

Metrics (lower elite quality is better):
- final elite quality and total logical LLM calls per run;
- calls-to-target: first cumulative LLM-call count with quality <= target
  (default target = mean final quality of the reference arm);
- anytime mean quality over the common call budget (step function from 0);
- per-action count, mean cost, success rate, and success rate after inner
  stagnation (the C1-vs-C2 diagnostic).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REFERENCE_MODE = "dual_hifo_independent"


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    events_path = run_dir / "dual_hifo_events.jsonl"
    progress, decisions = [], []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "progress":
            progress.append(event)
        elif event.get("event") == "decision":
            decisions.append(event)
    if not progress:
        raise ValueError(f"No progress events in {events_path}")
    return {
        "run": str(run_dir),
        "mode": progress[0].get("mode", "unknown"),
        "seed": progress[0].get("seed"),
        "curve": [(float(p["llm_calls"]), p.get("elite_quality")) for p in progress],
        "decisions": decisions,
    }


def _final_quality(curve) -> float | None:
    values = [q for _, q in curve if _finite(q)]
    return min(values) if values else None


def _calls_to_target(curve, target) -> float | None:
    if target is None:
        return None
    for calls, q in curve:
        if _finite(q) and q <= target + 1e-12:
            return calls
    return None


def _anytime_mean(curve, budget) -> float | None:
    """Mean best-so-far quality over [0, budget]; quality before the first record = first record."""
    points = [(c, q) for c, q in curve if _finite(q)]
    if not points or budget is None or budget <= 0:
        return None
    area, best = 0.0, points[0][1]
    prev = 0.0
    for calls, q in points:
        if calls >= budget:
            break
        area += best * (max(calls, prev) - prev)
        prev = max(calls, prev)
        best = min(best, q)
    area += best * (budget - prev)
    return area / budget


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _std(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def _action_stats(decisions) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for d in decisions:
        grouped[d["action"]].append(d)
    result = {}
    for action, items in sorted(grouped.items()):
        def success(d):
            return _finite(d.get("q_before")) and _finite(d.get("q_after")) and d["q_after"] < d["q_before"] - 1e-12
        stagnant = [d for d in items if (d.get("raw") or {}).get("inner_stagnation_rounds", 0) >= 1]
        result[action] = {
            "count": len(items),
            "mean_cost": _mean([d.get("cost_llm_calls") for d in items]),
            "mean_reward": _mean([d.get("reward") for d in items]),
            "success_rate": sum(map(success, items)) / len(items),
            "success_rate_after_stagnation": (sum(map(success, stagnant)) / len(stagnant)) if stagnant else None,
        }
    return result


def summarize(runs: list[dict[str, Any]], target: float | None = None,
              reference_mode: str = REFERENCE_MODE) -> dict[str, Any]:
    if target is None:
        target = _mean([_final_quality(r["curve"]) for r in runs if r["mode"] == reference_mode])
    totals = [r["curve"][-1][0] for r in runs]
    budget = min(totals) if totals else None
    run_rows = []
    for r in runs:
        run_rows.append({
            "run": r["run"],
            "mode": r["mode"],
            "seed": r["seed"],
            "final_quality": _final_quality(r["curve"]),
            "total_llm_calls": r["curve"][-1][0],
            "calls_to_target": _calls_to_target(r["curve"], target),
            "anytime_mean_quality": _anytime_mean(r["curve"], budget),
            "decisions": len(r["decisions"]),
        })
    modes: dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in runs}):
        rows = [row for row in run_rows if row["mode"] == mode]
        mode_runs = [r for r in runs if r["mode"] == mode]
        reached = [row["calls_to_target"] for row in rows if row["calls_to_target"] is not None]
        modes[mode] = {
            "n_runs": len(rows),
            "final_quality_mean": _mean([row["final_quality"] for row in rows]),
            "final_quality_std": _std([row["final_quality"] for row in rows]),
            "total_llm_calls_mean": _mean([row["total_llm_calls"] for row in rows]),
            "reached_target": len(reached),
            "calls_to_target_mean": _mean(reached),
            "anytime_mean_quality": _mean([row["anytime_mean_quality"] for row in rows]),
            "actions": _action_stats([d for r in mode_runs for d in r["decisions"]]),
        }
    return {
        "target_quality": target,
        "reference_mode": reference_mode,
        "common_llm_budget": budget,
        "runs": run_rows,
        "modes": modes,
    }


def _fmt(value, digits=6) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Dual-HiFo Global Controller Report",
        "",
        f"- Target elite quality: {_fmt(summary['target_quality'])} (reference: `{summary['reference_mode']}`)",
        f"- Common LLM-call budget for anytime metric: {_fmt(summary['common_llm_budget'])}",
        "",
        "## Arms",
        "",
        "| Mode | Runs | Final quality (mean ± std) | LLM calls (mean) | Reached target | Calls-to-target (mean) | Anytime mean quality |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, m in summary["modes"].items():
        lines.append(
            f"| `{mode}` | {m['n_runs']} | {_fmt(m['final_quality_mean'])} ± {_fmt(m['final_quality_std'])} "
            f"| {_fmt(m['total_llm_calls_mean'])} | {m['reached_target']}/{m['n_runs']} "
            f"| {_fmt(m['calls_to_target_mean'])} | {_fmt(m['anytime_mean_quality'])} |"
        )
    lines += ["", "## Actions (C1 vs C2 diagnostic)", "",
              "| Mode | Action | Count | Mean cost | Mean reward | Success rate | Success after stagnation |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for mode, m in summary["modes"].items():
        for action, s in m["actions"].items():
            lines.append(
                f"| `{mode}` | {action} | {s['count']} | {_fmt(s['mean_cost'])} | {_fmt(s['mean_reward'])} "
                f"| {_fmt(s['success_rate'])} | {_fmt(s['success_rate_after_stagnation'])} |"
            )
    lines += ["", "## Runs", "", "| Run | Mode | Seed | Final quality | LLM calls | Calls-to-target | Decisions |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["runs"]:
        lines.append(
            f"| {row['run']} | `{row['mode']}` | {_fmt(row['seed'])} | {_fmt(row['final_quality'])} "
            f"| {_fmt(row['total_llm_calls'])} | {_fmt(row['calls_to_target'])} | {row['decisions']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="Run directories containing dual_hifo_events.jsonl")
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--reference-mode", default=REFERENCE_MODE)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    runs = []
    for run in args.runs:
        try:
            runs.append(load_run(run))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"skip {run}: {exc}")
    if not runs:
        print("No usable runs")
        return 1
    summary = summarize(runs, target=args.target, reference_mode=args.reference_mode)
    text = render_markdown(summary)
    if args.md_out:
        Path(args.md_out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if not args.md_out:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
