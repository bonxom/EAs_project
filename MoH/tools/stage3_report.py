#!/usr/bin/env python3
"""Aggregate Stage 3 campaign results into report + CSV/JSON.

Usage: uv run python tools/stage3_report.py outputs/dualhifo/stage3_...
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_log_metrics(log_path: Path) -> dict:
    if not log_path.is_file():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    evals = [int(x) for x in re.findall(r"Eval call: (\d+)/\d+", text)]
    gaps = [float(x) for x in re.findall(r"ave_gap=([0-9.eE+-]+)", text)]
    final_pb = None
    m = re.search(r"Final provider budget: (\{[^}]+\})", text)
    if m:
        try:
            final_pb = json.loads(m.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            final_pb = {"raw": m.group(1)}
    meta_utils = re.findall(r"meta_utility: val=([0-9.eE+-]+)", text)
    accepted = "Improvement successful" in text
    stop = None
    for line in text.splitlines():
        if "stop_reason=" in line:
            stop = line.rsplit("stop_reason=", 1)[-1].strip()
    return {
        "max_eval_call": max(evals) if evals else 0,
        "n_evals_logged": len(evals),
        "best_gap_in_log": min(gaps) if gaps else None,
        "worst_gap_in_log": max(gaps) if gaps else None,
        "final_provider": final_pb,
        "meta_utility_values": [float(x) for x in meta_utils],
        "accepted_any": accepted,
        "stop_reason_log": stop,
    }


def parse_checkpoint_meta(ckpt_dir: Path) -> dict:
    meta_path = ckpt_dir / "metadata.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    elites = {}
    for art in meta.get("elite_artifacts", []):
        size = art.get("source_size")
        p = ckpt_dir / art.get("path", "")
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            best = None
            if data.get("elites"):
                best = min(float(e["training_utility"]) for e in data["elites"])
            elites[size] = {
                "n_elites": len(data.get("elites", [])),
                "best_training_utility": best,
            }
    return {
        "training_sizes": meta.get("training_sizes"),
        "seed": meta.get("seed"),
        "source_run": meta.get("source_run"),
        "optimizer_sha256": (meta.get("optimizer") or {}).get("sha256"),
        "elites_by_size": elites,
        "metadata": meta,
    }


def parse_dual_hifo_events(path: Path) -> dict:
    if not path.is_file():
        return {}
    outer, inner = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        (outer if o.get("level") == "outer" else inner).append(o)
    outer_sel = [e for e in outer if e.get("event") == "selection"]
    regimes = {}
    for e in inner:
        reg = (e.get("state") or {}).get("regime")
        if reg:
            regimes[reg] = regimes.get(reg, 0) + 1
    return {
        "n_outer": len(outer),
        "n_inner": len(inner),
        "outer_selections": [
            {
                "accepted": e.get("accepted"),
                "regime": (e.get("state") or {}).get("regime"),
                "stagnation": (e.get("state") or {}).get("stagnation_count"),
            }
            for e in outer_sel
        ],
        "inner_regimes": regimes,
    }


def collect_run(run_dir: Path, log_path: Path, tag: str, mode: str, seed: int, status: str) -> dict:
    row = {
        "tag": tag,
        "mode": mode,
        "seed": seed,
        "status": status,
        "run_dir": str(run_dir),
    }
    row.update(parse_log_metrics(log_path))
    ckpt = run_dir / "checkpoint"
    row["checkpoint"] = parse_checkpoint_meta(ckpt)
    row["dual_hifo"] = parse_dual_hifo_events(run_dir / "dual_hifo_events.jsonl")
    # utility.csv
    util_csv = run_dir / "logs" / "utility.csv"
    if util_csv.is_file():
        rows = list(csv.DictReader(util_csv.open(encoding="utf-8")))
        row["utility_csv_rows"] = rows
        # best val gap per size if columns allow
        by_size = {}
        for r in rows:
            for k, v in r.items():
                if "utility" in k.lower() or "gap" in k.lower():
                    try:
                        by_size.setdefault(k, []).append(float(v))
                    except (TypeError, ValueError):
                        pass
        row["utility_best_by_col"] = {k: min(vs) for k, vs in by_size.items() if vs}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    campaign_dir = args.campaign_dir
    status_path = campaign_dir / "campaign_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))

    results = []
    for run in status.get("runs", []):
        if run.get("status") == "skipped_existing":
            results.append({**run, "checkpoint": {}, "dual_hifo": {}})
            continue
        run_dir = Path(run["run_dir"])
        log = Path(run.get("log", campaign_dir / f"{run['tag']}.log"))
        results.append(
            collect_run(
                run_dir, log, run["tag"], run["mode"], int(run["seed"]), run["status"]
            )
        )

    # Best validation meta utility per mode/seed from checkpoint metadata
    for r in results:
        meta = (r.get("checkpoint") or {}).get("metadata") or {}
        r["training_meta_utility"] = meta.get("training_meta_utility")
        r["stop_reason_ckpt"] = meta.get("stop_reason")
        r["eval_calls_ckpt"] = meta.get("eval_calls")

    summary = status.get("summary", {})
    out_json = args.out_json or (campaign_dir / "stage3_results.json")
    out_json.write_text(
        json.dumps(
            {
                "campaign": {k: v for k, v in status.items() if k != "runs"},
                "summary": summary,
                "runs": results,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # CSV: one row per run
    out_csv = args.out_csv or (campaign_dir / "stage3_runs.csv")
    fields = [
        "tag", "mode", "seed", "status", "stop_reason", "training_meta_utility",
        "best_gap_in_log", "max_eval_call", "n_evals_logged",
        "accepted_any", "duration_s", "exit_code",
        "req", "att", "run_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            pb = r.get("final_provider") or {}
            w.writerow(
                {
                    "tag": r.get("tag"),
                    "mode": r.get("mode"),
                    "seed": r.get("seed"),
                    "status": r.get("status"),
                    "stop_reason": r.get("stop_reason") or r.get("stop_reason_ckpt"),
                    "training_meta_utility": r.get("training_meta_utility"),
                    "best_gap_in_log": r.get("best_gap_in_log"),
                    "max_eval_call": r.get("max_eval_call"),
                    "n_evals_logged": r.get("n_evals_logged"),
                    "accepted_any": r.get("accepted_any"),
                    "duration_s": r.get("duration_s"),
                    "exit_code": r.get("exit_code"),
                    "req": pb.get("requests"),
                    "att": pb.get("attempts"),
                    "run_dir": r.get("run_dir"),
                }
            )

    # Markdown report
    lines = []
    lines.append("# Stage 3 Overnight Dual-HiFo Experiment Report")
    lines.append("")
    lines.append(f"- Campaign: `{status.get('campaign_tag')}`")
    lines.append(f"- Git SHA: `{status.get('git_sha')}`")
    lines.append(f"- Model: `{status.get('model')}`")
    lines.append(f"- Sizes: {status.get('problem_sizes')}")
    lines.append(f"- Started: {status.get('started_utc')} · Ended: {status.get('ended_utc')}")
    lines.append(f"- Duration: {status.get('total_duration_s')}s")
    lines.append(f"- Budget: n_iterations={status.get('n_iterations')}, pop={status.get('pop_size')}, "
                 f"max_eval_calls/size={status.get('max_eval_calls_per_subtask')} "
                 f"(total={status.get('max_eval_calls_total')}), "
                 f"requests={status.get('provider_budget', {}).get('max_requests')}, "
                 f"attempts={status.get('provider_budget', {}).get('max_attempts')}")
    lines.append("")
    lines.append("## 1. Run counts")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for k in ("n_completed", "n_failed", "n_timeout", "n_budget_exhausted",
              "n_incomplete", "n_skipped", "n_runs_attempted"):
        lines.append(f"| {k} | {summary.get(k, 0)} |")
    lines.append("")
    lines.append("## 2. Per-run results")
    lines.append("")
    lines.append("| Tag | Mode | Seed | Status | Meta-util | Best gap (log) | Evals | Accepted | Dur (s) | Stop |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|---:|---|")
    for r in results:
        mu = r.get("training_meta_utility")
        bg = r.get("best_gap_in_log")
        lines.append(
            f"| {r.get('tag')} | {r.get('mode')} | {r.get('seed')} | {r.get('status')} | "
            f"{mu if mu is not None else '-'} | {bg if bg is not None else '-'} | "
            f"{r.get('max_eval_call', '-')} | {r.get('accepted_any')} | "
            f"{r.get('duration_s', '-')} | {r.get('stop_reason') or r.get('stop_reason_ckpt') or '-'} |"
        )
    lines.append("")
    lines.append("## 3. Checkpoint elites (best training utility per size)")
    lines.append("")
    for r in results:
        elites = (r.get("checkpoint") or {}).get("elites_by_size") or {}
        if not elites:
            continue
        lines.append(f"### {r['tag']}")
        lines.append("")
        lines.append("| Size | Best train util | N elites |")
        lines.append("|---:|---:|---:|")
        for size in sorted(elites, key=lambda x: int(x) if str(x).isdigit() else 0):
            e = elites[size]
            lines.append(f"| {size} | {e.get('best_training_utility')} | {e.get('n_elites')} |")
        lines.append("")
    lines.append("## 4. HiFo decisions (completed dual/inner/outer runs)")
    lines.append("")
    for r in results:
        dh = r.get("dual_hifo") or {}
        if not dh:
            continue
        lines.append(f"### {r['tag']}")
        lines.append(f"- outer events={dh.get('n_outer')}, inner events={dh.get('n_inner')}")
        lines.append(f"- inner regimes: {dh.get('inner_regimes')}")
        for s in dh.get("outer_selections") or []:
            lines.append(f"- outer selection: accepted={s.get('accepted')} regime={s.get('regime')} stag={s.get('stagnation')}")
        lines.append("")
    lines.append("## 5. Checkpoint paths")
    lines.append("")
    for r in results:
        ck = Path(r.get("run_dir", "")) / "checkpoint"
        if (ck / "metadata.json").is_file():
            lines.append(f"- `{r['tag']}` → `{ck}`")
    lines.append("")
    lines.append("## 6. Caveats")
    lines.append("")
    lines.append("- Validation gaps on TSP20 may be near-zero due to known small-size overfit; compare modes on TSP50/100/200.")
    lines.append("- No test-set selection: optimizers chosen on validation only.")
    lines.append("- Do not claim Dual-HiFo superior without multi-seed completed comparisons.")
    lines.append("")

    out_md = args.out_md or (campaign_dir / "OVERNIGHT_EXPERIMENT_REPORT.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
