"""Offline tests for the global-controller experiment report."""

import json

from tools.controller_report import load_run, main, render_markdown, summarize


def _write(run_dir, mode, progress, decisions=()):
    run_dir.mkdir(parents=True)
    lines = []
    for action, calls, q in progress:
        lines.append({"level": "global", "event": "progress", "mode": mode, "seed": 0,
                      "action": action, "llm_calls": calls, "elite_quality": q})
    for d in decisions:
        lines.append({"level": "global", "event": "decision", "mode": mode, **d})
    (run_dir / "dual_hifo_events.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return run_dir


def _decision(action, cost, q_before, q_after, stagnation=0, reward=0.0):
    return {"action": action, "cost_llm_calls": cost, "q_before": q_before, "q_after": q_after,
            "reward": reward, "raw": {"inner_stagnation_rounds": stagnation}}


def _runs(tmp_path):
    ind = _write(tmp_path / "ind", "dual_hifo_independent",
                 [("SEED", 100, 1.0), ("OUTER_STEP", 600, 0.8), ("OUTER_STEP", 1100, 0.7)])
    rule = _write(tmp_path / "rule", "dual_hifo_global_rule",
                  [("SEED", 100, 1.0), ("INNER_EXPLORE", 150, 0.9), ("INNER_EXPLOIT", 200, 0.7),
                   ("OUTER_EXPLORE", 700, 0.6)],
                  [_decision("INNER_EXPLORE", 50, 1.0, 0.9, reward=0.2),
                   _decision("INNER_EXPLOIT", 50, 0.9, 0.7, reward=0.4),
                   _decision("OUTER_EXPLORE", 500, 0.7, 0.6, stagnation=2, reward=0.03)])
    return ind, rule


def test_calls_to_target_uses_reference_arm_final_quality(tmp_path):
    ind, rule = _runs(tmp_path)
    summary = summarize([load_run(ind), load_run(rule)])
    assert summary["target_quality"] == 0.7
    runs = {r["mode"]: r for r in summary["runs"]}
    assert runs["dual_hifo_independent"]["calls_to_target"] == 1100
    assert runs["dual_hifo_global_rule"]["calls_to_target"] == 200
    assert runs["dual_hifo_global_rule"]["final_quality"] == 0.6
    modes = summary["modes"]
    assert modes["dual_hifo_global_rule"]["reached_target"] == 1
    # Anytime mean quality over the common budget [0, 700].
    assert modes["dual_hifo_independent"]["anytime_mean_quality"] > modes["dual_hifo_global_rule"]["anytime_mean_quality"]
    stats = modes["dual_hifo_global_rule"]["actions"]
    assert stats["INNER_EXPLORE"]["count"] == 1
    assert stats["OUTER_EXPLORE"]["mean_cost"] == 500
    assert stats["INNER_EXPLOIT"]["success_rate"] == 1.0
    assert stats["OUTER_EXPLORE"]["success_rate_after_stagnation"] == 1.0
    assert stats["INNER_EXPLORE"]["success_rate_after_stagnation"] is None


def test_explicit_target_and_markdown(tmp_path):
    ind, rule = _runs(tmp_path)
    summary = summarize([load_run(ind), load_run(rule)], target=0.65)
    runs = {r["mode"]: r for r in summary["runs"]}
    assert runs["dual_hifo_independent"]["calls_to_target"] is None
    assert runs["dual_hifo_global_rule"]["calls_to_target"] == 700
    text = render_markdown(summary)
    assert "Calls-to-target" in text
    assert "dual_hifo_global_rule" in text and "dual_hifo_independent" in text


def test_cli_writes_outputs(tmp_path):
    ind, rule = _runs(tmp_path)
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    assert main([str(ind), str(rule), "--md-out", str(md), "--json-out", str(js)]) == 0
    assert "Calls-to-target" in md.read_text()
    assert json.loads(js.read_text())["target_quality"] == 0.7
