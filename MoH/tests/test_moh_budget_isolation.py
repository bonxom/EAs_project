"""Offline tests for MoH budget, seed-bound, solution isolation, and full_moh parity.

No real LLM, no TSP dataset, no subprocess eval.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import moh as moh_module
from dual_hifo import DualHiFoMoH
from moh import MoH

MOH_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_A = '''def improve_algorithm(population, utility, language_model, function_format, task):
    response = language_model.prompt("writer", "candidate a", temperature=0.1)
    return "idea-a", response, utility(response, "idea-a", task)
'''

CANDIDATE_B = '''def improve_algorithm(population, utility, language_model, function_format, task):
    response = language_model.prompt("writer", "candidate b", temperature=0.1)
    return "idea-b", response, utility(response, "idea-b", task)
'''


class QuietLLM:
    batch_size = 2
    model = "quiet"
    temperature = 0.0

    def prompt(self, expertise, message, temperature=None):
        return "ok"

    def prompt_batch(self, expertise, message_batch, temperature=None):
        return ["ok"] * len(message_batch)


class ScriptedLLM:
    batch_size = 2
    model = "scripted"
    temperature = 0.7

    def __init__(self):
        self.calls = []

    def prompt(self, expertise, message, temperature=None):
        self.calls.append(("prompt", expertise, message, temperature))
        if "analyze a given function provided as a string" in expertise:
            return "```txt\n2\n```"
        return "```python\n# idea\n" + CANDIDATE_A + "\n```"

    def prompt_batch(self, expertise, message_batch, temperature=None):
        self.calls.append(("batch", expertise, len(message_batch), temperature))
        return ["```python\n# idea\n" + CANDIDATE_A + "\n```"] * len(message_batch)


class FakeGeneratedImprover:
    """Stands in for GeneratedImprover: same parent-side call contract, no child process."""

    def __init__(self, code, timeout=3600, seed=0):
        self.code = code
        self.timeout = timeout
        self.seed = seed

    def validate(self):
        return None

    def __call__(self, population, utility, language_model, function_format, task):
        return "idea-a", CANDIDATE_A, utility(CANDIDATE_A, "idea-a", task)


def _cfg(isolate=False, seed_max=2, n_iterations=1):
    return OmegaConf.create({
        "n_iterations": n_iterations,
        "seed": 0,
        "pop_size": 5,
        "timeout": 5,
        "optimizer_timeout": 5,
        "max_eval_calls": 100,
        "seed_max_attempts": seed_max,
        "isolate_base_solutions": isolate,
        "problem": {
            "problem_name": "tsp_gls",
            "problem_size": [20, 50],
            "threshold": {"default": 6},
        },
    })


def _make_moh(workdir, isolate=False, seed_max=2, llm=None):
    os.chdir(workdir)
    return MoH(_cfg(isolate=isolate, seed_max=seed_max), str(MOH_ROOT), llm or QuietLLM())


# ---------------------------------------------------------------------------
# F2: eval budget pre-check
# ---------------------------------------------------------------------------

def test_eval_budget_exhausted_returns_sentinel_without_subprocess(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moh = _make_moh(tmp_path)
    moh.max_eval_calls = 1
    moh._total_eval_calls = 1

    def _should_not_run(*args, **kwargs):
        raise AssertionError("_run_code must not be called when budget is exhausted")

    monkeypatch.setattr(moh, "_run_code", _should_not_run)
    score = moh.evaluate_heuristic(CANDIDATE_A, "tsp_gls-20", "val")
    assert score == 1e6


def test_eval_budget_remaining_still_invokes_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moh = _make_moh(tmp_path)
    moh.max_eval_calls = 2
    moh._total_eval_calls = 0
    moh._cur_iter = 0

    class FakeProc:
        def communicate(self, timeout=None):
            return b"", b""

    def fake_run_code(algorithm_str, problem_type, mode, stdout_path):
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write("1.23\n")
        return FakeProc()

    monkeypatch.setattr(moh, "_run_code", fake_run_code)
    score = moh.evaluate_heuristic(CANDIDATE_A, "tsp_gls-20", "val")
    assert score == 1.23
    assert moh._total_eval_calls == 1


# ---------------------------------------------------------------------------
# F1: seed generation bound
# ---------------------------------------------------------------------------

def test_generate_seed_algorithm_raises_when_no_seeds_after_bound(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moh = _make_moh(tmp_path, seed_max=2)
    monkeypatch.setattr(moh, "generate_directions", lambda *a, **k: ["dir1", "dir2"])
    monkeypatch.setattr(moh, "evaluate_heuristic", lambda *a, **k: 1e6)
    with pytest.raises(RuntimeError, match="no valid seeds"):
        moh.generate_seed_algorithm("def f():\n    return 1", "tsp_gls-20", 6.0)


def test_generate_seed_algorithm_accepts_partial_population(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moh = _make_moh(tmp_path, seed_max=1)
    monkeypatch.setattr(moh, "generate_directions", lambda *a, **k: ["dir1"])
    monkeypatch.setattr(
        moh.heu_llm, "prompt",
        lambda *a, **k: "{seed idea}\n```python\ndef improve_algorithm():\n    return 1\n```",
    )
    calls = {"n": 0}

    def eval_ok(algorithm_str, problem_type, mode="val"):
        calls["n"] += 1
        return 1.0 if calls["n"] <= 2 else 1e6

    monkeypatch.setattr(moh, "evaluate_heuristic", eval_ok)
    selected = moh.generate_seed_algorithm(
        "def improve_algorithm():\n    return 1", "tsp_gls-20", 6.0,
    )
    # Duplicate solutions are rejected by Pop; one round yields one unique seed.
    assert moh.subtask_pop.get_subtask_size("tsp_gls-20") == 1
    assert selected["utility"] == 1.0
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# F3: solution isolation
# ---------------------------------------------------------------------------

def test_isolate_base_solutions_writes_under_run_local_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moh = _make_moh(tmp_path, isolate=True)
    monkeypatch.setattr(
        moh, "generate_seed_algorithm",
        lambda *a, **k: (moh.subtask_pop.save_solution("tsp_gls-20", "i", CANDIDATE_A, 1.0)
                         or moh.subtask_pop.get_random_solution("tsp_gls-20")),
    )
    shared = Path(moh.root_dir) / "problems" / "tsp_gls" / "solution_tsp_gls-20.json"
    before = shared.read_bytes() if shared.exists() else None

    moh.read_base_algorithm("tsp_gls-20")

    iso = Path(moh.run_logger.output_dir) / "base_solutions" / "solution_tsp_gls-20.json"
    assert iso.is_file()
    after = shared.read_bytes() if shared.exists() else None
    assert after == before


def test_shared_base_solutions_when_isolation_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moh = _make_moh(tmp_path, isolate=False)
    path = moh._solution_path("tsp_gls-20")
    assert path == os.path.join(moh.root_dir, "problems", "tsp_gls", "solution_tsp_gls-20.json")


# ---------------------------------------------------------------------------
# F4: run-local heuristic file
# ---------------------------------------------------------------------------

def test_run_code_uses_run_local_heuristic_and_sys_executable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    moh = _make_moh(tmp_path)
    moh._cur_iter = 0
    shared_gpt = Path(moh.root_dir) / "problems" / "tsp_gls" / "gpt.py"
    gpt_before = shared_gpt.read_bytes() if shared_gpt.exists() else None
    captured = {}

    class FakeProc:
        def communicate(self, timeout=None):
            return b"", b""

    def fake_popen(cmd, stdout=None, stderr=None):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(moh_module.subprocess, "Popen", fake_popen)
    stdout_path = str(tmp_path / "stdout.txt")
    moh._run_code(CANDIDATE_A, "tsp_gls-20", "val", stdout_path)

    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    assert "--heuristic-path" in cmd
    heuristic_path = cmd[cmd.index("--heuristic-path") + 1]
    assert os.path.isfile(heuristic_path)
    assert Path(heuristic_path).parent.name == "logs"
    gpt_after = shared_gpt.read_bytes() if shared_gpt.exists() else None
    assert gpt_after == gpt_before


# ---------------------------------------------------------------------------
# E5: full_moh parity between MoH and DualHiFoMoH(mode=full_moh)
# ---------------------------------------------------------------------------

def _stub_improver(population, utility, language_model, function_format, subtask):
    # try_improvement trusts the improver's returned meta utility (lower is better).
    # Seed meta utility is 1.0 via FakeGeneratedImprover; 0.5 forces acceptance.
    return "idea-b", CANDIDATE_B, 0.5


def _fake_eval(self, algorithm_str, problem_type, mode="val"):
    if algorithm_str == CANDIDATE_B:
        return 0.5
    if algorithm_str == CANDIDATE_A:
        return 1.0
    return 1e6


def _fake_read_base(self, subtask):
    self.subtask_pop.save_solution(subtask, "base-idea", CANDIDATE_A, 10.0)


def _run_one_iteration(cls, workdir, monkeypatch):
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(MoH, "evaluate_heuristic", _fake_eval)
    monkeypatch.setattr(MoH, "read_base_algorithm", _fake_read_base)
    monkeypatch.setattr(moh_module, "GeneratedImprover", FakeGeneratedImprover)
    llm = ScriptedLLM()
    cfg = _cfg(isolate=True, n_iterations=1)
    if cls is DualHiFoMoH:
        cfg["dual_hifo"] = {"mode": "full_moh"}
    np.random.seed(0)
    optimizer = cls(cfg, str(MOH_ROOT), llm)
    optimizer.get_seed()
    seed_meta = optimizer.meta_utility_val
    optimizer._cur_iter = 0
    accepted, candidate, _driver = optimizer.try_improvement(_stub_improver, _stub_improver)
    meta_csv = Path(optimizer.run_logger.output_dir) / "logs" / "meta_utility.csv"
    with meta_csv.open() as f:
        meta_rows = [tuple(row[:3]) for row in list(csv.reader(f))[1:]]
    util_csv = Path(optimizer.run_logger.output_dir) / "logs" / "utility.csv"
    with util_csv.open() as f:
        util_rows = [tuple(row[:3]) for row in list(csv.reader(f))[1:]]
    return {
        "seed_meta": seed_meta,
        "meta_utility_val": optimizer.meta_utility_val,
        "improver_str": optimizer.improver_str,
        "accepted": accepted,
        "candidate": candidate,
        "improver_pop": optimizer.improver_pop.get_population("meta-optimizer"),
        "subtask_pop": {
            task: optimizer.subtask_pop.get_population(task) for task in optimizer.subtask_list
        },
        "llm_calls": llm.calls,
        "meta_rows": meta_rows,
        "util_rows": util_rows,
        "events": (Path(optimizer.run_logger.output_dir) / "dual_hifo_events.jsonl").exists(),
    }


def test_full_moh_parity_with_dualhifo_full_moh_mode(tmp_path, monkeypatch):
    plain = _run_one_iteration(MoH, tmp_path / "plain", monkeypatch)
    dual = _run_one_iteration(DualHiFoMoH, tmp_path / "dual", monkeypatch)

    assert plain["seed_meta"] == dual["seed_meta"]
    assert plain["meta_utility_val"] == dual["meta_utility_val"]
    assert plain["improver_str"] == dual["improver_str"]
    assert plain["accepted"] is True
    assert dual["accepted"] is True
    assert plain["improver_pop"] == dual["improver_pop"]
    assert plain["subtask_pop"] == dual["subtask_pop"]
    assert plain["llm_calls"] == dual["llm_calls"]
    assert plain["meta_rows"] == dual["meta_rows"]
    assert plain["util_rows"] == dual["util_rows"]
    assert plain["events"] is False
    assert dual["events"] is False
