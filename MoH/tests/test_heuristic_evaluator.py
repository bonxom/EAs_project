import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from deployment_fakes import HEURISTIC

FAKE_EVAL = r'''import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("size", type=int)
parser.add_argument("root")
parser.add_argument("mode")
parser.add_argument("timeout")
parser.add_argument("seed", type=int)
parser.add_argument("--count", type=int, required=True)
parser.add_argument("--heuristic-path", required=True)
parser.add_argument("--result-path", required=True)
parser.add_argument("--workers", type=int, required=True)
args = parser.parse_args()
dataset = Path(args.root) / "problems" / "tsp_gls" / "data" / args.mode / f"{args.mode}_instance_data_sol_{args.size}.pkl"
with dataset.open("rb") as source:
    data = pickle.load(source)
available = len(data["cost"])
indices = (random.Random(args.seed).sample(range(available), args.count)
           if args.mode == "val" else list(range(args.count)))
code = Path(args.heuristic_path).read_bytes()
payload = {
    "size": args.size,
    "mode": args.mode,
    "mean_gap_pct": 1.25,
    "status": "ok",
    "failures": 0,
    "indices": indices,
    "code_sha256": hashlib.sha256(code).hexdigest(),
    "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    "elapsed_seconds": 0.01,
    "seed": args.seed,
    "count": args.count,
    "solver_settings": {
        "workers": min(args.workers, args.count),
        "instance_timeout": int(args.timeout),
        "time_limit": 20,
        "iteration_limit": 1000,
        "perturbation_moves": 1,
    },
    "instances": [],
}
Path(args.result_path).write_text(json.dumps(payload))
print("1.25")
'''


def _dataset(count):
    return {
        "cost": list(range(1, count + 1)),
        "distance_matrix": list(range(count)),
        "coordinate": list(range(count)),
        "optimal_tour": list(range(count)),
    }


def _root(tmp_path: Path, *, val_count=8, test_count=100) -> Path:
    root = tmp_path / "root"
    problem = root / "problems" / "tsp_gls"
    problem.mkdir(parents=True)
    (problem / "eval.py").write_text(FAKE_EVAL)
    (problem / "gpt.py").write_text("SHARED_SENTINEL = True\n")
    for mode, count in (("val", val_count), ("test", test_count)):
        directory = problem / "data" / mode
        directory.mkdir(parents=True)
        with (directory / f"{mode}_instance_data_sol_500.pkl").open("wb") as output:
            pickle.dump(_dataset(count), output)
    return root


def test_evaluation_does_not_modify_shared_gpt(tmp_path):
    from utils.heuristic_evaluator import HeuristicEvaluator

    root = _root(tmp_path)
    shared = root / "problems" / "tsp_gls" / "gpt.py"
    before = shared.read_bytes()
    result = HeuristicEvaluator(root, tmp_path / "out", workers=2).evaluate(
        HEURISTIC, 500, "val"
    )
    assert result.status == "ok"
    assert shared.read_bytes() == before
    assert Path(result.stdout_path).read_text().strip().endswith("1.25")


def test_missing_test_instances_fail_preflight(tmp_path):
    from utils.heuristic_evaluator import HeuristicEvaluator

    evaluator = HeuristicEvaluator(
        _root(tmp_path, test_count=99), tmp_path / "out", test_count=100
    )
    with pytest.raises(ValueError, match="requires 100"):
        evaluator.preflight([500], include_test=True)


def test_worker_failure_is_counted_in_mean():
    from problems.tsp_gls.eval import summarize_results

    summary = summarize_results([(10.0, 2.0), None])
    assert summary["failures"] == 1
    assert summary["mean_gap_pct"] == 500001.0
    assert summary["instances"][1] == {
        "objective": 1e6,
        "gap_pct": 1e6,
        "status": "failed",
    }


def test_solver_exception_returns_failed_marker(monkeypatch):
    from problems.tsp_gls import eval as eval_module

    monkeypatch.setattr(eval_module, "_UPDATE_EDGE_DISTANCE", None)
    params = (0, [[0.0]], [[0.0, 0.0]], 1.0, [0], 1)
    assert eval_module.solve(params) is None


def test_invalid_route_is_counted_as_failed_instance(monkeypatch):
    from problems.tsp_gls import eval as eval_module

    monkeypatch.setattr(eval_module, "check_valid_route", lambda route, size: False)
    result = eval_module.summarize_results(
        [eval_module.validated_result(10.0, 8.0, [0, 0], 2)]
    )
    assert result["failures"] == 1
    assert result["mean_gap_pct"] == 1e6


def test_randomized_heuristic_is_rejected_before_subprocess(tmp_path):
    from utils.heuristic_evaluator import HeuristicEvaluator

    root = _root(tmp_path)
    randomized = (
        "import random\n"
        "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
        "    return edge_distance * random.random()\n"
    )
    result = HeuristicEvaluator(root, tmp_path / "out").evaluate(
        randomized, 500, "val"
    )
    assert result.status == "failed"
    assert result.mean_gap_pct is None
    assert "deterministic" in result.error
    assert not (Path(result.stdout_path).parent / "result.json").exists()


def test_failed_evaluator_does_not_return_success_score(tmp_path):
    from utils.heuristic_evaluator import HeuristicEvaluator

    root = _root(tmp_path)
    (root / "problems" / "tsp_gls" / "eval.py").write_text(
        "import sys; print('broken'); raise SystemExit(2)\n"
    )
    result = HeuristicEvaluator(root, tmp_path / "out").evaluate(
        HEURISTIC, 500, "val"
    )
    assert result.status == "failed"
    assert result.mean_gap_pct is None
    assert result.error


def test_process_timeout_reaps_workers(tmp_path):
    from utils.heuristic_evaluator import HeuristicEvaluator

    root = _root(tmp_path)
    process = Mock(pid=12345, returncode=None)
    process.communicate.side_effect = [
        subprocess.TimeoutExpired("eval", 0.1),
        ("terminated", None),
    ]
    with (
        patch("utils.heuristic_evaluator.subprocess.Popen", return_value=process),
        patch("utils.heuristic_evaluator.os.killpg") as killpg,
    ):
        result = HeuristicEvaluator(
            root, tmp_path / "out", evaluation_timeout=0.1
        ).evaluate(HEURISTIC, 500, "val")
    assert result.status == "failed"
    assert "timed out" in result.error
    killpg.assert_called()
    assert process.communicate.call_count == 2


def test_configured_count_matches_report(tmp_path):
    from utils.heuristic_evaluator import HeuristicEvaluator

    root = _root(tmp_path, test_count=128)
    result = HeuristicEvaluator(
        root, tmp_path / "out", test_count=128, workers=32
    ).evaluate(HEURISTIC, 500, "test")
    assert result.status == "ok"
    assert len(result.indices) == 128
    payload = json.loads(
        (Path(result.stdout_path).parent / "result.json").read_text()
    )
    assert payload["count"] == 128
    assert payload["solver_settings"]["workers"] == 8


def test_result_hash_and_indices_match_request(tmp_path):
    from utils.heuristic_evaluator import HeuristicEvaluator

    root = _root(tmp_path, val_count=20)
    evaluator = HeuristicEvaluator(root, tmp_path / "out", seed=42)
    manifest = evaluator.preflight([500], include_test=False)[500]["val"]
    result = evaluator.evaluate(HEURISTIC, 500, "val")
    assert result.status == "ok"
    assert result.indices == manifest["indices"]
    assert result.dataset_sha256 == manifest["dataset_sha256"]
    assert result.code_sha256 == hashlib.sha256(HEURISTIC.encode()).hexdigest()


def test_malformed_result_is_failed(tmp_path):
    from utils.heuristic_evaluator import HeuristicEvaluator

    root = _root(tmp_path)
    (root / "problems" / "tsp_gls" / "eval.py").write_text(
        "import sys; from pathlib import Path; "
        "Path(sys.argv[sys.argv.index('--result-path') + 1]).write_text('{}')\n"
    )
    result = HeuristicEvaluator(root, tmp_path / "out").evaluate(
        HEURISTIC, 500, "val"
    )
    assert result.status == "failed"
    assert result.mean_gap_pct is None
