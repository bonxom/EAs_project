import json

import pytest
from tools.controller_deployment_report import (
    load_deployment,
    main,
    render_markdown,
    summarize,
)


def _write(run_dir, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(rows), encoding="utf-8")
    return run_dir


def _row(size, test_gap, val_gap, status="ok", test_n=100):
    return {
        "size": size,
        "status": status,
        "validation_gap_pct": val_gap,
        "test_gap_pct": test_gap,
        "val_n_instances": 8,
        "test_n_instances": test_n,
        "test_failures": 0,
        "error": None,
        "seed": 0,
    }


def test_groups_runs_by_mode_and_summarises_test_gaps(tmp_path):
    independent_0 = _write(
        tmp_path / "dual_hifo_independent_seed0",
        [_row(500, 1.0, 0.5), _row(1000, 2.0, 1.5)],
    )
    independent_1 = _write(
        tmp_path / "dual_hifo_independent_seed1",
        [_row(500, 3.0, 2.5), _row(1000, 4.0, 3.5)],
    )
    rule = _write(
        tmp_path / "dual_hifo_global_rule_seed0",
        [_row(500, 0.5, 0.4), _row(1000, 1.5, 1.4)],
    )

    summary = summarize(
        [load_deployment(path) for path in (independent_0, independent_1, rule)]
    )

    independent = summary["modes"]["dual_hifo_independent"]
    assert independent["run_count"] == 2
    assert independent["tsp500_test_gap_pct"]["mean"] == pytest.approx(2.0)
    assert independent["tsp500_test_gap_pct"]["std"] == pytest.approx(2**0.5)
    assert independent["tsp500_test_gap_pct"]["n"] == 2
    assert independent["tsp1000_test_gap_pct"]["mean"] == pytest.approx(3.0)
    assert independent["avg_test_gap_pct"]["mean"] == pytest.approx(2.5)

    rule_summary = summary["modes"]["dual_hifo_global_rule"]
    assert rule_summary["avg_test_gap_pct"]["mean"] == pytest.approx(1.0)
    assert rule_summary["tsp500_val_gap_pct"]["mean"] == pytest.approx(0.4)

    first = next(
        run
        for run in summary["runs"]
        if run["mode"] == "dual_hifo_independent" and run["seed"] == 0
    )
    assert first["status"] == "ok"
    assert first["tsp500_test_gap_pct"] == pytest.approx(1.0)
    assert first["avg_test_gap_pct"] == pytest.approx(1.5)
    assert first["test_n_instances"] == {500: 100, 1000: 100}


def test_partial_report_never_fabricates_average(tmp_path):
    run = _write(
        tmp_path / "dual_hifo_global_rule_seed0",
        [_row(500, 1.0, 0.5), _row(1000, None, 1.5, status="test_failed")],
    )

    result = load_deployment(run)

    assert result["status"] == "partial"
    assert result["tsp500_test_gap_pct"] == 1.0
    assert result["tsp1000_test_gap_pct"] is None
    assert result["avg_test_gap_pct"] is None

    summary = summarize([result])
    assert summary["modes"]["dual_hifo_global_rule"]["avg_test_gap_pct"]["n"] == 0
    assert summary["modes"]["dual_hifo_global_rule"]["avg_test_gap_pct"]["mean"] is None


def test_invalid_report_is_isolated(tmp_path):
    bad = tmp_path / "dual_hifo_global_bandit_seed0"
    bad.mkdir()
    (bad / "report.json").write_text("{not json", encoding="utf-8")
    good = _write(
        tmp_path / "dual_hifo_global_rule_seed0",
        [_row(500, 1.0, 0.5), _row(1000, 2.0, 1.5)],
    )

    summary = summarize([load_deployment(bad), load_deployment(good)])

    assert summary["runs"][0]["status"] == "invalid"
    assert summary["runs"][0]["error"]
    assert summary["modes"]["dual_hifo_global_rule"]["run_count"] == 1
    assert summary["modes"]["dual_hifo_global_bandit"]["run_count"] == 1


def test_markdown_lists_modes_and_gaps(tmp_path):
    run = _write(
        tmp_path / "dual_hifo_global_bandit_seed0",
        [_row(500, 1.25, 1.0), _row(1000, 2.5, 2.0)],
    )

    text = render_markdown(summarize([load_deployment(run)]))

    assert "dual_hifo_global_bandit" in text
    assert "TSP-500" in text
    assert "TSP-1000" in text
    assert "1.250000" in text
    assert "1.875000" in text


def test_cli_writes_markdown_and_json(tmp_path):
    run = _write(
        tmp_path / "dual_hifo_global_rule_seed0",
        [_row(500, 1.0, 0.5), _row(1000, 2.0, 1.5)],
    )
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json.out"

    assert (
        main(
            [
                str(run),
                "--md-out",
                str(markdown_path),
                "--json-out",
                str(json_path),
            ]
        )
        == 0
    )

    text = markdown_path.read_text(encoding="utf-8")
    assert "dual_hifo_global_rule" in text
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["modes"]["dual_hifo_global_rule"]["tsp500_test_gap_pct"][
        "mean"
    ] == pytest.approx(1.0)
    assert payload["runs"][0]["run"].endswith("dual_hifo_global_rule_seed0")
