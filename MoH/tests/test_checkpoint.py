import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from deployment_fakes import HEURISTIC, OPTIMIZER, training_population
from omegaconf import OmegaConf


def _write_legacy_run(run_dir: Path, accepted_rows, sizes=(20, 50)) -> None:
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "pop" / "improver").mkdir(parents=True)
    (run_dir / "pop" / "subtask").mkdir(parents=True)
    (run_dir / "code" / "improver").mkdir(parents=True)
    with (run_dir / "logs" / "meta_utility.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["iteration", "meta_val_utility", "accepted", "timestamp"])
        writer.writerows(accepted_rows)
    seed = {"meta-optimizer": [{"idea": "seed", "best_sol": OPTIMIZER, "utility": 2.0}]}
    (run_dir / "pop" / "improver" / "pre.json").write_text(json.dumps(seed))
    snapshot = {
        f"tsp_gls-{size}": [
            {"idea": f"elite-{size}", "best_sol": HEURISTIC, "utility": size / 1000}
        ]
        for size in sizes
    }
    (run_dir / "pop" / "subtask" / "iter_1.json").write_text(json.dumps(snapshot))


def test_checkpoint_keeps_elites_and_checks_hashes(tmp_path):
    from utils.checkpoint import export_checkpoint, load_checkpoint

    destination = tmp_path / "checkpoint"
    export_checkpoint(
        destination,
        OPTIMIZER,
        training_population(),
        {"training_sizes": [20, 50, 100, 200], "seed": 42},
    )
    checkpoint = load_checkpoint(destination)
    assert checkpoint.code == OPTIMIZER
    assert {elite.source_size for elite in checkpoint.elites} == {20, 50, 100, 200}
    assert not hasattr(checkpoint.elites[0], "training_utility")
    (destination / "meta_optimizer.py").write_text(OPTIMIZER + "\n# tampered")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(destination)


def test_export_refuses_existing_destination(tmp_path):
    from utils.checkpoint import export_checkpoint

    destination = tmp_path / "checkpoint"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError):
        export_checkpoint(
            destination,
            OPTIMIZER,
            training_population(),
            {"training_sizes": [20, 50, 100, 200]},
        )
    assert marker.read_text() == "untouched"


def test_elite_hash_tampering_is_rejected(tmp_path):
    from utils.checkpoint import export_checkpoint, load_checkpoint

    destination = export_checkpoint(
        tmp_path / "checkpoint",
        OPTIMIZER,
        training_population((20,)),
        {"training_sizes": [20]},
    )
    elite_path = destination / "elites" / "tsp20.json"
    elite = json.loads(elite_path.read_text())
    elite["elites"][0]["code"] += "\n# changed"
    elite_path.write_text(json.dumps(elite))
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(destination)


def test_two_size_run_exports_only_two_elite_files(tmp_path):
    from utils.checkpoint import export_checkpoint

    destination = export_checkpoint(
        tmp_path / "checkpoint",
        OPTIMIZER,
        training_population((20, 50)),
        {"training_sizes": [20, 50]},
    )
    assert sorted(path.name for path in (destination / "elites").iterdir()) == [
        "tsp20.json",
        "tsp50.json",
    ]


def test_checkpoint_metadata_excludes_credentials(tmp_path):
    from utils.checkpoint import export_checkpoint, redacted_config

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "api_key": "top-secret",
            "nested": {"password": "also-secret", "count": "${seed}"},
        }
    )
    destination = export_checkpoint(
        tmp_path / "checkpoint",
        OPTIMIZER,
        training_population((20,)),
        {"training_sizes": [20], "config": redacted_config(cfg)},
    )
    metadata_text = (destination / "metadata.json").read_text()
    metadata = json.loads(metadata_text)
    assert "top-secret" not in metadata_text
    assert "also-secret" not in metadata_text
    assert metadata["config"] == {"seed": 42, "nested": {"count": 42}}


def test_metadata_cannot_reference_parent_directory(tmp_path):
    from utils.checkpoint import export_checkpoint, load_checkpoint

    destination = export_checkpoint(
        tmp_path / "checkpoint",
        OPTIMIZER,
        training_population((20,)),
        {"training_sizes": [20]},
    )
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["optimizer"]["path"] = "../outside.py"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="relative|path"):
        load_checkpoint(destination)


def test_last_rejected_iteration_exports_previous_active_code(tmp_path):
    import moh
    from utils.checkpoint import export_run_checkpoint, load_checkpoint

    run_dir = tmp_path / "run"
    _write_legacy_run(
        run_dir,
        [(0, 1.0, True, "now"), (1, 2.0, False, "later")],
    )
    accepted = OPTIMIZER.replace("responses =", "# accepted\n    responses =")
    (run_dir / "code" / "improver" / "iter_0.py").write_text(accepted)
    (run_dir / "code" / "improver" / "candidate_noise.py").write_text("raise SystemExit")
    checkpoint = load_checkpoint(
        export_run_checkpoint(run_dir, tmp_path / "checkpoint")
    )
    assert checkpoint.code == accepted

    optimizer = moh.MoH.__new__(moh.MoH)
    optimizer.get_seed = Mock()
    optimizer.iteration = 1
    optimizer.max_eval_calls = None
    optimizer._total_eval_calls = 0
    optimizer.improver_str = accepted
    optimizer.meta_utility_val = 1.0
    optimizer.subtask_pop = training_population((20, 50))
    optimizer.subtask_pop.policy_metadata = Mock(return_value={"version": "fixture"})
    optimizer.improver_pop = Mock()
    optimizer.try_improvement = Mock(return_value=(False, None, Mock()))
    optimizer.run_logger = Mock(output_dir=str(tmp_path / "training"))
    optimizer.cfg = OmegaConf.create(
        {"problem": {"problem_size": [20, 50]}, "seed": 7, "optimizer_timeout": 10}
    )
    with patch("moh.export_checkpoint") as export:
        optimizer.run_meta_optimizer()
    assert export.call_args.args[1] == accepted


def test_no_accepted_iteration_exports_seed(tmp_path):
    from utils.checkpoint import export_run_checkpoint, load_checkpoint

    run_dir = tmp_path / "run"
    _write_legacy_run(
        run_dir,
        [(0, 2.0, False, "now"), (1, 2.0, False, "later")],
    )
    checkpoint_dir = export_run_checkpoint(run_dir, tmp_path / "checkpoint")
    assert load_checkpoint(checkpoint_dir).code == OPTIMIZER


def test_legacy_conversion_recovers_seed_config_and_validation_provenance(tmp_path):
    from utils.checkpoint import export_run_checkpoint

    run_dir = tmp_path / "run"
    _write_legacy_run(run_dir, [(0, 2.0, False, "now")], sizes=(20,))
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text(
        "seed: 73\nproblem:\n  problem_size: [20]\napi_key: hidden\n"
    )
    dataset = tmp_path / "val_instance_data_sol_20.pkl"
    dataset.write_bytes(b"fixture dataset")
    (run_dir / "logs" / "eval_iter0_s0_e1_val.txt").write_text(
        f"[*] mode=val, data={dataset}, seed=73, n_instances=3\n"
        "[*] instance_indices=[5, 2, 9]\n"
    )
    destination = export_run_checkpoint(run_dir, tmp_path / "checkpoint")
    metadata_text = (destination / "metadata.json").read_text()
    metadata = json.loads(metadata_text)
    provenance = metadata["validation_provenance"]["20"]
    assert metadata["seed"] == 73
    assert metadata["config"]["seed"] == 73
    assert "hidden" not in metadata_text
    assert provenance["status"] == "known"
    assert provenance["indices"] == [5, 2, 9]
    assert provenance["dataset_sha256"] == hashlib.sha256(
        dataset.read_bytes()
    ).hexdigest()


def test_checkpoint_records_optimizer_digest(tmp_path):
    from utils.checkpoint import export_checkpoint

    destination = export_checkpoint(
        tmp_path / "checkpoint",
        OPTIMIZER,
        training_population((20,)),
        {"training_sizes": [20]},
    )
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["optimizer"]["sha256"] == hashlib.sha256(
        OPTIMIZER.encode()
    ).hexdigest()


@pytest.mark.parametrize('mixed', [False, True])
def test_run_conversion_carries_population_policy(tmp_path, mixed):
    from utils.checkpoint import export_run_checkpoint, load_checkpoint

    run = tmp_path / 'run'
    _write_legacy_run(run, [(0, 2, False, 'now')])
    path = run / 'pop/subtask/iter_1.json'
    snapshot = json.loads(path.read_text())
    policy = {'version': 'behavior-maxmin-log1p-v1', 'diversity_fraction': .3}
    for entries in snapshot.values():
        entries[0]['population_policy'] = dict(policy)
    if mixed:
        snapshot['tsp_gls-50'][0]['population_policy']['diversity_fraction'] = 0
    path.write_text(json.dumps(snapshot))
    if mixed:
        with pytest.raises(ValueError, match='polic'):
            export_run_checkpoint(run, tmp_path / 'checkpoint')
    else:
        checkpoint = load_checkpoint(export_run_checkpoint(run, tmp_path / 'checkpoint'))
        assert checkpoint.metadata['population_policy'] == policy
