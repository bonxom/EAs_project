# Controller Zero-Shot Deployment Test on TSP-500/1000 — Spec and Plan

Date: 2026-11-21. Branch: `cross-level-adaptive-foresight`.

Builds on `docs/superpowers/specs/2026-11-21-cross-level-global-controller-design.md`
and on the checkpoint-driven deployment flow from the `moh-deployment` branch
(`MoH/deploy.py`, `MoH/inference.py`, `MoH/utils/deployment_report.py`,
`MoH/cfg/deployment.yaml`).

## Spec

### Intended behavior

After the three controller arms finish training
(`dual_hifo_independent`, `dual_hifo_global_rule`, `dual_hifo_global_bandit`),
evaluate each arm's **frozen checkpoint** zero-shot on the unseen TSP-500 and
TSP-1000 test splits, exactly as `moh-deployment` does:

- fresh target populations for 500 and 1000, seeded from the checkpoint and its
  stored training elites;
- 10 inference iterations, 8 validation instances, 100 independent test
  instances, 8 workers, one seed per arm;
- the validation winner per target is frozen and evaluated on test exactly
  once; test scores never feed generation, ranking, or selection;
- per-target failure is recorded and does not crash the remaining target.

Results are aggregated per arm (mean ± std across seeds of the TSP-500,
TSP-1000 and average test gap) so the three arms can be compared on the same
table as the full-budget deployment report.

### Scope

- `MoH/run_controller_experiment.sh`: add `--stage all|train|deploy`
  (default `all`), `--target-sizes` (default `[500,1000]`) and `--deploy-dir`
  (default `<out-dir>/deployment`). `--stage train` keeps today's behavior.
- New `MoH/tools/controller_deployment_report.py`: offline aggregation of
  `report.json` files into Markdown/JSON.
- New offline tests: `MoH/tests/test_controller_deployment_report.py` and
  `MoH/tests/test_controller_experiment_script.py` (fake `uv`, no network).
- `MoH/DUAL_HIFO.md`: document the deployment stage and report.

Out of scope: changing `deploy.py`, `inference.py`, evaluation semantics,
LLM clients, the controller itself, or running live LLM/experiments in tests.

### Constraints

- Tests never call an LLM or execute generated code; `deploy.py` is faked at the
  process boundary.
- The runner never deletes artifacts: completed runs and checkpoints are
  skipped, incomplete directories are reported for inspection.
- Bash 3.2 compatible (macOS `/bin/bash`): non-empty arrays on all paths that
  expand an array under `set -u`.
- All checks remain reproducible with seeds.

### Acceptance criteria

1. `--stage train` performs the same 3-arm training calls as before (same Hydra
   overrides, equal provider budget, skip rules) and never runs `deploy.py`.
2. `--stage deploy|all` invokes `deploy.py` once per `(mode, seed)` checkpoint
   with `checkpoint=<train-run>/checkpoint`, `target_sizes=[500,1000]`,
   `inference_iterations=10`, `val_n_instances=8`, `test_n_instances=100`,
   `workers=8`, the arm seed, the configured base URL, and a run directory under
   the deploy directory.
3. Completed deployments (`report.json` non-empty) are skipped on rerun;
   missing checkpoints and incomplete run directories are warned about without
   failing the campaign or deleting anything.
4. `controller_deployment_report.py`:
   - groups run directories by `<mode>_seed<N>`;
   - reports per-run and per-mode mean/std for TSP-500, TSP-1000 and the average
     test gap, with `n` = runs contributing;
   - never fabricates an average when one target is missing or failed;
   - isolates invalid JSON / missing files without crashing other runs;
   - writes Markdown and JSON; exit code 0 when at least one run is readable.
5. `uv run pytest -q` passes (including the new tests) and
   `uv run ruff check .` has no new violations versus the 50 pre-existing ones.

## Plan

1. Write this spec/plan (done).
2. RED: add `test_controller_deployment_report.py` and
   `test_controller_experiment_script.py`; run them and confirm the expected
   failures (missing module; `Unknown argument: --stage`).
3. Implement `tools/controller_deployment_report.py`.
4. Extend `run_controller_experiment.sh` with the deployment stage and report
   calls.
5. Update `DUAL_HIFO.md`.
6. GREEN: run the new tests, `uv run pytest -q`, `uv run ruff check .`, and a
   manual offline smoke of the script with a fake `uv` shim.
