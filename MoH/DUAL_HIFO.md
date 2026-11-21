# Dual-HiFo built on `moh-deployment`

This implementation **starts from** the `moh-deployment` `MoH/` training implementation, **not** the repository's separate `src/moh/` Full-MoH prototype. It preserves `improve_algorithm(population, utility, language_model, function_format, task)` and its child-process execution.

## Integration

1. From the extracted patch root, run `./apply_dualhifo.sh /absolute/path/to/EAs_project`. This creates a new `dualhifo` branch from the verified `moh-deployment` commit and copies the supplied entrypoint.
2. `MoH/main.py` in this package already has the following conditional integration:

```python
    from moh import MoH
    if cfg.get("dual_hifo") is not None:
        from dual_hifo import DualHiFoMoH
        optimizer = DualHiFoMoH(cfg, ROOT_DIR, heu_llm=heu_llm, meta_llm=meta_llm)
    else:
        optimizer = MoH(cfg, ROOT_DIR, heu_llm=heu_llm, meta_llm=meta_llm)
```

3. From `MoH/`, run `uv run python main.py --config-name=dualhifo dual_hifo.mode=dual_hifo_independent` (live LLM calls incur provider costs).
4. For offline validation: `uv run pytest -q tests/test_dual_hifo.py tests/test_dual_hifo_integration.py`.

The local 11-test controller/hook suite passed in a Python 3.13 environment. Full MoH/ Hydra integration, ruff, and real TSP-GLS training have not been executed here; verify those in the repository's Python 3.12 environment before treating this as research-ready.

## Experiment modes

- `full_moh`: original MoH evolution with zero HiFo prompt modification.
- `inner_hifo`: only candidate heuristic prompts and observations.
- `outer_hifo`: only meta-optimizer prompts and observations.
- `dual_hifo_independent`: both levels, no inner-to-outer history injection.
- `dual_hifo_cross_level`: both levels active with synthesized inner evidence (top validated AST mechanisms across subtasks, multi-task stagnation/progress regimes, and cross-task reuse directives) injected into outer meta-optimizer prompts.
- `dual_hifo_global_rule`: both navigators active; a rule-based cross-level global controller reads both levels' states and chooses per iteration between cheap inner passes (`INNER_EXPLORE`, `INNER_EXPLOIT`) and outer optimizer evolution (`OUTER_EXPLORE`, `OUTER_EXPLOIT`). No cross-level prompt text is injected, isolating the control decision.
- `dual_hifo_global_bandit`: same coordinated action space as `dual_hifo_global_rule`, controlled by a disjoint LinUCB contextual bandit rewarded by cost-normalised elite-quality improvement per LLM call. Model state is persisted to `global_controller_state.json` and supports optional cross-run warm start (`dual_hifo.global_controller.warm_start_path`).

## Cross-level global controller protocol

Compare three arms under an **equal shared LLM-call budget** (`provider_budget.max_requests`) with high `n_iterations` so budget terminates every run (do not compare at fixed `n_iterations`, because inner passes cost ~10× fewer LLM calls than outer steps):

1. `dual_hifo_independent` (reference: two independent navigators, every step is an outer step)
2. `dual_hifo_global_rule` (hand-designed C1 local-optimum vs C2 weak-optimizer diagnosis)
3. `dual_hifo_global_bandit` (online LinUCB learning from cost-normalised feedback)

Run via `MoH/run_controller_experiment.sh` and aggregate `dual_hifo_events.jsonl` (`progress` and `decision` events) using:

```sh
cd MoH && uv run python -m tools.controller_report outputs/controller_exp/*/ --md-out report.md --json-out report.json
```

Reported metrics: calls-to-target (first cumulative LLM calls reaching the reference arm's final elite quality), anytime mean elite quality over the common call budget, and per-action cost/success rate (including post-stagnation success rate for C1 vs C2 diagnosis).

### Zero-shot deployment test on TSP-500/1000

The same script can run the `moh-deployment` checkpoint-driven test for every
trained arm. `--stage all` (default) trains and then deploys; `--stage train`
or `--stage deploy` runs only one phase:

```sh
cd MoH
bash run_controller_experiment.sh --stage all --seeds "0 1 2"
```

Both stages require `LOCAL_LLM_API_KEY` with the default local profile (or
`OPENAI_API_KEY` with `--llm-profile openai`); the runner rejects a missing key
before creating any run directory.

For each `(mode, seed)` checkpoint under `<out-dir>/<mode>_seed<N>/checkpoint`,
the deployment stage calls `deploy.py` with `target_sizes=[500,1000]`,
`inference_iterations=10`, `val_n_instances=8`, `test_n_instances=100`,
`workers=8`, `evaluate_test=true` and an output directory under
`<out-dir>/deployment/<mode>_seed<N>/`. Completed deployments
(`report.json`), incomplete directories and missing checkpoints are skipped
with a message; the runner never deletes artifacts. Validation selects the
frozen heuristic per target and test is evaluated exactly once, as in the
`moh-deployment` flow.

Per-run tables and the arm comparison are written to
`<out-dir>/deployment/deployment_tables.tsv` and
`<out-dir>/deployment/deployment_report.{md,json}`. To aggregate existing
deployment directories manually:

```sh
cd MoH && uv run python -m tools.controller_deployment_report \
  outputs/controller_exp/deployment/*/ \
  --md-out deployment_report.md --json-out deployment_report.json
```

The report groups `report.json` rows into TSP-500, TSP-1000 and average test
gaps (mean ± std across seeds). An average is only reported when both targets
have a finite test result; invalid or partial runs are listed and never counted
as complete. Deployment calls the configured external LLM and can incur
provider cost.

## Stage 1 local audit (2026-10-17)

Applied on the `dualhifo` branch after the patch was verified with offline tests:

- **F1** `generate_seed_algorithm` is bounded by `seed_max_attempts` (default 15) and at most 5 directions per round; raises `RuntimeError` if zero valid seeds are found.
- **F2** `evaluate_heuristic` short-circuits with sentinel `1e6` when `max_eval_calls` is exhausted (no subprocess spawn).
- **F3** `isolate_base_solutions` (config flag; `true` in `dualhifo.yaml`) redirects `solution_*.json` read/write to a run-local `base_solutions/` directory with provenance logging; shared files are never deleted.
- **F4** `_run_code` writes a run-local heuristic file under `logs/` and passes `--heuristic-path` to `eval.py` (no shared `gpt.py` overwrite); uses `sys.executable`; reaps the child after `kill()`.
- **E3** Inner observation moved from `_make_subtask_utility` to a `evaluate_heuristic` override so seed-phase evaluations are counted.
- **E5** `tests/test_moh_budget_isolation.py` adds a `MoH` vs `DualHiFoMoH(mode=full_moh)` parity regression (identical meta utility, improver string, populations, LLM call logs, CSV rows; no `dual_hifo_events.jsonl`).

## Scientific limitations

- Foresight policy and credit equations are project-defined v1 policies, not copied verbatim from the HiFo paper.
- AST structural duplicates are **not** semantic diversity; semantic diversity is intentionally `null`.
- Insight distillation is deterministic AST-based, with no auxiliary provider calls. Full LLM-based abstract distillation is separate future work; do not describe this v1 extractor as LLM insight distillation.
- Inner feedback uses validation evaluation results and minimizes gap; failures with the sentinel 1e6 are not treated as improvements.
- Logging: run-local `dual_hifo_events.jsonl` tracks inner evaluations, outer selections, per-iteration `progress` snapshots (with cumulative logical LLM calls and elite quality), and global controller `decision` events. Existing evaluator, population and checkpoint semantics are unchanged.
- Global controller limitations:
  - Bandit reward is myopic (immediate elite-quality gain per LLM call); delayed payoffs of a newly accepted optimizer are not backpropagated across steps.
  - Meta-utility drift: as inner passes improve `subtask_pop`, later candidate optimizers are scored on stronger starting populations than earlier ones; incumbent re-measured utility is logged in `inner_pass.incumbent_remeasured_utility` for post-hoc audit while preserving MoH acceptance semantics.
  - A single run performs ~10–40 global decisions, which is a small-sample regime for online contextual bandits; use multi-seed evaluation and optional `warm_start_path` when assessing learned policies.
- CVRP is not integrated: `moh-deployment` currently has TSP-GLS tasks; add CVRP independently after verifying its evaluator and score direction.
- The inherited MoH implementation's parent service, optimizer acceptance, and deployment semantics still require their own audit; these patches are not evidence of full MoH reproduction parity.
