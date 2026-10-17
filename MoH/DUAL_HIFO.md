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
- `dual_hifo_cross_level`: both levels, with past inner summaries in outer prompts.

## Scientific limitations

- Foresight policy and credit equations are project-defined v1 policies, not copied verbatim from the HiFo paper.
- AST structural duplicates are **not** semantic diversity; semantic diversity is intentionally `null`.
- Insight distillation is deterministic AST-based, with no auxiliary provider calls. Full LLM-based abstract distillation is separate future work; do not describe this v1 extractor as LLM insight distillation.
- Inner feedback uses validation evaluation results and minimizes gap; failures with the sentinel 1e6 are not treated as improvements.
- Logging: run-local `dual_hifo_events.jsonl` tracks inner evaluations and outer selections. Existing evaluator, population and checkpoint semantics are unchanged.
- CVRP is not integrated: `moh-deployment` currently has TSP-GLS tasks; add CVRP independently after verifying its evaluator and score direction.
- The inherited MoH implementation's parent service, optimizer acceptance, and deployment semantics still require their own audit; these patches are not evidence of full MoH reproduction parity.
