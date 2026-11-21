# Cross-Level Global Controller — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-11-21-cross-level-global-controller-design.md`

**Baseline (2026-11-21):** `uv run pytest -q` → 128 passed, 10 subtests passed.
`uv run ruff check .` → 46 pre-existing violations, none of them in the files touched below.

## Steps

1. **Tests first** (should fail with ImportError / missing behaviour):
   - `MoH/tests/test_global_controller.py`: mode flags, config validation,
     reward, rule diagnosis, LinUCB determinism, synthetic C1/C2 learning,
     state save/load, navigator regime override, `LLMCallMeter`.
   - `MoH/tests/test_global_controller_integration.py`: real `DualHiFoMoH`
     with FakeLLM, `FakeGeneratedImprover`, fake eval. Covers the inner pass,
     rule escalation to outer, subtask-failure isolation, bandit decisions +
     state file, independent progress events.
   - `MoH/tests/test_controller_report.py`: calls-to-target / per-action stats.
2. **`MoH/utils/global_controller.py`** (new, provider-independent):
   `ACTIONS`, `GlobalControllerConfig`, `NavigatorSignals`, `Decision`,
   `compute_reward`, `RuleBasedPolicy`, `LinUCBPolicy`, `GlobalController`.
3. **`MoH/utils/dual_hifo.py`**: `MODES` gets a 4th field, plus
   `HiFoConfig.global_controller`, the navigator override, `LLMCallMeter`, and
   mixin changes (meter install, inner pass, global `try_improvement`, progress
   and decision events, elite quality).
4. **`MoH/dual_hifo.py`**: pop the `global_controller` sub-dict and pass it on
   with the seed.
5. **`MoH/cfg/dualhifo.yaml`**: list the new modes and set controller defaults.
6. **`MoH/tools/controller_report.py`**: aggregate events → Markdown/JSON.
7. **`MoH/run_controller_experiment.sh`**: 3 arms × seeds under an equal provider
   budget, then the report. Skips existing runs and never deletes anything.
8. **Docs**: add the new modes, protocol and limitations to `MoH/DUAL_HIFO.md`.
9. **Verify**: run the new tests, then `uv run ruff check <touched files>`,
   `uv run ruff check .` (no new violations), and `uv run pytest -q`.
