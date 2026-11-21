# R-MoH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Reflective & Uncertainty-Guided Meta-Optimization of Heuristics (R-MoH) via three targeted hooks into the existing MoH architecture, preserving zero breaking changes for baseline comparison via `cfg.rmoh.enabled`.

**Architecture:** Hook 1 instruments the inner loop evaluation pipeline to capture a `TrajectoryDiagnostics` telemetry packet and a single-shot LLM reflection summary per optimizer. Hook 2 replaces greedy culling in the outer loop with a genealogy-tracking `MetaPopulation` governed by the `Meta-UIQ` multi-armed bandit metric and elitist preservation. Hook 3 injects diagnostic-guided repair directives and reflection summaries into the outer self-invocation meta-prompt.

**Tech Stack:** Python 3.12, uv, Hydra/OmegaConf, NumPy, existing `BaseClient` / `FakeLLM`, `GeneratedImprover` subprocess sandbox, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-17-rmoh-design.md`

---

## Global Constraints

- Python 3.12.
- Use `uv`.
- All experiments must be reproducible with seeds.
- Unit tests must never call external LLM APIs (use `FakeLLM` mandatory for tests).
- Never execute generated code in the main process (use `GeneratedImprover` multiprocessing sandbox with timeouts).
- Failure of one generated candidate must not crash an experiment.
- Keep core abstractions provider-independent.
- Do not hardcode OpenAI-specific logic outside `llm/openai_client.py`.
- Preserve 100% backward compatibility: when `cfg.rmoh.enabled == False`, the runner functions identically to baseline MoH.
- Before completing each task: run `uv run ruff check .` and `uv run pytest -q`.

---

## Review Focus

1. **Feature flag isolation:** Running with `cfg.rmoh.enabled = False` must execute the exact baseline logic and produce identical outputs.
2. **Inner loop telemetry safety:** Child process evaluation crashes, timeouts, or invalid code must be captured by `TrajectoryCollector` without crashing the parent process or interrupting evaluation.
3. **Meta-UIQ exploration-exploitation stability:** Scores must be properly normalized across varying subtask scales so UCB bounds remain numerically stable without division by zero or NaN.
4. **Token economy:** Reflection generation must be called at most once per optimizer evaluation (not per inner generation), strictly adhering to the 60-word constraint.
5. **Elitism preservation:** Culling by lowest Meta-UIQ must never delete the optimizer with the highest absolute historical utility.

---

## File Structure

| File | Responsibility |
|---|---|
| `MoH/cfg/config.yaml` | Add default `rmoh: {enabled: false, alpha: 0.6, c: 0.15, reflection: true, summary_max_words: 60}` |
| `MoH/utils/diagnostics.py` | `TrajectoryDiagnostics` dataclass and `TrajectoryCollector` helper for inner-loop telemetry |
| `MoH/utils/meta_population.py` | `OptimizerIndividual` and `MetaPopulation` managing genealogy, Min-Max normalization, Meta-UIQ scoring, selection, and elitist culling |
| `MoH/utils/diagnostic_prompts.py` | Rule-based directive generator `build_diagnostic_directive` and reflection prompt formatter |
| `MoH/moh.py` | Integrate the 3 hooks into `MoH` methods gated by `self.cfg.rmoh.enabled` |
| `MoH/tests/test_diagnostics.py` | Unit tests for Trajectory Diagnostics collection, slope, valid_rate, and reflection summary |
| `MoH/tests/test_meta_population.py` | Unit tests for Meta-UIQ formula, Min-Max normalization, genealogy tracking, and elitist culling |
| `MoH/tests/test_diagnostic_prompts.py` | Unit tests for dynamic directive injection and prompt formatting |
| `MoH/tests/test_rmoh_integration.py` | End-to-end integration test comparing baseline mode vs. R-MoH mode using `FakeLLM` |

---

## Tasks

### Task 1: Configuration & Trajectory Diagnostics Module (Hook 1)

**Files:**
- Modify: `MoH/cfg/config.yaml`
- Create: `MoH/utils/diagnostics.py`
- Create: `MoH/tests/test_diagnostics.py`

- [x] 1.1: Add `rmoh` section to `MoH/cfg/config.yaml`.
- [x] 1.2: Implement `TrajectoryDiagnostics` dataclass and `TrajectoryCollector` in `MoH/utils/diagnostics.py`.
- [x] 1.3: Implement metrics calculation in `TrajectoryCollector.finalize()`:
  - `valid_rate = n_valid / n_total`
  - `improvement_slope = (best_score_last - best_score_first) / max(n_evals - 1, 1)`
  - `stagnation_steps = consecutive non-improving evaluations at the end`
  - `diversity_score = std(population_scores)`
  - `failure_modes = list of distinct error types`
- [x] 1.4: Implement `generate_reflection_summary(llm, optimizer_code, best_heuristics, worst_heuristics, max_words=60)` in `MoH/utils/diagnostics.py`.
- [x] 1.5: Write unit tests in `MoH/tests/test_diagnostics.py` covering:
  - Telemetry collection on valid, invalid, and timed-out heuristic evaluations.
  - Correct slope, stagnation, and diversity calculations.
  - Reflection summary generation with `FakeLLM`.
- [x] 1.6: Run `uv run pytest -q MoH/tests/test_diagnostics.py` and `uv run ruff check .`.

---

### Task 2: Meta-UIQ Outer Population Manager (Hook 2)

**Files:**
- Create: `MoH/utils/meta_population.py`
- Create: `MoH/tests/test_meta_population.py`

- [x] 2.1: Implement `OptimizerIndividual` in `MoH/utils/meta_population.py` tracking:
  - `opt_id`, `code`, `thought`, `raw_utility`, `normalized_utility`, `diagnostics`, `selection_count`, `offspring_utilities`.
- [x] 2.2: Implement `compute_meta_uiq(total_outer_steps: int, alpha: float, c: float) -> float`:
  - When no offspring: expected offspring $Q = 0.8 \cdot \text{utility} + 0.2 \cdot \rho_{\text{valid}}$.
  - When offspring exist: expected offspring $Q = \text{mean}(\text{offspring\_utilities})$.
  - Exploitation term: $\alpha \cdot \text{utility} + (1 - \alpha) \cdot Q$.
  - Exploration term: $c \cdot \sqrt{\ln(\max(T, 1) + 1) / (n + 1)}$.
- [x] 2.3: Implement `MetaPopulation` class:
  - `normalize_utilities()`: Min-Max normalization for minimization objectives into $[0, 1]$ where higher is better.
  - `select_parent(total_outer_steps: int) -> OptimizerIndividual`: Probabilistic / Top-K selection based on Meta-UIQ; increments `parent.selection_count`.
  - `add_offspring(parent_id: str, individual: OptimizerIndividual)`: Appends offspring utility to parent's record.
  - `cull()`: Removes individual with lowest Meta-UIQ, strictly preserving the global elitist individual with highest raw utility.
- [x] 2.4: Write unit tests in `MoH/tests/test_meta_population.py` covering:
  - Normalization with identical scores, single candidate, and extreme ranges.
  - Meta-UIQ calculation under varying selection counts and offspring utilities.
  - Selection probability ranking and elitist culling preservation.
- [x] 2.5: Run `uv run pytest -q MoH/tests/test_meta_population.py` and `uv run ruff check .`.

---

### Task 3: Diagnostic-Guided Self-Invocation Prompt Builder (Hook 3)

**Files:**
- Create: `MoH/utils/diagnostic_prompts.py`
- Create: `MoH/tests/test_diagnostic_prompts.py`

- [x] 3.1: Implement `build_diagnostic_directive(diag: TrajectoryDiagnostics) -> str`:
  - Injects low-validity directive when `valid_rate <= 0.70`.
  - Injects stagnation directive when `stagnation_steps >= 3 or abs(improvement_slope) <= 1e-4`.
  - Injects diversity directive when `diversity_score <= 1e-3`.
- [x] 3.2: Implement `format_reflective_meta_prompt(base_prompt: str, parent: OptimizerIndividual) -> str`:
  - Prepends diagnostic directives and `parent.diagnostics.reflection_summary` cleanly before the base metaheuristic generation instructions.
  - Preserves standard function signatures and output constraints.
- [x] 3.3: Write unit tests in `MoH/tests/test_diagnostic_prompts.py` validating:
  - Directives triggered under respective threshold conditions.
  - Graceful fallback when diagnostics or reflection summary are empty.
  - Integrity of prompt formatting without syntax corruption.
- [x] 3.4: Run `uv run pytest -q MoH/tests/test_diagnostic_prompts.py` and `uv run ruff check .`.

---

### Task 4: MoH Orchestrator Integration & Dual-Mode Toggle

**Files:**
- Modify: `MoH/moh.py`
- Create: `MoH/tests/test_rmoh_integration.py`

- [x] 4.1: Integrate `TrajectoryCollector` into `MoH._make_subtask_utility` and `MoH.get_improver`:
  - When `self.cfg.rmoh.enabled == True`, attach collector to record heuristic evaluations and compute diagnostics + reflection.
  - When `self.cfg.rmoh.enabled == False`, maintain existing raw evaluation and return values.
- [x] 4.2: Integrate `MetaPopulation` into `MoH`:
  - Initialize `self.rmoh_pop = MetaPopulation(size=cfg.pop_size, alpha=cfg.rmoh.alpha, c=cfg.rmoh.c)` if enabled.
  - Hook parent selection and offspring feedback in `try_improvement()`.
  - Hook Meta-UIQ culling with elitist preservation.
- [x] 4.3: Integrate `format_reflective_meta_prompt` into `MoH.try_improvement`:
  - If R-MoH enabled, dynamically build prompt using parent's diagnostic directives and reflection.
  - If disabled, pass unchanged `self.meta_prompts`.
- [x] 4.4: Write integration test in `MoH/tests/test_rmoh_integration.py`:
  - Run 2 iterations with `FakeLLM` with `rmoh.enabled: false` (verifies baseline unchanged).
  - Run 2 iterations with `FakeLLM` with `rmoh.enabled: true` (verifies all 3 hooks trigger, logs update, and no crash occurs).
- [x] 4.5: Run full test suite: `uv run pytest -q` and `uv run ruff check .`.
