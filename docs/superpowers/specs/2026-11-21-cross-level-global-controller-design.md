# Cross-Level Global Controller for Dual-HiFo — Design

Date: 2026-11-21. Builds on `MoH/utils/dual_hifo.py` (Dual-HiFo v1).

## 1. Problem analysis

Dual-HiFo v1 has two Evolutionary Navigators (`HiFoController`), one per level.
Each picks EXPLORE / EXPLOIT / BALANCE from hand-set thresholds
(`stagnation_patience`, `progress_patience`, `duplicate_threshold`) and only
changes prompt text. Neither navigator reads the other's state, so when inner
search stalls the system cannot tell apart:

- **C1 — heuristic local optimum:** the optimizer is fine; the inner level just
  needs more exploration. Cheap fix: run the current optimizer again with an
  EXPLORE directive.
- **C2 — weak optimizer:** the heuristic-optimizer itself caps progress. The fix
  is outer search: generate a new optimizer strategy. This costs much more.

### 1.1 Findings in this codebase that shape the design

1. **The inner loop is not its own loop.** In `MoH`, inner search only happens
   inside `meta_utility()`, while each *candidate optimizer* is being scored. Each
   outer iteration runs one outer step. So with two navigators that only edit
   prompts, no decision can make the run use fewer LLM calls. The hypothesis
   would be untestable. **For a controller to save calls, it needs an action that
   costs less.** We therefore add an explicit *inner pass*: run the incumbent
   accepted optimizer once per subtask, with no meta-generation and no validator
   call.
2. **Costs are lopsided.** With the seed improvers, one inner pass costs about
   `N_subtasks × (1 + k)` heuristic-LLM calls (k = number of directions). One outer
   step costs the meta-improver calls plus, for each candidate optimizer, one
   validator call and a full inner run on every subtask. That is roughly 10× more.
   Any savings come from choosing C1-type actions when they are enough.
3. **Meta-utility drifts.** Subtask populations keep improving, so later
   candidates are scored on better populations than the incumbent was. This
   already happens in MoH. Inner passes make it stronger. We keep MoH's acceptance
   rule unchanged and log the incumbent's re-measured utility for analysis.
4. **Small sample size.** A run makes about 10–40 decisions. Any learned
   controller must use a low-dimensional context, cheap updates, and optional
   warm start across seeds/runs. A deep RL policy is not justified.
5. **Reward is myopic.** An accepted optimizer pays off later, but a bandit only
   credits the immediate improvement. This is accepted for v1 and documented as
   a limitation.

## 2. Scope

In scope:

- Two new Dual-HiFo modes: `dual_hifo_global_rule` and `dual_hifo_global_bandit`.
  Both turn on both navigators. The global controller reads both levels' states
  and picks one action per outer iteration.
- Action space (level × regime):
  `INNER_EXPLORE`, `INNER_EXPLOIT` (inner pass with the incumbent optimizer and
  the inner navigators' regime forced) and `OUTER_EXPLORE`, `OUTER_EXPLOIT`
  (standard MoH outer step with the outer navigator's regime forced).
- Rule-based global policy (hand-designed diagnosis) and a LinUCB contextual
  bandit.
- LLM call metering and uniform `progress` telemetry for every HiFo mode, so the
  three arms can be compared on the same metric.
- An offline report tool for anytime curves / calls-to-target, plus an
  experiment runner script.

Out of scope: full RL, LLM-based insight distillation, changing MoH acceptance
semantics, CVRP, running the live LLM campaign. Global modes do **not** inject
cross-level evidence text into outer prompts. Prompts stay identical to
`dual_hifo_independent` apart from the forced regime line, so the comparison
isolates the decision mechanism.

## 3. Experiment arms

| Arm | Mode | Decision maker | Step type |
|---|---|---|---|
| A | `dual_hifo_independent` (existing) | two local navigators | always outer |
| B | `dual_hifo_global_rule` | rule-based global controller | inner or outer |
| C | `dual_hifo_global_bandit` | LinUCB global controller | inner or outer |

**Protocol.** All arms get the same `provider_budget.max_requests`. Set
`n_iterations` high enough that the budget is what stops the run. Use ≥3 seeds per
arm and identical LLM / eval settings. Do not compare at equal `n_iterations`,
because a global arm's step is cheaper by design.

**Hypothesis H1.** B and C reach the final elite quality of A with fewer LLM
calls (calls-to-target(B,C) < calls-to-target(A)), without a worse final quality
at equal budget.

**Metrics** (all from `dual_hifo_events.jsonl`):

- Elite quality `Q` = size-weighted mean of the best validation gap per training
  subtask in `subtask_pop` (lower is better). It is the training output
  exported to the checkpoint.
- Cumulative LLM calls (logical prompts, heuristic + meta roles) and eval calls.
- Calls-to-target: first cumulative call count with `Q ≤ target`. Default target =
  mean final `Q` of arm A.
- Normalised anytime area (mean `Q` over the common call budget).
- Per-action usage, mean cost, and success rate (Q improved). Success rate after
  inner stagnation, compared between inner and outer actions, is the
  diagnostic for C1 vs C2.
- Secondary (existing tools): `meta_utility_val` and zero-shot deployment gap.

## 4. Design

### 4.1 State (cross-level context)

Built each decision from both navigators and the controller's own history:

| Feature | Source | Normalisation |
|---|---|---|
| recent inner improvement | relative drop of `Q` over last `window` decisions | `x/(x+0.01)` |
| recent outer improvement | relative drop of `meta_utility_val` over last `window` decisions | `x/(x+0.01)` |
| inner stagnation rounds | decisions since `Q` last improved | `n/(n+patience)` |
| outer stagnation rounds | outer navigator `stagnation_count` | `n/(n+patience)` |
| inner diversity | `1 − mean(inner exact_duplicate_rate)` | [0,1] |
| outer diversity | `1 − outer exact_duplicate_rate` | [0,1] |
| failed inner streak | consecutive inner actions without `Q` gain | `n/(n+retry_limit)` |
| action cost (per arm) | EMA of LLM calls of this action | `c/(c+cost_scale)`; unseen = 0.5 |
| action efficacy (per arm) | EMA of reward of this action | [0,1]; unseen = 0 |
| bias | constant | 1 |

The 7 shared features plus 2 per-arm features plus the bias give d = 10.

### 4.2 Reward

`reward = min(1, (max(0, Q_before − Q_after) / max(|Q_before|, 1e-12)) / max(cost,1) / efficiency_scale)`.
That is the relative quality gain per LLM call, scaled by `efficiency_scale`
(default 1e-3 = 0.1 % per call → reward 1). Missing `Q` gives reward 0.

### 4.3 Rule-based global policy (arm B)

Evaluated in order:

1. Last action improved `Q`, or an outer step accepted a new optimizer →
   `INNER_EXPLOIT` (reason `PROGRESS` / `NEW_OPTIMIZER`).
2. Failed inner streak ≥ `inner_retry_limit` → outer escalation:
   `OUTER_EXPLOIT` if the outer navigator has `progress_count ≥ progress_patience`,
   otherwise `OUTER_EXPLORE` (reason `INNER_EXPLORATION_EXHAUSTED`, diagnosis C2).
3. Otherwise → `INNER_EXPLORE` (reason `TRY_CHEAP_INNER_EXPLORATION`, diagnosis C1).

### 4.4 LinUCB policy (arm C)

Disjoint LinUCB: each action has `A_a = λI + Σxxᵀ` and `b_a = Σ r x`, and is scored
as `θ_aᵀx_a + α√(x_aᵀA_a⁻¹x_a)`. Ties are broken with a NumPy `Generator` seeded
from the run seed. The model state is written to `global_controller_state.json`
after every update. It can be loaded through `warm_start_path` for cross-seed
transfer.

### 4.5 Integration (`DualHiFoMoHMixin`)

- `HiFoController` gains `set_override(regime, reason)` / `clear_override()`. While
  set, the override replaces the local regime in `guidance()` only. `observe()`
  behaviour is unchanged.
- `try_improvement` in global modes: decide → run the inner pass or the outer
  step (with the override) → measure the LLM-call delta and `Q` → controller
  feedback → `decision` event. An inner action returns
  `(False, None, previous_algo)`, so `run_meta_optimizer` keeps the incumbent.
- Inner pass: for each subtask, `read_base_algorithm` → `get_improver(improver_str)`
  with the inner override → save the solution and append to `utility.csv` → persist
  the population. If one subtask fails, it is logged and skipped.
- `LLMCallMeter` wraps `heu_llm` / `meta_llm` in every non-`full_moh` mode. It is
  a transparent counter and forwards `budget`, `batch_size`, etc.
- Every non-`full_moh` mode writes a `progress` event after seed and after each
  iteration: `mode`, `llm_calls`, `heu_calls`, `meta_calls`, `eval_calls`,
  `elite_quality`, `meta_utility`, `action`.

### 4.6 Configuration

`dual_hifo.global_controller` sub-dict (validated `GlobalControllerConfig`):
`inner_retry_limit=2`, `patience=3`, `window=3`, `ucb_alpha=0.5`,
`ridge_lambda=1.0`, `cost_scale=100.0`, `efficiency_scale=1e-3`, `ema=0.3`,
`warm_start_path=null`. Seed = top-level `seed`.

## 5. Constraints

No generated code runs in the parent process (the inner pass reuses
`GeneratedImprover`). No real LLM in tests (FakeLLM). Failures in a single
candidate or subtask never crash the run. The controller module is
provider-independent. `full_moh` parity is preserved (no meter, no events).

## 6. Acceptance criteria

1. `HiFoConfig` accepts the two new modes. Both levels are enabled,
   `cross_level_enabled` is false, and `global_controller` is `rule` / `bandit`.
   Existing modes give `None`.
2. The rule policy gives `INNER_EXPLORE` with no history, `INNER_EXPLOIT` after a
   `Q` gain or an accepted optimizer, and `OUTER_EXPLORE` / `OUTER_EXPLOIT` once
   the failed inner streak reaches the limit.
3. Reward is in [0,1], 0 with no gain or with missing `Q`, and decreases as cost
   grows for equal gain.
4. LinUCB is deterministic under a seed. It learns a context-dependent best arm
   in a synthetic C1/C2 environment. State save/load reproduces its choices.
5. In a fake-LLM MoH run, an inner action makes no meta-generation call, keeps
   `improver_str`, adds heuristics, and logs a `decision` event with
   cost > 0. A rule-mode run escalates to an outer step whose meta prompt carries
   the forced regime.
6. Independent mode writes `progress` events with monotone `llm_calls` and makes
   no decisions. `full_moh` parity test still passes.
7. The report tool computes calls-to-target, final quality and per-action stats
   from event files and renders Markdown/JSON.
8. `uv run ruff check` is clean on new/modified files. `uv run pytest -q` passes.
