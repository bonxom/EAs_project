# MoH attribution and adaptation notes

Adapted from the read-only MoH checkout `/home/bonxom/Code/MoH`, revision
`e8c6154911d8182f2a53bcc17b11be0a063a53de` (MIT, copyright 2026 Yiding Shi).
The original MIT notice is retained in [LICENSE](LICENSE).

Sources adapted:

- `problems/tsp_gls/gls.py`: predecessor/successor 2-opt and relocation moves,
  nearest-neighbor traversal and alternating local search, adapted in
  `src/moh/problems/tsp_gls/local_search.py` and `tour.py`.
- `problems/tsp_gls/eval.py`: guided perturbation, edge penalties, top-five
  default edge selections and best-route resets, adapted in `solver.py`.
- `problems/tsp_gls/gpt.py`: `UPSTREAM_BASELINE_SOURCE` in `baselines.py`, with
  the missing `import numpy as np` added. Identity and penalty sources are
  local handwritten alternatives.

Intentional differences and retained details:

- NumPy/Python replaces Numba. Fixed iteration counts replace upstream's
  machine-dependent 20-second stopping rule; worker deadlines treat timeout
  as evaluation failure. No upstream evaluator or pickle reader is vendored.
- Distances and guidance are validated; copies isolate callback mutations.
  Guidance accepts a nonzero diagonal, as the upstream baseline produces one.
  True costs always use original distances. Best/reset routes are copied to
  avoid upstream's mutable aliasing of current and best routes.
- Stable neighbor ordering resolves ties reproducibly. Each city's index is
  explicitly filtered from its neighbor list: removing the first sorted entry
  is incorrect when another city has zero distance. Moves also skip self.
- All-city traversal retains the upstream exclusion of city `n - 1` as a move
  origin. That city remains a neighbor and a guided per-city endpoint.
  Per-city traversal applies each increasingly improving move immediately;
  its returned delta is the sum of applied moves rather than upstream's last
  (most negative) delta. This changes reporting, not route traversal.
- Callback tours preserve upstream `route2tour` order (successor of zero first,
  zero last, no repeated endpoint). Returned tours are closed and start at zero.
- Gap selection uses row-major `np.argmax`; both directions are zeroed after
  selection. Diagonal selections increment the same entry twice, including
  repeated `(0, 0)` selection on an all-zero gap, preserving upstream behavior.
- Integer distance matrices are promoted before move arithmetic to prevent
  overflow. Original local-search comparisons retain upstream `np.isclose`
  suppression of numerically negligible improvements.

The seed-7 nine-city fixture in `tests/programs/test_gls_solver.py` was recorded
from this revision's `gls.py` with only Numba imports/decorators removed:
nearest-neighbor plus local search gives tour
`(0, 2, 3, 6, 5, 7, 1, 4, 8, 0)` and cost `3.026072316423462`.
The test embeds the recorded result and never imports the external checkout.

## Optimizer programs, prompts and orchestration

Additional adaptations from the same upstream revision:

- `problems/meta/seed_algorithm.py` → `src/moh/optimizers/seeds/basic.py`:
  retain random-parent selection, JSON directions, batch generation, evaluation
  and minimum-score return. Prompt wording is shortened and function formats
  are included explicitly; directions/messages are truncated to the worker's
  batch size before sending.
- `problems/meta/seed_algorithm_improved.py` → `multi_temperature.py`:
  retain temperatures `[0.7, 1.0]` and source-based evaluation caching.
  Correct the descending intermediate ranking (`reverse=True`) to ascending
  because the objective is minimize. `best_parent.py` is a local basic-seed
  variant selecting the best parent, rather than a third upstream seed.
- `utils/utils.py` → `src/moh/optimizers/helpers.py`: rewrite extraction as
  bounded-response parsing with explicit rejection of empty/ambiguous fences
  and preservation of list alignment. Seed modules import this path instead
  of the upstream global `utils` package. The parent reads bundled source
  files as text; generated code executes only in workers.
- `prompts/meta/desc.txt` and `prompts/tsp_gls/{desc,task,plan,size}.txt`:
  function contracts and optimization guidance are adapted and rewritten in
  `src/moh/prompts/{program_optimizer,gls_heuristic}.py` and seed messages.
  This is conceptual/contract adaptation, not a verbatim copy of all prompts.
- `moh.py`, `utils/population.py` and `utils/run_logger.py` inform the approved
  orchestration design, parent populations/snapshots and artifact lineage.
  The new outer/composition pipeline is still being integrated when these
  notes are written; its complete runtime behavior has not yet been verified.

Intentional orchestration/evaluation changes in the approved design:

- Optimizer `exec` moves out of the main process into a worker, including
  module-level statements. Structured JSON callbacks replace global LLM state
  and use of printed output as a score channel. Real invocation deadlines
  replace LLM-based iteration checking.
- The parent verifies returned sources against its evaluation ledger and
  compares fitness with its own score. Population transactions roll back on
  failed invocations, including partially evaluated multi-task candidates;
  work already consumed remains counted. Active optimizer and global best
  are separate records.
- Upstream test mode declares `n_tests=10` but samples only five indices in
  `problems/tsp_gls/eval.py`. Explicit, disjoint validation/test index lists
  replace this inconsistent sampling. Held-out scores do not enter search.
- Synthetic smoke references use exact Held–Karp for at most 12 cities;
  large local datasets must provide consistent reference tours and costs.
  NPZ loading uses `allow_pickle=False`; the separate trusted-pickle converter
  is explicit opt-in and can execute pickle code before validation.

These notes record algorithm and interface fidelity, not paper-result,
wall-clock-budget or dataset equivalence. No benchmark dataset was downloaded
or live LLM campaign run. Python/NumPy and fixed iterations cannot establish
performance equivalence to the original Numba evaluator. Worker deadlines and
process cleanup provide crash/timeout containment, not a hostile-code OS
sandbox. See [the Vietnamese reading guide](../../docs/moh-reproduction.md)
for ownership, scores, the upstream-to-new mapping and remaining limits.
