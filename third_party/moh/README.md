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
