# Mini-MoH v0 implementation record

Implemented all eleven stages of [the plan](plans/2026-09-24-mini-moh-v0.md)
directly on `main`, as explicitly requested. Base commit: `5e96f6b`.

## Verification

- `uv run ruff check .`: passed.
- `uv run pytest -q`: 185 passed in 31.08 seconds after final fixes.
- `uv run python -m moh.main --mode tsp-demo`: both baselines succeeded.
- Two standalone smoke experiments succeeded with byte-identical event JSONL.
  Each retained 40 heuristic sources, four optimizer specifications, and 199 events.
  Winner: `o000001`; utility: `-3.6656793619332095`.
- The full suite reran the twice-run semantic smoke comparison after review fixes.
- No live API request was made. The real adapter was tested with injected offline
  transports and the installed SDK's exception/response interface.

Every stage began with failing contract tests and ended with a passing full suite
and Ruff check. Stages were committed separately. A corrected TSP test expectation
uses the independently verified tour `(0, 1, 3, 2, 0)` and length 6.

## Independent review

The reviewer inspected the complete change through `14c23c5` and identified three
issues. All were reproduced with failing regression tests and fixed in one pass:

1. Wrong-type JSON failure codes could raise `TypeError` in the parent. Explicit
   string validation now rejects them as protocol failures.
2. Unpaired Unicode surrogates in generated source could abort persistence or
   worker setup. Generation rejects them as `GenerationError`; direct runner input
   returns a failed evaluation.
3. Frozen records could retain callers' mutable lists. Sequence fields now copy
   inputs into tuples before validation.

The regression run demonstrated nine failures before fixes. All 185 tests passed
afterward. No review findings were deferred, and no second review was claimed.

## Decisions and limits

- Cached workflow references and a manually maintained ledger replaced helper
  scripts whose relative dependency directory was absent. Risk: bookkeeping drift.
- Execution continued across task boundaries after each verification gate, honoring
  the request to implement. Tradeoff: more work completed before user review.
- Source diversity fills successful duplicates before failed unique sources, keeping
  the spec's global failure-bottom ranking. Tradeoff: less diversity among failures.
- The manifest was created directly before `uv add`, preserving repository files.
  Tradeoff: packaging configuration is maintained manually.
- Live provider availability was not assessed. Account and model access remain
  unverified until a separately configured integration run.
- Scientific superiority was not assessed. Functional correctness does not prove
  any performance advantage or reproduce paper results.
- Hostile-code isolation was not assessed. V0 provides Linux crash/timeout
  containment for locally controlled code; adversarial inputs need stronger OS isolation.

The requested direct-on-main workflow leaves all commits local; no push or merge
was performed. Temporary execution bookkeeping was removed after this record was
saved. Existing specs, plans, and local skills were preserved.
