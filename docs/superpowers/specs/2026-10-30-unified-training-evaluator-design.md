# Unified training evaluator

Status: Ready for user review before implementation planning.

## Intent and scope

Implement the first prioritized repository improvement: training and deployment
share the existing HeuristicEvaluator. The user authorizes replacing the legacy
training path without backwards compatibility. Preserve the two-level search:
the inner loop searches heuristics; the outer loop searches optimizer programs.
LLMClient remains a provider-independent generation service.

This change covers training evaluation and run-local population persistence.
Authoritative optimizer score verification, strict search budgets, paired
optimizer comparisons, and new search objectives remain separate follow-ups.

## Design decision

Reuse utils/heuristic_evaluator.py rather than extending the legacy stdout parser
or introducing another evaluation service. The shared service already isolates
candidate files, validates result metadata, and terminates evaluator process
groups on timeout. MoH retains responsibility for search and evaluation counters.

## Evaluation contract

MoH constructs a HeuristicEvaluator rooted at the source directory and writes its
artifacts beneath RunLogger.output_dir. Training configuration exposes
val_n_instances (default 8), workers (default 8), instance_timeout (default 90),
and evaluation_timeout (default null). Replace the old training timeout key;
optimizer_timeout remains independent. seed controls validation instance selection.

Preflight all configured training sizes against validation data before any LLM
request. Reject unsupported problems, invalid settings, and missing or insufficient
datasets as configuration failures. Training never preflights or evaluates test
data. Deployment retains its existing separate final test evaluation.

Remove MoH._run_code and the training subprocess/stdout parser. The utility adapter
delegates candidate evaluation to HeuristicEvaluator.evaluate(code, size, "val").
Successful finite scores preserve the evaluator's mean-gap semantics, including
its existing per-instance penalties. Failed evaluations map to the existing 1e6
search penalty and log the structured failure reason. Unexpected infrastructure
errors must remain distinguishable from candidate failures.

Reject a task not configured for this run and reject modes other than validation
before launching evaluation. Maintain actual evaluation-attempt counters so
existing iteration diagnostics remain useful. Do not add strict budget enforcement
in this change. Use the evaluator's source validation instead of the legacy
substring check for "random".

Persist the validation manifest and structured EvalResult records in the run
workspace, including code and dataset hashes, indices, seed, solver settings,
status, and stdout location. Use strict JSON; unsuccessful optional scores are null.
Do not persist credentials.

## Population and artifact ownership

Initialize each training run with fresh in-memory populations. read_base_algorithm
generates seeds only when the current run lacks the required population; it never
loads source-tree solution_*.json files. Persist current populations under the
existing run-local pop/subtask directory. Repeated calls in the same run reuse
its in-memory population, without regenerating seeds unnecessarily.

No training writes to problems/tsp_gls/gpt.py or source-tree solution files.
Do not delete pre-existing files. Resuming or importing old populations is outside
scope, so scores from another dataset or run cannot enter implicitly.
Final checkpoint export continues to consume the run's in-memory population.

## Engineering constraints

- Python 3.12; use uv.
- Tests use FakeLLM and never call external LLM APIs.
- Generated code loads and executes only in subprocesses with timeouts.
- One generated candidate failure must not crash an experiment.
- Preserve provider-independent core interfaces.
- Leave unrelated working-tree changes intact.

## Acceptance and test strategy

Write and run regression tests before implementation and confirm expected failures:

1. Training delegates to the shared evaluator with the configured size and val
   mode; finite successful results become search scores.
2. Failed and nonfinite results produce the failure penalty and diagnostics;
   a subsequent valid candidate can still be evaluated.
3. Unknown tasks and test-mode requests cannot launch an evaluation.
4. Preflight rejects bad validation data before FakeLLM receives a request.
5. Legacy source-tree population files are ignored, even if they contain a
   deceptively good score. Existing gpt.py and solution files remain unchanged.
6. Two training instances with distinct workspaces keep separate candidate files,
   results, and population snapshots; run-local reuse avoids duplicate seeding.
7. Structured records contain the evaluation identity and failure status and can
   be parsed as strict JSON. Existing checkpoint and deployment tests still pass.
8. Existing evaluator subprocess tests continue to exercise code isolation,
   timeout cleanup, and result validation rather than duplicating those internals.

Update README and training config examples to describe the new keys, fresh-run
population semantics, and artifact locations. Run focused tests and then
`uv run ruff check .` and `uv run pytest -q` from the repository root.

## Baseline and limitations

The preceding repository inspection found 95 passing tests and 10 passing
subtests, with 35 Ruff findings in existing Python code. Record any differences
when validating this change; fix regressions introduced here.

Seeded dataset selection does not guarantee bitwise reproducibility of live LLM
responses or time-limited GLS. This change does not claim otherwise. Subprocess
isolation here provides execution and timeout separation, not a security sandbox.
