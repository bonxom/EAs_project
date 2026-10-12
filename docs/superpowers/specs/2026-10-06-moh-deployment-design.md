# MoH deployment design

## Agreed intent

Deploy the learned program `I*_T` to TSP500/TSP1000 as a heuristic optimizer.
Training passes an optimizer population and optimizer utility to this program.
Inference passes a heuristic population and target heuristic utility to the same
program. Inference does not search for or replace optimizer programs.

The user's correction is authoritative: retain training elite solutions for
idea generation/modification, build fresh target populations, and never use
training utility values to rank target heuristics.

## Global constraints

- Python 3.12.
- Use uv.
- All experiments must be reproducible with seeds.
- Unit tests must never call external LLM APIs.
- FakeLLM is mandatory for tests.
- Never execute generated code in the main process.
- Generated programs must have timeouts.
- Failure of one generated candidate must not crash an experiment.
- Keep core abstractions provider-independent.
- Do not hardcode OpenAI-specific logic outside llm/openai_client.py.

Reuse the existing provider-independent `BaseClient` interface; introduce no new
provider-specific code in deployment. Actual provider configuration remains in
the existing `utils/llm_client/openai.py` adapter.

## Checkpoint

Export the active accepted optimizer at training completion, including a run
whose last iteration was rejected. Export up to three best heuristic entries
per actual training size, ordered by training validation utility. Keep elite
code and ideas; numeric training scores are provenance only.

```text
checkpoint/
  meta_optimizer.py
  elites/tsp20.json
  elites/tsp50.json
  elites/tsp100.json
  elites/tsp200.json
  metadata.json
```

Use actual configured sizes; a two-size run exports two elite files. Metadata
has schema version 1, problem name, actual training sizes, seed, complete
resolved config with credentials removed, source run, validation provenance,
and SHA-256 hashes of optimizer and elite artifacts. Resolve config without
serializing API keys. Write metadata last in a temporary sibling directory,
then rename into a previously nonexistent destination. Refuse overwrite.

Loader verifies schema, relative artifact paths, sizes, file hashes, and Python
syntax without executing code. Missing/corrupt artifacts fail before LLM calls.
Legacy run conversion uses the last accepted code artifact and the latest
completed subtask snapshot. It must not choose a timestamped candidate or assume
`iter_9.py` exists. If there were no accepted iterations, use the seed optimizer
entry from `pop/improver/pre.json`. Mark unavailable validation provenance as
unknown; such elites can still be reference material, never target scores.

## Evaluation

Validation samples eight unique instances from `data/val` using the run seed;
every candidate for a target size sees the same indices. Test reads a configured
number of instances from `data/test`. Current code uses 100 while existing
tests expect 128. Preserve 100 as the deployment default matching available
data; support explicit 128 and fail early if the data has fewer instances.
Record actual counts and indices. A 100-instance report must not be labeled a
128-instance reproduction.

Give each evaluation an isolated code file under its deployment run. Never
rewrite the shared `problems/tsp_gls/gpt.py`. Execute the evaluator and generated
heuristics in subprocesses, with at most eight instance workers. Retain stdout
and a structured JSON result with per-instance cost, gap, failure status and
indices, mean gap, code hash, dataset hash, seed and GLS settings. Preserve the
legacy final numeric stdout line for existing training callers.

Separate candidate failure (penalty; continue search) from infrastructure/data
failure (explicit failed evaluation; do not select it or fabricate a successful
test report). Per-instance candidate failures stay in the denominator with the
existing penalty `1e6` and contribute to the failure count.

## Initialization and inference

Each target size starts with a fresh `Pop` inside the deployment run. Never
read shared `solution_tsp_gls-*.json`. A provider-independent LLM wrapper adds
deterministically selected training elite ideas/code as reference context to
initialization and subsequent generation prompts. Exclude their numeric scores
from prompt context. Record reference source sizes and code hashes.

Initialize up to five unique successful target heuristics with at most 25
generation attempts. Score every seed on target validation. Proceed with at
least one valid seed, recording partial initialization; otherwise fail that
target and continue other targets. These limits are configurable.

Run the checkpoint program directly for ten iterations per target:

```python
I_star(population=target_pop, utility=target_validation_utility,
       language_model=reference_llm, function_format=heuristic_format,
       task=f"tsp_gls-{size}")
```

Use `GeneratedImprover` for subprocess execution. The utility callback scores
and saves every successfully evaluated heuristic, including a candidate the
program does not return. Population mutations inside the worker are not the
source of truth. Verify returned code against the callback's score cache, or
score it on target validation if it was never evaluated. Ignore the program's
self-reported numeric utility. Do not insert `None`, nonfinite values or failed
evaluations into the population. Derive child RNG seeds from run seed, size and
iteration using SHA-256, not Python's randomized `hash()`.

Cache validation by exact code hash, target size, seed, dataset fingerprint and
solver settings. Never reuse a score across sizes, splits or runs. Enforce a
per-target evaluation budget before launching an evaluation, including seed
initialization. Default budget is 1000. A hit to the score cache consumes no
additional evaluation. Record an early stop when the budget is exhausted.

The immutable checkpoint code is the driver for every iteration. There is no
optimizer population, `meta_utility()` call or optimizer acceptance policy in
this flow. Timeouts, malformed returns and generated exceptions are recorded
per iteration; retain existing valid candidates and continue within the budget.

## Selection and test

Select the smallest target validation gap across all evaluated candidates and
save a standalone heuristic file plus selection metadata. Then freeze that
choice and evaluate it once on target test. Test does not call an LLM or feed
back into generation, ranking or selection. Write successful and failed target
outcomes to `report.json` and `report.csv`. Include target size, actual counts,
validation/test gaps, per-instance failures, elapsed time, heuristic/checkpoint
hashes, dataset fingerprints, seed and early-stop reason.

## CLI and scope

Add `MoH/deploy.py` and `MoH/cfg/deployment.yaml`; preserve the training entrypoint.
Preflight checkpoint integrity, target files and instance counts for all targets
before spending LLM calls. Resolve checkpoint paths against Hydra's original
working directory. Support a subprocess-only offline smoke test with FakeLLM.

No automatic cloud publication, external API experiment, test-data generation,
model change, three-run aggregation, or unrelated lint cleanup is part of this
implementation. These can be requested after the deployment path is verified.
