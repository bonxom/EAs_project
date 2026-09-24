# Mini-MoH v0 specification

Status: Implemented v0 on 2026-09-24; see [implementation and validation record](../2026-09-24-mini-moh-implementation.md).
Source: `docs/moh_codex_project_plan.md`; project rules: `AGENTS.md`.

## Purpose and success

Build a minimal research implementation of two-level heuristic optimization for Euclidean TSP. The inner loop searches heuristic programs; the outer loop searches configurations of heuristic optimizers. An LLM supplies candidate code or optimizer specifications; it is not itself an optimizer.

Success means a small FakeLLM experiment completes both loops, evaluates optimizers on multiple TSP sizes, retains executable candidate sources and evaluation history, and continues after individual candidate failures. A separately configured OpenAI adapter supports real heuristic and optimizer-spec generation. This is an implementation inspired by MoH, not a claim to reproduce paper results.

## Scope and approach

Three approaches are possible: build only the inner loop, build the constrained two-level system, or begin with unrestricted optimizer program evolution. The constrained two-level system is selected for this draft because it demonstrates the defining architecture while keeping optimizer execution testable. Inner-loop-only work is an intermediate milestone; arbitrary optimizer programs require a separate execution and isolation design.

V0 includes TSP10/20/50, seeded baseline heuristics, generated heuristic execution, mutation and crossover, multi-task evaluation, optimizer-spec generation, outer population selection, configuration, CLI, and experiment artifacts. No database, UI, distributed execution, or upstream comparison is required. Generated Python optimizer programs are a follow-up milestone and must preserve the constrained implementation as a baseline.

## Project constraints

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

## Components and responsibilities

`Heuristic` stores ID, source, optional idea, and metadata. Evaluations are separate records tied to task, instances, and seed; a utility from one context must not silently become another context's score.

`Population[T]` holds candidates with explicit capacity and stable ordering. Selection uses successful evaluations ordered by descending utility, breaking ties by candidate ID. Failed candidates rank below every successful candidate.

`Task` supplies fixed instances and trusted scoring rules. `HeuristicRunner` evaluates heuristic source in a child process. `EvaluationResult` records status, optional finite utility, instance lengths, and a bounded error description.

`LLMClient.generate(prompt: str) -> str` is a provider-independent boundary. `FakeLLM` returns deterministic fixtures for heuristic, reflection, and optimizer-spec prompts. Its behavior must not depend on an unrelated optimizer consuming a shared response queue. Provider errors become structured generation failures at the orchestration boundary.

`HeuristicOptimizer` searches heuristics under a validated optimizer specification. `OptimizerEvaluator` runs one optimizer independently on each task and aggregates its scores. `MetaOptimizer` proposes validated specifications. The outer driver evaluates children before survivor selection.

## TSP contract

Instances contain finite coordinates sampled uniformly from the unit square using explicit seeds and NumPy generators. Supported configured sizes are 10, 20, and 50; instance count must be positive. Initial smoke tasks are TSP10 and TSP20 with three instances each.

Each heuristic defines `select_next_node(current_node, unvisited, coordinates) -> int`. A child-process driver starts at city 0, presents unvisited cities in sorted order, validates each returned city, visits every city once, and closes the tour at city 0. Reject booleans, noninteger indices, already visited cities, out-of-range indices, exceptions, and nonfinite results. Calculate lengths from authoritative original coordinates so heuristic mutation cannot alter scoring.

Nearest-neighbor ties select the smallest city index. The random-choice baseline uses a seeded random generator in the worker. Seed Python and NumPy worker RNGs explicitly. Fresh process state prevents candidate globals leaking between evaluations.

Successful heuristic utility is negative mean tour length across the task's instances; higher is better. Any failed instance makes the evaluation failed. Failed utility is JSON null, with status determining its bottom rank; do not serialize Infinity or NaN or average a numeric failure sentinel.

## Execution boundary

Source loading and function execution happen only in a subprocess, including module-level code. A trusted parent validates bounded JSON results; never deserialize candidate-controlled pickle. Capture candidate output separately from the result protocol. Limit source, result, and captured-output sizes, enforce a configurable wall-clock deadline, and terminate and reap the worker process group on timeout. Default limits: 64 KiB source, 1 MiB result, 64 KiB captured output, and five seconds per instance.

Syntax errors, missing functions, invalid returns, process exits, malformed results, and timeouts produce structured failures. Parent configuration errors and output-directory failures abort clearly rather than being mislabeled candidate failures.

Subprocess isolation provides crash and timeout containment; it does not guarantee filesystem or network isolation against hostile code. Prompts forbid imports, file IO, and network IO, but these instructions are not security enforcement. V0 assumes locally controlled experiments. Stronger OS isolation is required before accepting adversarial programs or introducing unrestricted optimizer programs.

## Inner search and optimizer specification

Use one canonical field name, `use_reflection`, resolving the roadmap's inconsistent naming.

An `OptimizerSpec` has parent_selection (`random`, `best`, `tournament`), generation_operator (`mutate`, `crossover`), use_reflection (boolean), survivor_selection (`elitist`, `diversity`), and population_size (integer 2 through 10). Reject unknown fields and type coercion, including booleans supplied as integers.

Initialize each optimizer/task pair with its own baseline candidates and evaluation records. Fill remaining population slots with independently seeded random baselines. Evaluate the initial population once. Each inner iteration selects parents, requests one child, evaluates it, and retains up to the spec's capacity. A failed generation or evaluation consumes the iteration and leaves successful parents available.

Best selection uses score order; random selection is uniform without replacement; tournament selection samples up to two candidates per parent and picks the best, excluding an already selected parent. Crossover uses two distinct candidate IDs; mutation uses one. Reflection makes one idea request followed by one code request; otherwise generation makes one code request. Record the idea and both calls.

Elitist survival retains top-ranked candidates. Diversity survival first retains candidates with distinct exact source hashes in score order, then fills remaining slots from the remaining ranked candidates. It is source diversity, not a claim of behavioral diversity.

## Optimizer evaluation and outer search

Each optimizer uses the same per-task instance sets and evaluation seed schedule. Every task starts with a fresh heuristic population, and every optimizer receives independent RNG and FakeLLM state. Do not carry evolved heuristics or mutable evaluations between tasks or optimizers.

Optimizer utility is the weighted mean of its best successful heuristic utilities across all configured tasks. Weights must be finite, positive, and match the tasks; default weights are equal. If any task has no successful heuristic, optimizer evaluation fails. Raw negative tour lengths are retained for consistency with the roadmap; they are not normalized across problem sizes.

Initialize the outer population with best-mutation and crossover specs, both without reflection and with elitist survival. Smoke configuration uses population size three for each initial inner optimizer. Child specs may change that size within the allowed bounds. Population-size and reflection changes alter work per optimizer; log evaluation and LLM call counts rather than claiming equal-cost comparisons.

Evaluate seed optimizers, then for each outer iteration request one JSON optimizer spec from the current population and scores. Validate, evaluate a valid child on every task, then retain the best outer_population_size optimizers. Invalid children consume that iteration without replacing parents. Cache identical specs only within an identical evaluation context. The last generated child must be evaluated before final selection. If all optimizers fail, finish with an explicit failed-run status and diagnostic artifacts.

## Reproducibility and records

Derive child seeds from the root seed and stable labels using a documented SHA-256-based derivation; never Python's randomized hash. Derivation labels cover task, instance, search generation, candidate, and evaluation repetition. Comparable candidate evaluations use the same task/instance seeds. Candidate IDs use deterministic counters; runtime timestamps do not affect search decisions.

For FakeLLM, identical configuration and seed must reproduce candidate sources, selection decisions, scores, and final winners, excluding timestamps, durations, and paths. Live LLM outputs are not guaranteed reproducible from seeds alone: retain prompts, responses, configuration, model metadata, and usage for audit and subsequent replay.

Each unique run directory contains resolved config.yaml, run.json, events.jsonl, heuristic source files, and optimizer JSON specs. Record lineage, seeds, task identity, status, utility, generation indexes, generation failures, and population updates. Write strict JSON with finite numbers or null. Never persist API keys or secret environment values. Record package versions and available repository revision in run metadata.

## Configuration and CLI

Expose `uv run python -m moh.main --config configs/smoke.yaml` and `--mode tsp-demo`. Smoke defaults: seed 42; two outer iterations; outer capacity two; two inner iterations; initial inner capacity three; TSP sizes 10 and 20; three instances per task; fake provider. Validate the entire config before starting workers or making provider calls. Reject unknown fields, duplicate task sizes, invalid weights, nonpositive deadlines, and invalid population capacities. Zero search iterations are permitted and return evaluated seed populations.

The real-provider example requires an explicit model setting and reads credentials only from OPENAI_API_KEY. The adapter has request timeouts and at most three attempts for transient failures. Core code handles generic generation failures without provider-specific exception types. Strip only an optional enclosing Markdown code fence from heuristic responses; parse optimizer output as JSON with schema validation.

## Acceptance and implementation stages

1. Core models and deterministic TSP: demonstrate valid seeded baseline tours and stable scoring.
2. Execution: prove module-level code runs outside the parent; syntax errors, invalid returns, excessive output, process exits, and infinite loops are contained and workers are reaped.
3. Inner search: run both seed optimizers for three generations using FakeLLM; verify selection, failure handling, and stable tie-breaking.
4. Provider adapter: test mocked transport, credentials, timeouts, bounded retries, and response handling without external calls.
5. Multi-task evaluation: prove independent populations, identical comparison instances, weighted scoring, and failed-task semantics.
6. Constrained meta-optimization: verify every spec option, validation failure, and child evaluation before outer selection.
7. Experiment integration: complete the smoke run twice with matching semantic results and complete artifacts; verify candidate failures do not abort it.

Logging and seed propagation are implemented alongside each owning stage. Each stage's detailed plan will include precise file paths, interfaces, test cases, and runnable commands. Run `uv run ruff check .` and `uv run pytest -q` before completing implementation tasks. A live-provider smoke run is a separate integration validation requiring configured credentials; ordinary tests remain offline.

## Follow-up boundary

Generated optimizer programs, stronger OS isolation, held-out generalization evaluation, equal-budget scientific comparisons, and comparison with the official MoH implementation need separate specifications. V0 completion establishes functional architecture, not measured superiority or research-result equivalence.
