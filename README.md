# Mini-MoH

A minimal research implementation inspired by Meta-Optimization of Heuristics,
using Euclidean TSP. Python 3.12, uv, and Linux are required.

The **inner loop** (`HeuristicOptimizer`) searches Python heuristic programs.
The **outer loop** (`MetaOptimizer`) proposes validated configurations of those
optimizers. `LLMClient` supplies text; it is not an optimizer. V0 evolves constrained
optimizer specifications, not arbitrary optimizer Python programs, and makes no
claim to reproduce paper results or outperform established baselines.

## Run offline

```bash
uv sync --locked
uv run python -m moh.main --mode tsp-demo
uv run python -m moh.main --config configs/smoke.yaml
uv run ruff check .
uv run pytest -q
```

The smoke configuration uses FakeLLM, seed 42, TSP10 and TSP20 with three instances
each, two inner iterations, two outer iterations, inner capacity three, and outer
capacity two. TSP50 is also supported. Zero iterations still evaluate the seed
populations. CLI exit codes: 0 success, 1 all candidates failed, 2 configuration or
infrastructure error. Output paths are relative to the current working directory.
All tests run offline; provider tests inject transport stubs.

## Search and scores

Each heuristic defines
`select_next_node(current_node, unvisited, coordinates) -> int`.
The worker starts at city 0, supplies sorted unvisited cities, visits every city
once, and closes the tour. The trusted parent validates the complete tour and
scores original coordinates. Mutation of worker inputs cannot alter scoring.
Nearest-neighbor ties choose the smallest city index; random baselines have
independent policy seeds.

Utility is negative mean tour length: higher is better. One failed instance
fails the candidate; failed utility is JSON `null`, never Infinity or NaN.
Successful candidates always rank above failures, with candidate ID breaking ties.
Optimizer utility is the positive weighted mean of its best successful heuristic
on every task. Any task without a successful heuristic fails the optimizer.
Weights default to equal and must match the task list. Scores are raw lengths,
not normalized across sizes.

Optimizer specs support random/best/tournament parent selection, mutation or
two-distinct-parent crossover, optional idea-then-code reflection, elitist or exact
source-hash diversity survival, and inner capacities 2–10. Diversity prefers unique
successful sources, fills with successful duplicates, then considers failures.
Returned populations are score ordered. Every outer child is evaluated before
selection, including the final child. Identical specs are reevaluated; no cache is
used. Different population sizes and reflection settings consume different work;
records include actual evaluation, instance-attempt, and LLM-call counts.

## Execution limits

Generated code, including module-level statements, runs only in a fresh child
process per instance. Default limits are 64 KiB source, 1 MiB result JSON, 64 KiB
combined stdout/stderr, and a five-second wall-clock deadline per instance.
Nonblocking pipes and process-group cleanup contain output floods, crashes, and
hangs. Linux subreaper support cleans up orphaned descendants in the worker group.
The process-wide subreaper setting remains enabled; reaping is restricted to each
worker's process group. This implementation assumes sequential experiments.

This is crash/timeout containment for locally controlled experiments, **not a
security sandbox**. Generated Python can access the filesystem and network and
could escape its process group. Prompts forbid imports and IO but do not enforce
those restrictions. Adversarial programs require stronger OS isolation.

## Reproducibility and artifacts

Seed derivation is exactly:

```python
payload = json.dumps([root, *labels], ensure_ascii=True, separators=(",", ":"))
seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")
```

Labels distinguish tasks, instances, candidate policy seeds, search generations,
and evaluation repetitions. All comparable heuristics use the same worker seed
schedule; optimizer identity affects lineage IDs, not comparison RNGs. Every
task/optimizer pair gets a fresh population and a fresh FakeLLM with independent
per-kind response cursors. Both Python and NumPy worker RNGs are seeded.

Each unique directory under `outputs/` contains:

- `config.yaml`: resolved configuration, including equal default weights.
- `run.json`: terminal status, winner, population, dependency versions and Git revision.
- `events.jsonl`: ordered strict JSON records with sources, lineage, contexts, seeds,
  prompts, responses, generation failures, scores, populations, and work counts.
- `heuristics/<id>.py`: every evaluated heuristic source, including failed programs.
- `optimizers/<id>.json`: every evaluated optimizer specification.

Fake runs reproduce sources, decisions, scores, winners, and event order. Unique
run-directory names and output paths are excluded from comparisons; runtime
versions and revision describe the execution environment. Artifact write failures
abort the run instead of becoming candidate failures. No environment dump or API
key is recorded.

## Real provider

Replace `YOUR_MODEL_NAME` in `configs/real_llm.yaml` with a model available to your
account, set `OPENAI_API_KEY` in your environment, then run:

```bash
uv run python -m moh.main --config configs/real_llm.yaml
```

The adapter uses the [official Responses API Python interface](https://developers.openai.com/api/docs/libraries),
with explicit request timeouts, SDK automatic retries disabled, and at most three
attempts for transient errors. Provider-specific SDK and credential handling live
only in `src/moh/llm/openai_client.py`. Permanent errors and malformed responses do
not retry. Prompts/responses and per-attempt model/usage records are retained;
known credentials are redacted at the adapter boundary.

Live responses are not seed-reproducible. Retained responses support auditing and
future replay tooling; an automated replay command is outside v0. No live-provider
validation was performed during implementation. Mocked tests do not establish
account or model access.

## Design

See the [specification](docs/superpowers/specs/2026-09-23-mini-moh-design.md) and
[implementation plan](docs/superpowers/plans/2026-09-24-mini-moh-v0.md).
Generated optimizer programs, hostile-code isolation, held-out evaluation,
equal-budget comparisons, and upstream paper-result comparisons are follow-up work.
