# Behavior-diverse Population Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain quality elites and complementary validation behaviors in training and deployment heuristic populations.

**Architecture:** A shared immutable behavior descriptor validates evaluation identity and per-instance gaps. A dedicated BehaviorPopulation extends the existing Pop interface for heuristic populations; outer optimizer Pop remains unchanged. Parent-owned evaluation callbacks admit all successful candidates, and snapshots preserve behavior and policy metadata.

**Tech Stack:** Python 3.12, uv, NumPy, pytest, Hydra/OmegaConf, existing subprocess evaluators; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-30-behavior-diverse-population-design.md` (approved by user on 2026-10-30).

## Global Constraints

- Use Python 3.12 and uv.
- FakeLLM is mandatory for tests involving LLM generation; no unit test calls external LLM APIs.
- Generated code continues to execute only in timeout-limited subprocesses.
- Preserve provider-independent abstractions.
- Keep the user's current MoH/.gitignore edits intact.
- Apply the policy to training subtask populations and deployment target populations.
- The outer optimizer population keeps its current score-based policy.
- Do not change the meta objective, budget policy, or optimizer comparison protocol.
- Do not revive the reverted unified-training-evaluator project.
- Test results never participate in behavior selection.
- Default diversity_fraction = 0.3; validate finite 0 <= fraction < 1.
- Baseline: 95 tests and 10 subtests pass; Ruff has 35 existing findings.

## Review Focus

1. A duplicate's different raw formatting must not bypass cleaned-code deduplication, but a raw code hash must still match its actual evaluator request (Tasks 1, 2, 4).
2. Effective workers can be lower than configured workers because of CPU count or instance count; identity must record actual settings consistently (Tasks 1, 3).
3. Registering every successful evaluation changes seed bookkeeping; duplicate admission must not increment the number of accepted seeds (Task 4).
4. JSON snapshots convert tuples into lists; round-trip must preserve identity equality and tie-breaking (Task 2).
5. Failed or forged final returns must not erase previously admitted good candidates, and a test evaluation must never update validation selection (Tasks 4, 5).

## File and interface map

| File | Responsibility |
|---|---|
| `MoH/utils/behavior.py` (new) | Immutable identity/descriptor, strict parsing, distance |
| `MoH/utils/behavior_population.py` (new) | Elite/diversity survivor policy and snapshots |
| `MoH/utils/population.py` | Small protected entry-replacement hook for subclass use |
| `MoH/utils/heuristic_evaluator.py` | Expose validated vectors and actual identity in EvalResult |
| `MoH/moh.py` | Training JSON result collection, parent admission, legacy reevaluation |
| `MoH/inference.py`, `MoH/deploy.py` | Target population admission and config wiring |
| `MoH/utils/checkpoint.py` | Preserve policy provenance when converting run snapshots |
| `MoH/utils/run_logger.py` | Persist diversity diagnostics with iteration artifacts |
| `MoH/cfg/config.yaml`, `MoH/cfg/deployment.yaml` | Policy configuration |
| `MoH/tests/test_behavior*.py` (new) | Descriptor, survivor, training regressions |
| Existing evaluator/inference/checkpoint/CLI tests and fakes | Integration coverage |
| `MoH/README.md` | Usage and ablation instructions |

Recommended execution is native, in dependency order. Before execution, follow
using-git-worktrees to determine workspace preference; do not create an isolation
worktree or modify the user's .gitignore during this planning task.

## Task 1: Define validated behavior and distance

**Files:** Create `MoH/utils/behavior.py`, `MoH/tests/test_behavior.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class EvaluationIdentity:
    size: int
    mode: str
    dataset_sha256: str
    indices: tuple[int, ...]
    seed: int
    solver_settings: tuple[tuple[str, int | float], ...]

@dataclass(frozen=True)
class BehaviorDescriptor:
    identity: EvaluationIdentity
    gaps: tuple[float, ...]
    mean_gap_pct: float

def parse_behavior(payload: dict, *, expected: EvaluationIdentity,
                   code_sha256: str) -> BehaviorDescriptor: ...
def behavior_distance(a: BehaviorDescriptor, b: BehaviorDescriptor) -> float: ...
def descriptor_to_dict(value: BehaviorDescriptor) -> dict: ...
def descriptor_from_dict(value: dict) -> BehaviorDescriptor: ...
```

These signatures define the new API; implement bodies during this task. Descriptor
construction and deserialization both validate values, not only parse_behavior.
Identity supports val/test for evaluator transport; population admission permits
only val. Require nonempty, unique nonnegative integer indices, positive size,
finite numeric settings, and canonical sorted settings tuples. Reject booleans
where integers or gap numbers are required. Hashes are 64 hexadecimal characters.

- [x] Write tests with a fixture payload containing size 100, val, two indices
  `(2, 7)`, seed 0, hashes `"a" * 64`/`"b" * 64`, gaps `[1, 3]`, mean 2, zero
  failures, and settings `{workers: 2, instance_timeout: 90, time_limit: 20,
  iteration_limit: 1000, perturbation_moves: 1}`. Test the acceptance contract:

```python
def test_descriptor_preserves_instance_order(valid_payload, identity):
    descriptor = parse_behavior(valid_payload, expected=identity,
                                code_sha256="b" * 64)
    assert descriptor.gaps == (1.0, 3.0)
    assert descriptor.mean_gap_pct == 2.0
    assert descriptor_from_dict(descriptor_to_dict(descriptor)) == descriptor

@pytest.mark.parametrize("gaps", [(float("nan"), 3), (float("inf"), 3), (-1, 3)])
def test_invalid_gaps_are_rejected(valid_payload, identity, gaps):
    valid_payload["instances"] = [
        {"gap_pct": value, "status": "ok"} for value in gaps
    ]
    with pytest.raises(ValueError):
        parse_behavior(valid_payload, expected=identity, code_sha256="b" * 64)
```

- [x] Add parameterized mutations: missing instances, one instead of two entries,
  failures=1, instance status failed, reversed indices, wrong dataset/code hash,
  wrong seed/size/mode/settings, bool count, inconsistent mean, duplicate index.
  Each must raise ValueError. Test distance zero for identical vectors, symmetry,
  and identity mismatch rejection. Test JSON list round-trip.
- [x] Run `uv run pytest -q MoH/tests/test_behavior.py`; record the expected missing
  module failure, then implement the API above.

```python
# Mean agreement after validating finite, nonnegative entries and exact length:
if not math.isclose(math.fsum(gaps) / len(gaps), reported_mean,
                    rel_tol=1e-9, abs_tol=1e-9):
    raise ValueError("Mean gap does not match instances")
# Distance after checking identical identities:
distance = math.sqrt(math.fsum(
    (math.log1p(x) - math.log1p(y)) ** 2
    for x, y in zip(a.gaps, b.gaps, strict=True)
) / len(a.gaps))
```

- [x] Rerun focused tests and Ruff on the two new files; commit only those files:
  `git commit -m "feat: validate heuristic behavior descriptors"`.

## Task 2: Implement deterministic elite/diversity survivors

**Files:** Create `MoH/utils/behavior_population.py`,
`MoH/tests/test_behavior_population.py`; modify `MoH/utils/population.py`.

**Consumes:** Task 1 descriptors and distance.
**Produces:**

```python
class BehaviorPopulation(Pop):
    def __init__(self, identities: dict[str, EvaluationIdentity], size: int,
                 diversity_fraction: float = 0.3): ...
    def admit(self, task: str, idea: str, code: str,
              behavior: BehaviorDescriptor) -> bool: ...
    def save_solution(self, task_name, idea, solution, utility): ...
    def policy_metadata(self) -> dict: ...
    def diagnostics(self, task: str) -> dict: ...
```

Add `Pop._replace_population(task_name, entries)` as the only protected storage
hook needed by the subclass. Reuse existing read access, rank sampling, and
deepcopy/pickling. Override BehaviorPopulation.save_solution to reject unmeasured
insertion with ValueError. Parent code uses admit; generated child insertion is
not an authority. Outer Pop behavior is unchanged.

- [x] Write a complementary-behavior regression with four distinct executable
  sources and one identity. Capacity 3, fraction 0.34 produces two quality slots:

```python
def test_complement_survives_redundant_candidate(identity):
    pop = BehaviorPopulation({"tsp_gls-100": identity}, 3, 0.34)
    vectors = [(0, 2), (0, 2.2), (0, 2.4), (3, 0)]
    codes = [f"def heuristic():\n    return {i}" for i in range(4)]
    for code, gaps in zip(codes, vectors, strict=True):
        pop.admit("tsp_gls-100", "candidate", code,
                  BehaviorDescriptor(identity, gaps, sum(gaps) / 2))
    survivors = pop.get_population("tsp_gls-100")
    assert {x["best_sol"] for x in survivors} == {
        clean_code(codes[i]) for i in (0, 1, 3)
    }
    assert pop.get_best_solution("tsp_gls-100")["utility"] == 1.0
```

- [x] Add equal-mean complementary vectors, duplicate cleaned code, capacity 1,
  fraction 0, capacity 10/fraction .3, invalid fraction including NaN/Infinity,
  invalid capacity including bool, identity mismatch, test identity, and finite
  exact-distance ties. Verify equal insertion sequences produce equal snapshots.
- [x] Run `uv run pytest -q MoH/tests/test_behavior_population.py`; confirm failure.
- [x] Implement admission with compile-only syntax checking and cleaned-code
  SHA-256 deduplication. Use measured descriptor mean as utility. Selection:

```python
quality_key = lambda e: (e["utility"], e["code_sha256"])
ordered = sorted(candidates, key=quality_key)
elite_slots = capacity - math.floor(capacity * diversity_fraction)
selected = ordered[:elite_slots]
remaining = ordered[elite_slots:]
while remaining and len(selected) < capacity:
    winner = min(remaining, key=lambda e: (
        -min(behavior_distance(descriptors[e["code_sha256"]],
                               descriptors[s["code_sha256"]]) for s in selected),
        *quality_key(e),
    ))
    selected.append(winner)
    remaining.remove(winner)
selected.sort(key=quality_key)
```

- [x] Persist entries as plain dictionaries with idea, best_sol, utility,
  code_sha256, behavior, and population_policy. This preserves existing list and
  task-to-list snapshot containers consumed by checkpoint conversion. Policy is
  `{version: "behavior-maxmin-log1p-v1", diversity_fraction: fraction}`.
  Override load methods to validate all entries before replacing state; compare
  identity, code hash, descriptor mean, policy version, and fraction. Reject
  legacy/mismatched snapshots without partially modifying the population.
- [x] Implement diagnostics returning population size, best mean gap (null when
  empty), and mean pairwise behavior distance (0 for fewer than two survivors).
  Add save/load/subtask/all-data and deepcopy tests; restored selection and seeded
  parent sampling must match. No nonfinite JSON values allowed.
- [x] Run focused tests and `uv run pytest -q MoH/tests/test_checkpoint.py
  MoH/tests/test_generated_improver.py`; commit the three task files after green.

## Task 3: Expose validated evaluator behavior

**Files:** Modify `MoH/utils/heuristic_evaluator.py`,
`MoH/tests/test_heuristic_evaluator.py`, `MoH/tests/deployment_fakes.py`;
extend `MoH/tests/test_behavior.py`.

**Consumes:** parse_behavior and EvaluationIdentity.
**Produces:** `EvalResult.behavior: BehaviorDescriptor | None = None` and
`EvalResult.instance_gaps: tuple[float, ...] = ()`; values must agree. Failure or
partial failure never produces a behavior eligible for admission. Existing scalar
failure reporting remains available for test reports.

- [x] Change FAKE_EVAL's successful JSON fixture to emit one real instance entry
  per requested index. Extend FakeEvaluator with an optional code-to-gap-vector
  mapping; compute fixture mean from that vector instead of independent values.
  Existing scalar score fixtures may emit a constant vector in the fake only.
- [x] Add a subprocess regression:

```python
def test_evaluator_exposes_behavior(tmp_path):
    evaluator = HeuristicEvaluator(_root(tmp_path), tmp_path / "out", workers=2)
    result = evaluator.evaluate(HEURISTIC, 500, "val")
    assert result.instance_gaps == (1.25,) * 8
    assert result.behavior.identity.indices == result.indices
    assert dict(result.behavior.identity.solver_settings)["workers"] == 2
```

- [x] Run that test red before adding fields. Add malformed vectors, mean mismatch,
  wrong seed/settings, and partial failure fixtures. Assert malformed payloads
  fail; partial results retain failure count but behavior is None.
- [x] Construct expected identities from the parent's manifest, seed, and actual
  `min(workers, count, os.cpu_count() or 1, 8)` worker count. Call parse_behavior
  after scalar result validation. Keep test-mode descriptors available for
  reporting, but BehaviorPopulation rejects them.
- [x] Rerun evaluator, target initialization, inference, deployment report tests.
  Existing timeout/process cleanup regressions must remain green. Commit only
  the evaluator and fixture/test changes after these pass.

## Task 4: Integrate parent-owned diversity into training

**Files:** Modify `MoH/moh.py`, `MoH/cfg/config.yaml`;
create `MoH/tests/test_behavior_training.py`; update
`MoH/tests/test_improver_validation.py` only where the new admission contract applies.

**Consumes:** BehaviorPopulation, parse_behavior.
**Produces:** `MoH._behavior_registry` keyed by `(task, cleaned_code_sha256)` and
`MoH._ensure_measured_candidate(task, code, idea)` returning the registered
BehaviorDescriptor or None after a failed evaluation. Store measured raw source
hash alongside the descriptor for audit; never use a canonical hash to validate
raw evaluator bytes.

- [x] Add a FakeLLM-backed training fixture using MoH with temporary prompts,
  temporary validation data, RunLogger output, and a stubbed _run_code writing
  realistic JSON results. Preserve real parent parsing and admission; mock only
  the evaluator process, not the policy under test.
- [x] Test three utility calls (elite, redundant, complementary), followed by
  returning only the elite. With capacity 3 and fraction .34, assert complementary
  survives when a fourth redundant candidate arrives. Assert forged returned
  utility -999 does not enter subtask population or per-task aggregation.

```python
# In the fixture's generated optimizer body:
for code in codes:
    utility(code, "evaluated", task)
return "claimed winner", codes[0], -999
# Assertions use measured values, not the supplied -999.
assert all(e["utility"] >= 0 for e in optimizer.subtask_pop.get_population(task))
assert optimizer.subtask_pop.check_solution(task, complementary_code)
```

- [x] Add tests for unseen returned code requiring evaluation; child mutation
  isolation; failed candidate followed by success; duplicate callbacks not
  incrementing seed population size; wrong-task or test-mode utility requests
  rejected before admission. Run `uv run pytest -q MoH/tests/test_behavior_training.py`
  and record red behavior.
- [x] Build expected validation identities with load_eval_dataset and the existing
  training solver settings (8 workers capped by count/CPU, timeout from cfg,
  time_limit 20, iteration_limit 1000, perturbation_moves 1). Use immutable
  identities for each run. Instantiate BehaviorPopulation for subtask_pop only.
  Add config:

```yaml
population:
  diversity_fraction: 0.3
```

- [x] Extend _run_code with a keyword-only result_path and pass --result-path to
  the existing eval.py subprocess. Each evaluation allocates a fresh JSON path
  beneath run logs; a missing result or nonzero process exit is a failed candidate.
  Keep subprocess isolation and existing timeouts; no shared-evaluator migration.
  Parse JSON using the expected raw code hash and Task 1 validator. Register and
  admit every successful validation result, returning its measured mean.
- [x] Update get_improver to resolve its returned code through
  _ensure_measured_candidate. In meta_utility use this measured value and remove
  the redundant scalar save_solution call. Leave outer optimizer selection intact.
- [x] Update generate_seed_algorithm to use actual population size after admission,
  with a target `min(5, cfg.pop_size)` so small populations can terminate. Avoid
  duplicate save_solution calls. No unrelated budget or retry redesign.
- [x] Add legacy population tests: file with score 0 and no descriptor is evaluated
  to its current score before admission; mismatched identity is reevaluated; failed
  candidates are excluded; if none survive, generate fresh seeds. Matching valid
  snapshots load without reevaluation. Repeat read_base_algorithm calls reuse the
  run population rather than repeatedly loading disk. Do not delete input files.
- [x] Implement strict load first, then controlled legacy/mismatch fallback reading
  code/idea only and calling evaluate_heuristic for each candidate. Save new entries
  with descriptors. Preserve valid candidates if a later reevaluation fails.
- [x] Run training, improver validation, generated subprocess, and population tests.
  Commit the training integration and config once green.

## Task 5: Integrate deployment and provenance

**Files:** Modify `MoH/inference.py`, `MoH/deploy.py`,
`MoH/cfg/deployment.yaml`, `MoH/utils/run_logger.py`, `MoH/utils/checkpoint.py`,
`MoH/tests/test_inference.py`, `MoH/tests/test_target_initialization.py`,
`MoH/tests/test_deploy_cli.py`, `MoH/tests/test_checkpoint.py`,
`MoH/tests/test_deployment_report.py`.

**Consumes:** BehaviorPopulation and EvalResult.behavior.
**Produces:** InferenceRunner(..., diversity_fraction=0.3) and
initialize_target_population(..., diversity_fraction=0.3, evaluation_identity=None).
Standalone initialization obtains identity from its first valid evaluated result;
runner supplies preflight identity and verifies every result against it.

- [x] Extend inference tests using FakeEvaluator behavior vectors and FakeLLM.
  Program evaluates complementary candidates but returns only the original seed;
  inspect saved population snapshot for the complementary survivor. Final selected
  heuristic must still be the best measured mean over valid evaluations.
- [x] Test startup parameter rejection before any LLM calls, partial failures not
  entering population, test-vector rejection, same-seed repeated runs, and fractions
  0/.3 reaching the actual policy. Keep poisoning/unscored-return tests active.
- [x] Run inference/initialization/CLI tests red for the new policy assertions.
- [x] Construct BehaviorPopulation from target identity; replace both initialization
  and iteration save_solution calls with admit using result.behavior. Retain
  evaluated candidates only when behavior is fully valid. In initialization avoid
  double admission while the runner closure's population variable is still None.
  Preserve budget/cache logic; enrich the cache key with ordered indices and actual
  settings to match descriptor identity.
- [x] Wire validated config to runner:

```python
fraction = cfg.get("population", {}).get("diversity_fraction", 0.3)
# Shared policy validation must reject bool, NaN, Infinity and values outside [0, 1).
runner = InferenceRunner(
    checkpoint, llm, evaluator, output_dir, heuristic_format,
    diversity_fraction=fraction,
)
```

The snippet illustrates the new keyword. In _execute_deployment retain the
existing iterations, population_size, seed_count, initialization_attempts,
reference_count, max_evaluations, optimizer_timeout, and seed keyword arguments
and append diversity_fraction; do not replace their configured values with defaults.

- [x] Add selection policy and diagnostics to iteration snapshots and selected
  metadata. Training RunLogger writes a companion diversity JSON for each subtask
  snapshot; existing utility CSV layout remains unchanged. Training checkpoint
  metadata includes population_policy. Conversion of a saved training run carries
  the entry policy into checkpoint metadata and rejects mixed policies rather
  than silently labeling the checkpoint incorrectly.
- [x] Add round-trip checkpoint/legacy conversion tests. Verify reference elites
  are evaluated on target validation data and never reuse training descriptors.
  Evaluate final test with a radically different vector; assert saved validation
  population and selected validation code remain unchanged.
- [x] Rerun all modified integration test modules. Commit only the integration,
  config, provenance and test files after green.

## Task 6: Documentation and final verification

**Files:** Modify `MoH/README.md`; update this plan's checkboxes with actual results.

- [x] Document default policy, floor rounding (capacity 10 => 7 elite / 3 diversity),
  failure exclusion, order dependence, identity matching, and legacy reevaluation
  costs. Document that quality-only ablation also requires valid measured vectors.
- [x] Add explicit commands, executed from MoH, for training and deployment
  overrides. These are instructions only: do not launch real LLM experiments.

```sh
uv run python main.py population.diversity_fraction=0 seed=0
uv run python main.py population.diversity_fraction=0.3 seed=0
uv run python deploy.py checkpoint=/absolute/checkpoint population.diversity_fraction=0
uv run python deploy.py checkpoint=/absolute/checkpoint population.diversity_fraction=0.3
```

- [x] Explain that comparison requires matched initial candidates, seeds, instance
  identities, provider configuration and evaluation budget; seeds alone do not
  reproduce live LLM output. Report best gap and mean pairwise behavior distance.
  Do not assert improved benchmark quality from unit tests.
- [x] Run from repository root:

```sh
uv run ruff check .
uv run pytest -q
git diff --check
git status --short
```

- [x] Fix regressions introduced by this feature; report pre-existing Ruff findings
  separately. Do not bulk-fix unrelated code or overwrite MoH/.gitignore.
- [x] Review the final diff against the spec and all five Review Focus cases.
  Commit README and plan result updates. Summarize implementation, test evidence,
  artifact/config changes, and remaining reproducibility limitations.
  Commit README and plan result updates. Summarize implementation, test evidence,
  artifact/config changes, and remaining reproducibility limitations.

## Coverage audit and handoff

Descriptor integrity and distance: Task 1. Elite preservation, deterministic
survivors, snapshots: Task 2. Evaluator transport and effective settings: Task 3.
Training callbacks and old scores: Task 4. Deployment, test isolation and policy
provenance: Task 5. Reproducible ablation instructions and repository checks: Task 6.
No benchmark result or real LLM run is required to complete implementation.

Implementation starts only after plan review and execution-method selection.
Recommended: native execution in this session, because Tasks 1–5 share a tight
descriptor/admission contract and need sequential integration. Alternative:
subagent-driven implementation with per-task reviews. No execution method has
been selected yet; no product code was changed while preparing this plan.
