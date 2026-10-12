# MoH Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Follow session authorization when choosing delegation; do not spawn agents merely because the header recommends them.

**Goal:** Deploy a learned `I*_T` directly as a heuristic optimizer, carrying elite reference solutions into fresh target populations and reporting independent test results.

**Architecture:** Training exports an immutable checkpoint and elite heuristics. A separate inference runner invokes the frozen program on target heuristic populations and target validation utility for ten iterations. A subprocess evaluator supplies trustworthy scores; a final test stage evaluates the already selected heuristic without LLM calls.

**Tech Stack:** Python 3.12, uv, Hydra/OmegaConf, existing `BaseClient`, `Pop`, `GeneratedImprover`, pytest/FakeLLM, JSON/CSV artifacts.

**Spec:** `docs/superpowers/specs/2026-10-06-moh-deployment-design.md`

## Global Constraints

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
- Inference invokes `I*_T` directly on heuristic populations, never on optimizer populations.
- Training utilities are provenance only. Every target population score comes from target validation.
- Generated code and populations belong to the deployment output directory.
- No live LLM experiment is needed to implement or validate this plan.

## Current baseline and decisions

- Work on the user's `moh-deployment` branch; preserve unrelated edits to `MoH/cfg/problem/tsp_gls.yaml` and `MoH/problems/tsp_gls/gpt.py`.
- `GeneratedImprover` already executes code in a child and forwards utility/LLM calls to the parent. Reuse it.
- `Pop` in a generated worker is a copied object. Save evaluated candidates in the parent's utility callback.
- `MoH.evaluate_heuristic()` currently rewrites shared `gpt.py` and parses stdout. Deployment needs an isolated evaluator, not a partially initialized `MoH` object.
- `read_base_algorithm()` loads shared populations with historic scores. Do not call it from deployment.
- Current `dataset_selection.INSTANCE_COUNTS` is `{"val": 8, "test": 100}`, while tests expect 128. Task 2 resolves this mismatch through an explicit count argument and tests for both 100 and 128.
- Existing data for sizes 100/200 has 10 validation and 100 test instances. Preflight the actual 500/1000 files; do not assume their counts.
- Export three elites per training size by default, and five target seeds with a 25-attempt initialization limit. All three limits are configurable.
- Baseline measured on 2026-10-06: `uv run pytest -q` reports 33 passed, 3 failed and 10 subtests passed; all three failures are the 100/128 mismatch described above. `uv run ruff check .` reports 49 existing violations. These are observations, not passing checks.

## Review Focus

1. Last training iteration rejected or no iteration accepted: checkpoint must contain the active optimizer, not an arbitrary candidate. Task 1 tests both paths.
2. Fabricated or stale utility returned by generated code: only evaluator/cache scores affect selection. Task 4 tests the callback and returned-code paths.
3. Candidate evaluated but not returned, or mutation of a child population: keep all successful parent-side evaluations. Task 4 tests both conditions.
4. Checkpoint/dataset paths after Hydra changes cwd, missing test instances, shared code-file races: preflight paths and isolate evaluation artifacts. Tasks 2 and 6 test these inputs.
5. Test failure or partially initialized target: report failures explicitly and never rerank using test scores. Tasks 3, 4 and 5 test these outcomes.

## File structure

| File | Responsibility |
|---|---|
| `MoH/utils/checkpoint.py` | Export/load checkpoints and convert an existing training run |
| `MoH/utils/heuristic_evaluator.py` | Isolated subprocess evaluation and typed results |
| `MoH/utils/elite_reference.py` | Add deterministic elite reference context to LLM prompts |
| `MoH/inference.py` | Initialize target populations and run direct heuristic search |
| `MoH/utils/deployment_report.py` | Freeze selections, evaluate test, write reports |
| `MoH/deploy.py`, `MoH/cfg/deployment.yaml` | Deployment CLI, preflight and wiring |
| `MoH/tests/deployment_fakes.py` | Shared FakeLLM/FakeEvaluator and reproducible checkpoint fixtures |
| `MoH/problems/tsp_gls/eval.py`, `dataset_selection.py` | Explicit counts, isolated heuristic loading and structured evaluation output |

### Task 1: Checkpoint export/load and training integration

**Files:** Create `MoH/utils/checkpoint.py`, `MoH/tests/test_checkpoint.py`, `MoH/tests/deployment_fakes.py`; modify `MoH/moh.py` at the end of `run_meta_optimizer()`.

**Interfaces:**
- Produce frozen `Elite(source_size: int, idea: str, code: str, code_sha256: str)` and `TrainingCheckpoint(code: str, elites: tuple[Elite, ...], metadata: dict)`.
- Produce `export_checkpoint(output_dir: Path, code: str, population: Pop, metadata: dict, elite_count: int = 3) -> Path`.
- Produce `load_checkpoint(checkpoint_dir: Path) -> TrainingCheckpoint` and `export_run_checkpoint(run_dir: Path, output_dir: Path, elite_count: int = 3) -> Path`.
- Exported elite JSON also contains `training_utility`; `Elite` deliberately omits it from the inference interface.

- [ ] **Step 1: Add reusable offline fixtures and checkpoint contract tests.** Use these fixture implementations in `deployment_fakes.py`:

```python
from utils.population import Pop

HEURISTIC = "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n    return edge_distance\n"
OPTIMIZER = '''from utils.utils import extract_code
def improve_algorithm(population, utility, language_model, function_format, task):
    responses = language_model.prompt_batch("expert", [function_format])
    for code in extract_code(responses):
        if code:
            utility(code, "generated", task)
    best = population.get_best_solution(task)
    return best["idea"], best["best_sol"], best["utility"]
'''

class FakeLLM:
    batch_size = 1
    model = "fake"
    temperature = 0.0
    def __init__(self, responses=None):
        self.responses = list(responses or [HEURISTIC])
        self.messages = []
    def prompt(self, expertise, message, temperature=None):
        self.messages.append(message)
        return self.responses.pop(0) if self.responses else HEURISTIC
    def prompt_batch(self, expertise, message_batch, temperature=None):
        return [self.prompt(expertise, message, temperature) for message in message_batch]

def training_population():
    sizes = [20, 50, 100, 200]
    pop = Pop([f"tsp_gls-{size}" for size in sizes], 10)
    for size in sizes:
        pop.save_solution(f"tsp_gls-{size}", f"elite-{size}", HEURISTIC, 0.01)
    return pop
```

Concrete roundtrip and corruption test:

```python
def test_checkpoint_keeps_elites_and_checks_hashes(tmp_path):
    import pytest
    from deployment_fakes import OPTIMIZER, training_population
    from utils.checkpoint import export_checkpoint, load_checkpoint
    destination = tmp_path / "checkpoint"
    export_checkpoint(destination, OPTIMIZER, training_population(),
                      {"training_sizes": [20, 50, 100, 200], "seed": 42})
    checkpoint = load_checkpoint(destination)
    assert checkpoint.code == OPTIMIZER
    assert {elite.source_size for elite in checkpoint.elites} == {20, 50, 100, 200}
    assert not hasattr(checkpoint.elites[0], "training_utility")
    (destination / "meta_optimizer.py").write_text(OPTIMIZER + "\n# tampered")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(destination)
```

Add named tests `test_export_refuses_existing_destination`, `test_elite_hash_tampering_is_rejected`, `test_last_rejected_iteration_exports_previous_active_code`, `test_no_accepted_iteration_exports_seed`, `test_two_size_run_exports_only_two_elite_files`, `test_checkpoint_metadata_excludes_credentials`, and `test_metadata_cannot_reference_parent_directory`. Build legacy-run fixtures with `csv.writer`, `json.dump`, and the fixture strings; no generated execution is needed.

- [ ] **Step 2: Run RED.** `uv run pytest -q MoH/tests/test_checkpoint.py`; expected missing checkpoint module or missing functionality, with no API calls.
- [ ] **Step 3: Implement dataclasses, export, integrity checks and conversion.** Use the following export call at training completion:

```python
export_checkpoint(
    Path(self.run_logger.output_dir) / "checkpoint", self.improver_str,
    self.subtask_pop,
    {"training_sizes": list(self.cfg.problem.problem_size),
     "seed": getattr(self.cfg, "seed", 0),
     "source_run": self.run_logger.output_dir,
     "training_meta_utility": self.meta_utility_val,
     "config": redacted_config(self.cfg)},
)
```

Define `redacted_config(cfg) -> dict` in `checkpoint.py`; convert OmegaConf with unresolved interpolation, recursively remove keys containing `api_key`, `token`, `secret`, or `password`, then resolve only remaining nonsecret settings. Never resolve or log credential values. Capture validation provenance from source evaluation headers (data path, seed, indices), mark absent provenance unknown, and fingerprint available dataset files by streaming reads. Enforce schema/hash validation on load, including elite files; compile syntax only. Define all checkpoint metadata defaults in export, including schema version and problem name.

For legacy conversion, read latest completed `pop/subtask/iter_N.json` and `meta_utility.csv`; select the last `accepted=True` code file, otherwise the sole seed entry in `pop/improver/pre.json`. Reject missing or ambiguous artifacts instead of searching timestamped candidate files. Add a guarded argparse module entrypoint supporting `--run`, `--output`, `--elite-count`.

- [ ] **Step 4: Run GREEN.** `uv run pytest -q MoH/tests/test_checkpoint.py MoH/tests/test_improver_validation.py`. In existing mocked run-loop tests, patch `moh.export_checkpoint`, set `run_logger.output_dir` to a temporary directory string, and give the test config `problem.problem_size`, `seed` and the existing timeout fields through `OmegaConf.create`. Mocking the export function alone is insufficient because its arguments are evaluated first. Expected all pass and mocked loops create no repository artifacts.
- [ ] **Step 5: Commit only Task 1 files.** `git commit -m "feat: export immutable MoH checkpoints with elite references"` after staging exact files above.

### Task 2: Isolated evaluator, explicit counts and structured results

**Files:** Create `MoH/utils/heuristic_evaluator.py`, `MoH/tests/test_heuristic_evaluator.py`; modify `MoH/problems/tsp_gls/eval.py`, `dataset_selection.py`, `MoH/tests/test_tsp_dataset_selection.py`.

**Interfaces:**
- Extend `select_instance_indices(data, mode, seed=0, count=None)` and `load_eval_dataset(root_dir, problem_size, mode, seed=0, count=None)`, preserving positional callers. Default val count 8, test count 100; accept test count 128 explicitly.
- Produce frozen `EvalResult(size: int, mode: str, mean_gap_pct: float | None, status: str, failures: int, indices: tuple[int, ...], code_sha256: str, dataset_sha256: str, elapsed_seconds: float, stdout_path: str, error: str | None)` with `status` equal to `ok` or `failed`.
- Produce `HeuristicEvaluator(root_dir, output_dir, seed=0, val_count=8, test_count=100, workers=8, instance_timeout=90, evaluation_timeout=None)` and `evaluate(code: str, size: int, mode: str) -> EvalResult`.
- Produce `preflight(sizes: list[int], include_test: bool) -> dict` returning data paths, fingerprints, selected indices and solver settings; this method never starts a solver or calls an LLM.
- Define the manifest shape as `{size: {"val": {"dataset_path": str, "dataset_sha256": str, "indices": tuple[int, ...]}, "test": {"dataset_path": str, "dataset_sha256": str, "indices": tuple[int, ...]}}}`; omit `test` when `include_test=False`. Evaluators expose `seed`, `val_count`, `test_count`, and `solver_settings` (workers, instance timeout, GLS time limit 20, iteration limit 1000, perturbation moves 1) for cache keys/report metadata. Dataset hashing uses streaming reads, not a second full in-memory copy.

- [ ] **Step 1: Write RED tests for both counts, empty/missing data and isolation.** Replace hardcoded test-default assertions with explicit 128-count calls, plus assertions for the 100 default:

```python
def test_test_count_is_explicit():
    data = dataset(128)  # Existing fixture in test_tsp_dataset_selection.py
    assert len(select_instance_indices(data, "test")) == 100
    assert len(select_instance_indices(data, "test", count=128)) == 128
```

In evaluator tests, use `tmp_path`, tiny trusted heuristic code, and fake data/solver subprocesses. Define `test_evaluation_does_not_modify_shared_gpt`, `test_missing_test_instances_fail_preflight`, `test_worker_failure_is_counted_in_mean`, `test_failed_evaluator_does_not_return_success_score`, `test_process_timeout_reaps_workers`, `test_configured_count_matches_report`, and `test_result_hash_and_indices_match_request`. Mock `subprocess.Popen` for timeout/failure cases and assert process cleanup; use one real subprocess smoke test for structured-result parsing, with trusted fixture code and no generated code executed in the test process.

- [ ] **Step 2: Run RED.** `uv run pytest -q MoH/tests/test_tsp_dataset_selection.py MoH/tests/test_heuristic_evaluator.py`; expected explicit-count/module failures.
- [ ] **Step 3: Implement a CLI-compatible evaluator extension.** Preserve existing positional arguments; add argparse flags `--count`, `--heuristic-path`, `--result-path`, `--workers`. Move generated heuristic loading out of top-level imports into the process-pool initializer `_initialize_heuristic(path)`, so spawn workers initialize the callable independently. Use `importlib.util.spec_from_file_location` only inside the evaluator's instance worker. Retain the final numeric stdout line.

Implement parent launch with this pattern:

```python
command = [sys.executable, "-u", str(eval_path), str(size), str(root_dir),
           mode, str(instance_timeout), str(seed), "--count", str(count),
           "--heuristic-path", str(code_path), "--result-path", str(result_path),
           "--workers", str(workers)]
timeout = evaluation_timeout or (math.ceil(count / workers) * instance_timeout + 30)
```

Write code/result/stdout to a unique directory inside the deployment output. Use a dedicated process group on Linux; on whole-evaluation timeout, terminate/kill the group and reap the parent, so no pool workers remain. Cap workers at `min(requested_workers, count, os.cpu_count() or 1)`, with requested workers no larger than 8. Reject malformed/nonfinite JSON results and mismatched count, size, mode, code hash or selected indices. Return a failed `EvalResult` for infrastructure errors; preserve failed instance penalties in successful evaluator summaries.

- [ ] **Step 4: Run GREEN.** Repeat Step 2 command; expected pass for 100/128 selection and isolated evaluation. Also run existing generated-improver tests to verify subprocess imports still work.
- [ ] **Step 5: Commit exact Task 2 files.** Message: `feat: evaluate deployment heuristics in isolated workspaces`.

### Task 3: Elite reference context and bounded target initialization

**Files:** Create `MoH/utils/elite_reference.py`, `MoH/inference.py`, `MoH/tests/test_target_initialization.py`; extend `MoH/tests/deployment_fakes.py` with `FakeEvaluator`.

**Interfaces:**
- Consume `Elite`, `EvalResult`, `HeuristicEvaluator`, `Pop` and `BaseClient` methods from Tasks 1/2.
- Produce `EliteReferenceLLM(client, elites, target_size, seed=0, reference_count=2)` exposing `prompt`, `prompt_batch`, `batch_size`, `model`, `temperature`. Prompts include idea/code/source size, never numeric training utilities.
- Produce frozen `SelectedHeuristic(size: int, idea: str, code: str, validation: EvalResult)`.
- Produce `initialize_target_population(size, elites, llm, evaluate_candidate, heuristic_format, population_size=10, seed_count=5, max_attempts=25, seed=0, reference_count=2) -> Pop`.
- `evaluate_candidate(code: str, idea: str) -> EvalResult` is supplied by Task 4's target session. Task 3 tests supply a FakeEvaluator callback.

- [ ] **Step 1: Add fake evaluation service.** `FakeEvaluator.evaluate` records `(code, size, mode)`, computes SHA-256 and returns `EvalResult` using a configurable score map; missing code entries default to 1.0. It exposes `calls`, can return `status="failed"` for selected codes, and never imports/evaluates the heuristic. Use this concrete assertion:

Add this actual fake implementation to `deployment_fakes.py` (which consumes Task 2's result type):

```python
import hashlib
from utils.heuristic_evaluator import EvalResult

class FakeEvaluator:
    def __init__(self, scores=None, failed_codes=()):
        self.scores = scores or {}
        self.failed_codes = set(failed_codes)
        self.calls = []
        self.seed, self.val_count, self.test_count = 0, 8, 100
        self.solver_settings = {"workers": 8, "instance_timeout": 90,
                               "time_limit": 20, "iteration_limit": 1000,
                               "perturbation_moves": 1}
    def preflight(self, sizes, include_test):
        modes = ("val", "test") if include_test else ("val",)
        return {size: {mode: {
            "dataset_path": f"fake/{mode}/{size}",
            "dataset_sha256": hashlib.sha256(f"{mode}:{size}".encode()).hexdigest(),
            "indices": tuple(range(self.val_count if mode == "val" else self.test_count)),
        } for mode in modes} for size in sizes}
    def evaluate(self, code, size, mode):
        self.calls.append((code, size, mode))
        failed = code in self.failed_codes
        manifest = self.preflight([size], include_test=True)[size][mode]
        return EvalResult(size=size, mode=mode,
            mean_gap_pct=None if failed else self.scores.get(code, 1.0),
            status="failed" if failed else "ok", failures=0,
            indices=manifest["indices"],
            code_sha256=hashlib.sha256(code.encode()).hexdigest(),
            dataset_sha256=manifest["dataset_sha256"], elapsed_seconds=0.0,
            stdout_path="fake", error="fake failure" if failed else None)
```

```python
def test_initialization_uses_elites_but_scores_on_target_validation():
    from deployment_fakes import FakeLLM, FakeEvaluator, HEURISTIC
    from utils.checkpoint import Elite
    from inference import initialize_target_population
    evaluator = FakeEvaluator({HEURISTIC: 1.0})
    llm = FakeLLM()
    elites = (Elite(20, "reuse this elite idea", HEURISTIC, "fixture-hash"),)
    pop = initialize_target_population(
        500, elites, llm,
        lambda code, idea: evaluator.evaluate(code, 500, "val"),
        HEURISTIC, seed_count=1, max_attempts=2,
    )
    assert pop.get_best_solution("tsp_gls-500")["utility"] == 1.0
    assert evaluator.calls == [(HEURISTIC, 500, "val")]
    assert "reuse this elite idea" in llm.messages[0]
```

Add `test_reference_selection_is_deterministic`, `test_no_training_numeric_scores_in_prompt`, `test_duplicate_seeds_do_not_satisfy_requested_count`, `test_generation_failure_respects_attempt_limit`, `test_partial_population_can_proceed`, `test_empty_population_fails_target`, and `test_shared_solution_files_are_never_read`.

- [ ] **Step 2: Run RED.** `uv run pytest -q MoH/tests/test_target_initialization.py`; expected missing initialization/reference modules.
- [ ] **Step 3: Implement deterministic context and finite initialization loop.** Sort elites by source size/code hash, select references with `random.Random(seed)` without replacement, and prepend the context to both single and batch prompts. Use `for attempt in range(max_attempts)`, extract/compile heuristic syntax without execution, evaluate on target validation, and save only unique successful candidates. Do not apply inherited training thresholds. Break once actual `Pop.get_subtask_size()` reaches `seed_count`; raise a clear target initialization error if still empty. Record partial initialization when at least one valid seed exists.
- [ ] **Step 4: Run GREEN.** Repeat Step 2; expected all pass with FakeLLM and FakeEvaluator.
- [ ] **Step 5: Commit exact Task 3 files.** Message: `feat: initialize target heuristics using training elite references`.

### Task 4: Direct inference with trustworthy population updates

**Files:** Extend `MoH/inference.py`; create `MoH/tests/test_inference.py`.

**Interfaces:**
- Produce `InferenceRunner(checkpoint, llm, evaluator, output_dir, heuristic_format, iterations=10, population_size=10, seed_count=5, initialization_attempts=25, reference_count=2, max_evaluations=1000, optimizer_timeout=3600, seed=0, program_factory=GeneratedImprover)`.
- Produce `run_target(size: int) -> SelectedHeuristic`; raises a target-specific initialization/failure error only when no valid heuristic exists.
- The runner owns one per-target session with `evaluate_candidate(code, idea) -> EvalResult`, population and score cache. Cache keys include size, mode, code hash, evaluator seed, dataset fingerprint and solver settings. Parent callbacks save every successful evaluation.
- During initialization the callback caches evaluations and the initializer fills the returned `Pop`; bind that exact population as the session's population before inference. Define `utility(code, idea=None, task=None) -> float` to call `evaluate_candidate`, returning its finite gap on success or `1e6` on a candidate failure. A budget exhaustion raises a private `BudgetExhausted` exception; check the budget before invoking the evaluator, then stop the target loop after the generated invocation returns or times out.

- [ ] **Step 1: Write RED regression tests.** Use a fake program factory with the exact GeneratedImprover call signature; never execute returned code in the main process. Add one real GeneratedImprover/FakeLLM integration test in a child.

```python
def test_inference_passes_heuristic_population_and_never_changes_driver(tmp_path):
    from deployment_fakes import FakeLLM, FakeEvaluator, OPTIMIZER, HEURISTIC
    from utils.checkpoint import TrainingCheckpoint, Elite
    from inference import InferenceRunner
    observed = []
    class FakeProgram:
        def __init__(self, code, timeout, seed):
            assert code == OPTIMIZER
        def __call__(self, population, utility, language_model, function_format, task):
            observed.append(task)
            score = utility(HEURISTIC, "candidate", task)
            return "candidate", HEURISTIC, -999  # Deliberately false score.
    checkpoint = TrainingCheckpoint(OPTIMIZER,
        (Elite(20, "reference", HEURISTIC, "fixture-hash"),), {})
    runner = InferenceRunner(checkpoint, FakeLLM(), FakeEvaluator({HEURISTIC: 1.0}),
        tmp_path, HEURISTIC, iterations=10, seed_count=1, program_factory=FakeProgram)
    selected = runner.run_target(500)
    assert observed == ["tsp_gls-500"] * 10
    assert selected.validation.mean_gap_pct == 1.0
    assert checkpoint.code == OPTIMIZER
```

Add `test_unreturned_evaluated_candidate_can_win`, `test_returned_unscored_code_is_evaluated`, `test_child_population_mutations_are_not_trusted`, `test_timeout_keeps_population_and_continues`, `test_malformed_return_keeps_population`, `test_nonfinite_or_failed_evaluations_are_not_selected`, `test_validation_cache_is_target_specific`, `test_budget_includes_initialization_and_never_overshoots`, and `test_iteration_seeds_are_stable_and_distinct`.

- [ ] **Step 2: Run RED.** `uv run pytest -q MoH/tests/test_inference.py`; expected missing runner behavior.
- [ ] **Step 3: Implement the runner and target session.** The algorithm must follow this shape:

```python
for iteration in range(self.iterations):
    child_seed = int.from_bytes(hashlib.sha256(
        f"{self.seed}:{size}:{iteration}".encode()).digest()[:4], "big")
    program = self.program_factory(self.checkpoint.code,
        timeout=self.optimizer_timeout, seed=child_seed)
    try:
        returned = program(population, utility, reference_llm,
                           self.heuristic_format, f"tsp_gls-{size}")
        idea, code, supplied_score = returned
        if code:
            evaluate_candidate(code, idea)  # Cache hit or trusted validation.
    except Exception as error:
        logger.warning("Target %s iteration %s failed: %s", size, iteration, error)
    save_iteration_snapshot()
    if evaluation_budget_exhausted():
        break
```

Define the shown closures inside `run_target`; snapshot exact code hashes, evaluation counts, best validation score, failure and stop reason. Do not invoke `MoH.meta_utility`, `MoH.try_improvement`, `MoH.read_base_algorithm`, or create an optimizer population. Validate `task` passed to the utility against the target, reject cross-task requests, and ignore `supplied_score`. Require a finite successful evaluator result before population insertion; cache candidate failures but never select them. Choose the smallest successful validation result across all evaluated candidates, not just the program's final return. Write run-scoped population snapshots.
- [ ] **Step 4: Run GREEN.** Repeat Step 2 plus target initialization tests; expected all pass, including the real child-process integration case.
- [ ] **Step 5: Commit exact Task 4 files.** Message: `feat: deploy learned optimizer directly for target heuristic search`.

### Task 5: Freeze selected heuristics and report independent test scores

**Files:** Create `MoH/utils/deployment_report.py`, `MoH/tests/test_deployment_report.py`.

**Interfaces:**
- Produce `save_selected_heuristic(selected: SelectedHeuristic, output_dir: Path) -> Path` writing `heuristics/tsp<size>.py` and `heuristics/tsp<size>.json`.
- Produce `evaluate_selected_heuristics(selected: list[SelectedHeuristic], evaluator: HeuristicEvaluator, output_dir: Path, checkpoint: TrainingCheckpoint, target_errors: dict[int, str] | None = None, evaluate_test: bool = True) -> list[dict]` and `write_report(rows: list[dict], output_dir: Path) -> None`.
- Report fields: `size`, `status`, `validation_gap_pct`, `test_gap_pct`, `val_n_instances`, `test_n_instances`, `test_failures`, `elapsed_seconds`, `heuristic_sha256`, `checkpoint_sha256`, `val_dataset_sha256`, `test_dataset_sha256`, `seed`, `error`. JSON also contains selection metadata, iteration/evaluation counts and early-stop reason from Task 4's snapshots.

- [ ] **Step 1: Add RED tests with FakeEvaluator and a poison LLM object.** The report functions must not accept or access an LLM.

```python
def test_test_scores_never_change_validation_selection(tmp_path):
    from deployment_fakes import FakeEvaluator, HEURISTIC, OPTIMIZER
    from inference import SelectedHeuristic
    from utils.checkpoint import TrainingCheckpoint
    from utils.deployment_report import evaluate_selected_heuristics
    evaluator = FakeEvaluator({HEURISTIC: 2.0})
    validation = evaluator.evaluate(HEURISTIC, 500, "val")
    selected = SelectedHeuristic(500, "chosen", HEURISTIC, validation)
    rows = evaluate_selected_heuristics([selected], evaluator, tmp_path,
                                      TrainingCheckpoint(OPTIMIZER, (), {}))
    assert evaluator.calls[-1] == (HEURISTIC, 500, "test")
    assert selected.code == HEURISTIC
    assert rows[0]["validation_gap_pct"] == validation.mean_gap_pct
```

Add `test_test_infrastructure_failure_is_reported`, `test_selected_artifact_hash_matches_evaluated_code`, `test_report_records_actual_instance_count`, `test_failed_target_does_not_hide_successful_target`, and `test_json_and_csv_have_matching_values`.
- [ ] **Step 2: Run RED.** `uv run pytest -q MoH/tests/test_deployment_report.py`; expected missing report module.
- [ ] **Step 3: Implement immutable selection artifacts and one test evaluation per successful target.** Serialize selection metadata before test; never update its validation score from a test result. A failed test sets `status="test_failed"`, null test gap, and explicit error; a failed target initialization adds `status="inference_failed"` and null scores. Include partial per-instance solver failures and penalties without falsely labeling them zero-gap success. Compute checkpoint/code hashes from actual bytes and write JSON/CSV with ordinary native serializers.
- [ ] **Step 3a: Implement validation-only reporting.** If `evaluate_test=False`, save selected artifacts and set `status="validation_only"`, `test_gap_pct=None`, `test_n_instances=0`; do not call `evaluator.evaluate(..., "test")` or access a test dataset. Add `test_validation_only_report_never_evaluates_test` and run it RED then GREEN with the Task 5 module.
- [ ] **Step 4: Run GREEN.** Repeat Step 2; expected all reports deterministic except measured runtime.
- [ ] **Step 5: Commit exact Task 5 files.** Message: `feat: report frozen deployment heuristics on independent test data`.

### Task 6: Deployment CLI, preflight, documentation and offline smoke

**Files:** Create `MoH/deploy.py`, `MoH/cfg/deployment.yaml`, `MoH/tests/test_deploy_cli.py`; modify `MoH/README.md`.

**Interfaces:**
- Produce provider-independent `run_deployment(cfg, llm, evaluator=None, checkpoint=None) -> list[dict]`, which preflights all target datasets/checkpoint before initializing any target.
- Produce Hydra `main(cfg)` that constructs one heuristic LLM using existing configuration and invokes `run_deployment`. Do not instantiate a meta LLM.
- CLI config consumes the APIs defined in Tasks 1–5. `run_deployment` catches target-specific errors, processes other targets, then calls the report stage.

- [ ] **Step 1: Write RED CLI tests.** Inject FakeLLM/FakeEvaluator, temporary checkpoints, and temporary dataset files. Cover `test_paths_resolve_from_original_cwd`, `test_bad_checkpoint_prevents_llm_calls`, `test_missing_test_data_prevents_llm_calls`, `test_two_target_run_exports_both_reports`, and `test_cli_config_can_request_128_without_silent_reduction`. Mock Hydra's original cwd for the path test and assert `llm.messages == []` on all preflight failures.
- [ ] **Step 2: Run RED.** `uv run pytest -q MoH/tests/test_deploy_cli.py`; expected missing deployment entrypoint.
- [ ] **Step 3: Add the config and wiring.** Use the following actual config content:

```yaml
defaults:
  - llm_client@heu: local
  - override hydra/output: local
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog
  - _self_
checkpoint: null
target_sizes: [500, 1000]
inference_iterations: 10
population_size: 10
seed_count: 5
initialization_attempts: 25
reference_count: 2
val_n_instances: 8
test_n_instances: 100
max_evaluations_per_target: 1000
seed: 0
optimizer_timeout: 3600
instance_timeout: 90
evaluation_timeout: null
workers: 8
evaluate_test: true
hydra:
  job:
    name: moh-deployment
    chdir: true
```

Require `checkpoint` before client instantiation. Resolve its relative path with `hydra.utils.to_absolute_path`; derive the project root from `Path(__file__).resolve().parent`. Build the evaluator, verify checkpoint and dataset counts, and call `GeneratedImprover(checkpoint.code, timeout=cfg.optimizer_timeout, seed=cfg.seed).validate()` in a subprocess before instantiating the LLM with run-local raw response logging. Read the existing TSP heuristic format from `prompts/tsp_gls/desc.txt`. Validate positive budgets/counts/timeouts, supported sizes, and `population_size >= seed_count`; pass `reference_count` through to initialization and iteration wrappers. Pass `evaluate_test` to preflight and the report stage so validation-only smoke runs read no test files. Implement guarded module entrypoints to remain safe under multiprocessing spawn.

- [ ] **Step 4: Document these future commands once they exist.** Conversion and deployment are separate steps:

```bash
uv run --directory MoH python -m utils.checkpoint --run /absolute/training/run --output /absolute/new/checkpoint
uv run --directory MoH python deploy.py checkpoint=/absolute/new/checkpoint 'target_sizes=[500]' inference_iterations=1 evaluate_test=false
uv run --directory MoH python deploy.py checkpoint=/absolute/new/checkpoint 'target_sizes=[500,1000]' inference_iterations=10 test_n_instances=128
```

The last two commands call external LLMs and are documented for the user, not executed as implementation validation. The 128 command requires 128 test instances for each target. Explain that checkpoint scores are training provenance and test outcomes cannot modify selection. Document live-run cost separately from offline tests.

- [ ] **Step 5: Run GREEN and offline smoke.** `uv run pytest -q MoH/tests/test_deploy_cli.py`; expected pass with injected fake services. `uv run --directory MoH python deploy.py --cfg job` must print deployment config without calling APIs. Add an end-to-end pytest subprocess smoke using a temporary checkpoint, FakeLLM, FakeEvaluator, one target and one iteration; assert a selected artifact/report and unchanged checkpoint hash.
- [ ] **Step 6: Run final verification and compare baseline.** Run `uv run ruff check .` and `uv run pytest -q`. Run Ruff on newly added files separately; expect no new violations. Full Ruff may still fail on existing files; list baseline violations separately and do not hide them through broad ignores. Full pytest must pass, including updated explicit-count tests. Run `git diff --check` on changed implementation/documentation files and review the diff for shared-file writes, leaked training scores and accidental outer-loop reuse.
- [ ] **Step 7: Commit exact Task 6 files.** Message: `feat: add checkpoint-driven MoH deployment CLI`.

## Completion contract

- A checkpoint carries the final active optimizer and elite ideas/code from actual training tasks.
- Inference calls that same program directly on new heuristic populations for the configured iterations or reports an explicit budget stop.
- All target scores are evaluator-owned validation scores; test is independent and cannot choose a winner.
- Artifacts/logs contain actual counts, selected indices, hashes, seed and failure reasons.
- A target or candidate failure does not erase other successful results.
- No generated code runs in the main process; no implementation test calls an external LLM API.
- No live experiment is claimed complete from the offline smoke test.

## Execution handoff

Review this plan and its design together. Recommend inline/native implementation
because all six tasks share the checkpoint/evaluation interfaces and the user is
already working on the dedicated deployment branch. If the user chooses task
delegation instead, use fresh implementers/reviewers only after that choice.
No implementation or live deployment is authorized by this planning task alone.

## Plan self-review record

| Spec requirement | Owning tasks |
|---|---|
| Active final program, elites, hashes, legacy conversion | 1 |
| Independent splits, explicit 100/128 counts, isolated generated evaluation | 2 |
| Elites in initial and modification prompts; fresh target scores | 3, 4 |
| Fixed direct driver, callback-owned population, budgets/timeouts | 4 |
| Frozen validation winner; independent test and failure reports | 5 |
| Preflight before API calls, Hydra paths, CLI and offline smoke | 6 |

Self-review resolved interface omissions for `reference_count`, evaluator
manifest shape, target utility return type and validation-only reports. Review
Focus items each have a named regression test in their owning task. The plan
contains no implementation placeholders and changes no existing user code.
