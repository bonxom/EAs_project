# Full MoH Development Baseline Freeze (`full_moh_dev_baseline_v1`)

> [!IMPORTANT]
> **CLASSIFICATION**: Development Sanity Baseline Only (`baseline_level = development_sanity`).
> **PAPER GATE**: **CLOSED**. This baseline establishes infrastructure correctness and execution accounting for the Full-MoH two-level optimization pipeline on TSP10 (N=3). It is **not** a paper-scale evaluation and does **not** include CVRP experiments or HiFo implementation.

---

## 1. Executive Summary & Freeze Identity

- **Implementation Identity Checkpoint**: Tag `full-moh-real-e2e-v1` at commit `28d4d7fe29cd2341dbb9c70336d2a49cc01a7407` (tag object `985c3ea243c7c8b0e4abc85798b21a90b9984344`).
- **Development Baseline Freeze Identity**: `full_moh_dev_baseline_v1` (tag `full-moh-dev-baseline-v1`), created as a direct docs-only child commit on top of `28d4d7fe29cd2341dbb9c70336d2a49cc01a7407`.
- **Governed Campaign Identity**: `campaign_full_moh_5faa7660` across 3 replicates (`run_full_moh_5faa7660_rep00`, `run_full_moh_368b2eba_rep01`, `run_full_moh_190c157a_rep02`).
- **Canonical Semantic Config Hash**: `5faa7660a6494103f4b814b9d10c3d062baae9d58ff2dbd26884739af7cc0aa7` (`configs/full_moh_sanity.yaml`, file SHA256 `522afaa2613f62ce934808dcbd5a865cea55337728d6ee630d9bc8bb228be9bc`).

---

## 2. Test Environment & Canonical Evidence

Canonical test suite execution was independently verified on Linux / WSL2 using the project virtual environment:

- **OS / Environment**: Linux / WSL2 (`sys.platform = linux`, `os.name = posix`, `platform.system = Linux`)
- **UV Executable**: `/home/ducanh/.local/bin/uv`
- **Python Executable**: `/home/ducanh/projects/EAs_project/.venv/bin/python`
- **Pytest Results**: **564 collected, 564 passed, 0 failed, 0 skipped** (duration 97.02s, exit code 0)
- **Pytest Execution Log SHA256**: `106bf8813ac72e7c8ac6f63a4a75504be3927f3ac6edeb37626c0b50f8beb689` (`/tmp/m7bp2_full_pytest.out`)
- **Ruff Check**: **All checks passed** (exit code 0)

---

## 3. Protocol & Semantics Specification

- **Problem & Scale**: TSP10, K=3 instances per optimizer evaluation, root seed = 42.
- **Outer Topology**: Steady-state \((\mu + 1)\), \(\mu = 2\), 2 generations, 3 replicates.
- **Inner Capability Budget**: `inner_max_generate = 5`, `inner_max_evaluate = 5`.
- **Model & Routing**: `ag/gemini-3.6-flash-low`, OpenAI-compatible `chat_completions`, reflection routing -> `TEXT`, mutation/crossover routing -> `CANDIDATE`.
- **Token Limits**: Outer `max_output_tokens = 2048`, Inner `max_output_tokens = 2048`.
- **Timeouts**: Provider HTTP = 30.0s, Optimizer compute = 5.0s, Evaluator = 5.0s, Global wall timeout = 500.0s.
- **SDK Retries**: 0 retries; 1 application attempt per logical request.
- **Utility Semantics**: Utility is defined as the maximum valid numeric inner score (finite negative scores valid). `utility = None` only if no valid numeric score exists. Exact score ties keep the first valid occurrence.
- **Outer Selection**: Steady-state \((\mu + 1)\). Higher numeric utility ranks above `None`. Exact utility ties are broken deterministically by `program.id`.
- **Candidate Contract**: Strict candidate contract — no web scraping, no automated repair, no hidden retries.

---

## 4. Reconciled Campaign Results & Metrics

- **Generate Requests**: 27 inner generate requests (9 reflection `TEXT`, 18 `CANDIDATE` intent).
- **Candidate Generation**: 17 valid provider candidate generations, 1 invalid provider candidate generation (success rate = 17 / 18).
- **Evaluate Requests**: 17 inner evaluate requests (14 valid numeric evaluations, 3 invalid evaluations).
- **Task-Instance Evaluations**: 51 task-instance evaluations (\(17 \times 3\)).
- **Optimizer Executions**: 12 total executed optimizers (10 with valid numeric utility, 2 with `None` utility).
- **k Distribution**: \(k_0 = 2\), \(k_1 = 6\), \(k_2 = 4\), \(k_{\ge 3} = 0\) (Mean \(k = 1.1666666667\), Median \(k = 1.0\), Max \(k = 2\)).
- **Offspring Metrics**: 6 total offspring, 4 with valid numeric utility, 2 immediate survival, 2 final survival.
- **Replicate Winners**:
  - `rep0`: `o000002` (utility = `-3.0954547423176586`)
  - `rep1`: `o000002` (utility = `-3.0954547423176586`)
  - `rep2`: `o000002` (utility = `-3.0954547423176586`)
- **Failure Taxonomy**: `candidate_contract_invalid` = 1, `invalid_return` = 1, `evaluator_exception` = 2.
- **Provider Attempts**: Outer = 6, Inner = 27, Total = 33 / 66 hard ceiling (50.0% realized).
- **Token Usage**:
  - Outer: Input = 15,164, Output = 2,250, Reasoning = 196, Total = 17,414
  - Inner: Input = 73,489, Output = 16,492, Reasoning = 7,385, Total = 89,981
  - Combined: Input = 88,653, Output = 18,742, Reasoning = 7,581, Total = 107,395

---

## 5. Forensic Corrections & Historical Log Qualification

1. **Winner Linkage**: Historical prose claiming `o000001` as winner was corrected to `o000002` (`REPORT_ONLY_LABEL_ERROR`).
2. **Candidate Intent vs TEXT Requests**: Reconciled to 18 candidate intent requests and 9 reflection TEXT requests (summing to 27 total generate requests).
3. **Evaluate Request Accounting**: Reconciled to 17 evaluate requests (14 valid numeric + 3 invalid), giving \(17 \times 3 = 51\) task instance evaluations.
4. **Candidate `a937` Attribution**: Candidate `a937ae6a1159...` is a `STATIC_BASELINE_CANDIDATE` artifact, not generated or evaluated by Rep0 `o000004` (`REPORT_SCRIPT_MISATTRIBUTION`). Rep0 `o000004` generated candidate `985e1137...` which encountered an evaluator runtime exception, terminating the worker without evaluating `a937`.
5. **Runtime Log Qualification**: Historical summary counters in `/home/ducanh/moh_runtime_evidence/m7bm_full_moh_n3_campaign.out` (e.g. `inner_evaluate_requests = 18`, `task_instance_evaluations = 54`) are un-reconciled diagnostic outputs. Canonical event-ledger accounting is authoritative.

---

## 6. HiFo Raw Signal Readiness & Paper Gate

- **Raw Signal Availability**:
  - `outer_foresight_population_utility_trajectory` = available
  - `inner_foresight_chronological_candidate_evaluation_trajectory` = available
  - `outer_hindsight_optimizer_source_to_utility` = available
  - `inner_hindsight_candidate_source_origin_to_score_error` = available
  - `cross_level_linkage` = available
- **HiFo Algorithm Implementation**: **UNIMPLEMENTED**. No Hindsight or Foresight optimization logic was modified or added.
- **Paper Gate**: **CLOSED**.
