# Full MoH Experimental Protocol and Fair Budget Contract (M7A)

## 0. Executive Summary

M7A freezes the experimental accounting and reproducibility contract before any HiFo mechanism is implemented.

The M6D pilot validates execution correctness; it is not reused as a paper-quality result.

Future Full-MoH and HiFo variants must use the same declared work opportunities and task instances unless an experiment explicitly studies that difference.

No Hindsight, Foresight, or cross-level experience mechanism was implemented during M7A.

---

## 1. Baseline Historical Validation Facts

- **Code Checkpoint**: `9979c02ce8df37466595fe223f4887cf19c47bd2`
- **Frozen Tags**: `full-moh-m6b-fix`, `full-moh-real-path-v1`, `full-moh-real-baseline-v1`
- **M6D Validation Status**: `FULL_BASELINE_VALIDATION_SUCCESS`
- **M6D Evidence SHA256**: `2f315bb4d66966197e0b61566880be7d3165b3917e34bca6b7583f1a97af6e5e`

---

## 2. Fairness Contract & Equivalence Principles

All comparisons between Full MoH baseline and future HiFo (Inner-HiFo, Outer-HiFo, Dual-HiFo) variants must adhere to strict fairness constraints:

1. **Identical Task Instances & Seeds**: All runs evaluate on identical problem instances derived deterministically from fixed `root_seed` and task parameters.
2. **Identical Model & Endpoint**: Target model (`ag/gemini-3.6-flash-low`), proxy (`9router` v0.5.81), and API mode (`chat_completions`).
3. **Identical Logical Work Budgets**: Same population size, same number of outer generations, same per-optimizer inner generation and evaluation limits.
4. **Identical Safety & Retry Constraints**: SDK retries strictly set to 0 (`max_retries = 0`). 1 application-to-proxy attempt allowed per logical LLM request.
5. **Identical Token Output Caps**: `outer_max_output_tokens = 2048`, `inner_max_output_tokens = 2048`.
6. **Token Accounting Policy (Policy A)**: HiFo adds context to prompts (e.g. reflections, state traces). Rather than capping input tokens artificially, comparisons grant equal logical LLM call opportunities and equal output caps, while explicitly recording and reporting separate input, output, and reasoning token usage metrics.

---

## 3. Budget Formulas

Logical work counts are the primary algorithmic work units. Provider attempts and token usage are secondary resource and safety metrics.

Let $P$ be `population_size`, $G$ be `generations`, $N_{gen}$ be `max_inner_generate_requests`, $N_{eval}$ be `max_inner_evaluate_requests`, and $A$ be `provider_attempt_limit_per_request`.

$$\text{Max Optimizer Executions} = P + G$$

$$\text{Max Outer Meta Generations} = G$$

$$\text{Max Inner Generate Requests} = (P + G) \times N_{gen}$$

$$\text{Max Inner Evaluate Requests} = (P + G) \times N_{eval}$$

$$\text{Max Provider Attempts} = (G \times A) + \left((P + G) \times N_{gen} \times A\right)$$

### Pilot Reproduction Checkpoint (M6D)
- $P = 1, G = 1, N_{gen} = 1, N_{eval} = 1, A = 1$
- Max Optimizer Executions = 2
- Max Outer Meta Generations = 1
- Max Inner Generate Requests = 2
- Max Inner Evaluate Requests = 2
- Max Provider Attempts = 3

Matches M6D runtime structure (2/1/2/2/3) exactly.

---

## 4. Failure Classification & Reporting Policy

No replicate is silently discarded. Every run produces an immutable `ExperimentManifest` classified into one of the following statuses:

1. `COMPLETED_VALID`: Run completed successfully with at least one valid optimizer evaluation yielding finite utility.
2. `COMPLETED_NO_VALID_UTILITY`: Run completed execution budget but no valid candidate heuristics passed evaluation.
3. `PROVIDER_FAILURE`: Run terminated early due to unrecoverable LLM provider/proxy errors.
4. `EXECUTION_FAILURE`: Run failed due to code sandbox execution timeout or environment failure.
5. `SAFETY_FAILURE`: Run stopped due to budget limit violation or safety check assertion.

---

## 5. Artifact & Trajectory Preservation

Every run saves:
- Canonical JSON manifest with SHA256 config hash and prompt fingerprints.
- Generated optimizer program source code stored as content-addressed files.
- Raw inner candidate evaluation metrics (scores, tour lengths, execution status) without HiFo interpretations (no hindsight labels, verbal gradients, or stagnation flags).
