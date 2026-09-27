# Full MoH Experimental Protocol and Fair Budget Contract (M7A / M7B-A)

## 0. Executive Summary

M7A froze the experimental accounting and reproducibility contract before any HiFo mechanism is implemented.

M7B-A connects the frozen experimental protocol to the frozen Full-MoH implementation without executing a real benchmark.

The M6D pilot validates execution correctness; it is not reused as a paper-quality result.

Future Full-MoH and HiFo variants must use the same declared work opportunities and task instances unless an experiment explicitly studies that difference.

Logical candidate evaluations and underlying task-instance evaluations remain separate work domains.

Official paper experiments remain blocked until every predeclared campaign-scale and artifact-retention field is resolved.

No Hindsight, Foresight, or cross-level experience mechanism was implemented during M7B-A.

---

## 1. Baseline Historical Validation Facts

- **Code Checkpoint**: `9979c02ce8df37466595fe223f4887cf19c47bd2`
- **M7A Checkpoint Tag**: `full-moh-m7a` (`d501d324656a09954f56f0874592e58fc43b9993`)
- **Frozen Tags**: `full-moh-m6b-fix`, `full-moh-real-path-v1`, `full-moh-real-baseline-v1`
- **M6D Validation Status**: `FULL_BASELINE_VALIDATION_SUCCESS`
- **M6D Evidence SHA256**: `2f315bb4d66966197e0b61566880be7d3165b3917e34bca6b7583f1a97af6e5e`

---

## 2. Production Semantics Audit (M7B-A)

1. **Offspring Generation Rate**: Source inspection of `run_outer_evolution()` confirms that each outer generation proposes and evaluates exactly **1** offspring optimizer program.
2. **Initial Population Provenance**: For `population_size = 2`, initial `OptimizerProgram` objects originate from deterministic pre-supplied template functions (`o000001` and `o000002`). Initial population generation consumes **0** outer meta-LLM calls.
3. **Task Instance Evaluation Domain**: `HeuristicRunner.evaluate()` evaluates each candidate heuristic across all $K$ deterministic instances of a task ($K = 3$ for `TSP10`). One logical inner evaluation request executes $K$ worker sandbox evaluations.
   $$\text{Max Task-Instance Evaluations} = \text{Max Inner Evaluate Requests} \times K$$
   For $P=2, G=2, N_{eval}=5, K=3$:
   $$\text{Max Task-Instance Evaluations} = 20 \times 3 = 60$$

---

## 3. Budget & Campaign Formulas

Let $P$ be `population_size`, $G$ be `generations`, $N_{gen}$ be `max_inner_generate_requests`, $N_{eval}$ be `max_inner_evaluate_requests`, $K$ be `instances_per_task`, and $A$ be `provider_attempt_limit_per_request`.

$$\text{Max Optimizer Executions} = P + G$$

$$\text{Max Outer Meta Requests} = G$$

$$\text{Max Inner Generate Requests} = (P + G) \times N_{gen}$$

$$\text{Max Inner Evaluate Requests} = (P + G) \times N_{eval}$$

$$\text{Max Task-Instance Evaluations} = (P + G) \times N_{eval} \times K$$

$$\text{Max Provider Attempts (per run)} = (G \times A) + \left((P + G) \times N_{gen} \times A\right)$$

### Current Baseline Configuration ($P=2, G=2, N_{gen}=5, N_{eval}=5, K=3, A=1$)
- Max Optimizer Executions = 4
- Max Outer Meta Requests = 2
- Max Inner Generate Requests = 20
- Max Inner Evaluate Requests = 20
- Max Task-Instance Evaluations = 60
- Max Provider Attempts (per run) = 22

### Campaign Budget Ceilings across $R$ Replicates
$$\text{Campaign Max Provider Attempts} = R \times \text{Max Provider Attempts (per run)}$$
- **Sanity Campaign ($R = 3$)**: $3 \times 22 = 66$ provider attempts.
- **Official Paper Campaign ($R = \text{TBD}$)**: Pending predeclared replicate count.

---

## 4. Replication & Seeding Policy

- **Task Root Seed** (`root_seed = 42`): Drives deterministic problem instance coordinate generation.
- **Replicate Seed** (`replicate_index = 0, 1, ...`): Drives algorithm-level stochastic choices if applicable.
- **LLM Nondeterminism Policy**: Model `ag/gemini-3.6-flash-low` over HTTP does not guarantee exact text reproducibility from seed alone. Scientific reproducibility relies on fixed request schemas, prompt fingerprints, code checkpoint SHA, preserved program/candidate artifacts, and repeated trials across predeclared replicate counts.

---

## 5. Artifact & Trajectory Preservation

Every run saves:
- Atomic `manifest.json` with SHA256 config hash, prompt fingerprints, work counts, token counts, and safety validation.
- Outer `OptimizerProgram` source code files saved as content-addressed files under `programs/<program_id>.py`.
- Raw trajectory log `trajectory.json` recording facts (event names, payloads, scores, evaluation status) without HiFo interpretations (no hindsight labels, verbal gradients, or stagnation flags).

---

## 6. Paper-Quality Gates

- **Sanity Baseline Campaign Ready**: **YES** (Controlled dry-run harness validated, $R=3$ sanity campaign budgeted at 66 provider attempts).
- **Official Paper Baseline Campaign Ready**: **NO** (Blocked pending declaration of official replicate count $R$, resolution of task scale sizes 20/50, and final campaign locking).
