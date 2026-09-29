# Dual-HiFo State / Event / Experience Schema v1

## Status

M8-A is a schema-only milestone.

It introduces representational contracts required for later Hindsight and
Foresight work. It does **not** implement a HiFo control policy, modify
Full-MoH behavior, add provider calls, generate verbal gradients, derive
semantic diversity, assign credit, extract insights, reuse insights, or prune
insights.

The frozen Full-MoH baseline remains the no-HiFo reference.

## 1. Experimental modes

The project represents five controlled experimental modes:

- `FULL_MOH`
- `INNER_HIFO`
- `OUTER_HIFO`
- `DUAL_HIFO_INDEPENDENT`
- `DUAL_HIFO_CROSS_LEVEL`

`FULL_MOH` explicitly enables no Hindsight or Foresight mechanism.

`INNER_HIFO` enables only the inner-level mechanisms.

`OUTER_HIFO` enables only the outer-level mechanisms.

`DUAL_HIFO_INDEPENDENT` enables both levels without cross-level trajectory
transfer.

`DUAL_HIFO_CROSS_LEVEL` enables both levels and permits later outer-level
consumption of an inner trajectory summary.

M8-A only represents these modes. It does not route runtime behavior through
them.

## 2. Source-supported HiFo concepts

The project source material motivates the following concepts:

### Foresight

- progress counter \(C_{\text{prog}}\)
- stagnation counter \(C_{\text{stag}}\)
- semantic diversity \(\Delta_p(t)\)
- `EXPLORE`
- `EXPLOIT`
- `BALANCE`
- verbal gradients

### Hindsight

- insight distillation
- effectiveness
- usage penalty
- recency bonus
- reuse
- pruning

### Two levels

Inner HiFo concerns the heuristic/candidate search trajectory.

Outer HiFo concerns the population of Heuristic-Optimizer programs and their
outer outcomes.

These concepts motivate the schemas, but M8-A does not freeze their numerical
algorithms.

## 3. Project engineering contracts

The following are engineering contracts introduced by this repository rather
than source-mandated equations.

### `EvidenceRef`

A deterministic pointer to raw chronological evidence.

It can identify:

- run
- HiFo level
- logical event sequence
- logical step
- optimizer program
- candidate source hash
- generation kind
- event kind
- failure code

A physical artifact is not treated as a capability event solely because the
artifact exists. Later reconstruction must follow raw event evidence.

No random UUID or required wall-clock timestamp is generated.

### `ProgressObservation`

Stores caller-supplied:

- logical step
- progress counter
- stagnation counter
- current best value
- previous best value
- last improvement step

It does not update those counters.

### `DiversityObservation`

Separates two evidence domains:

1. exact / structural duplication
2. semantic diversity

`exact_duplicate_rate` is not renamed or reinterpreted as semantic diversity.

M8-A does not derive:

`semantic_diversity = 1 - exact_duplicate_rate`

or any equivalent proxy.

### `CapabilityBudgetObservation`

Represents an observation of existing Full-MoH accounting:

- generate used / limit
- evaluate used / limit
- optional provider attempts used / limit

It does not redefine budget semantics.

### `ForesightState`

Combines policy-neutral observations:

- progress
- diversity
- budget
- recent failure codes
- evidence references
- active insight IDs

### `ForesightDecision`

Makes a later decision representable:

- level
- logical step
- `EXPLORE`, `EXPLOIT`, or `BALANCE`
- reason codes
- evidence references
- optional verbal gradient

M8-A provides no decision engine.

### `CreditComponents`

Represents separately:

- effectiveness
- usage penalty
- recency bonus
- optional aggregate credit
- optional credit-rule version

No aggregation equation is executed by this schema.

### `InsightRecord`

Represents a distilled insight and its lifecycle metadata:

- stable insight ID
- level
- abstract insight
- evidence
- credit components
- usage count
- logical creation / last-use steps
- `ACTIVE` or `PRUNED` status
- tags / mechanism labels

No extraction, ranking, reuse, credit update, or pruning behavior is included.

## 4. Inner trajectory contract

`InnerEvaluationPoint` represents one chronological inner evaluation outcome.

A valid point may contain a finite numeric score.

A failed point may instead contain an error code and no numeric score.

`InnerTrajectorySummary` can represent:

- optimizer identity
- optimizer source identity
- ordered evaluation points
- generate request count
- evaluate request count
- valid numeric evaluation count
- invalid evaluation count
- candidate source hashes
- failure-code counts
- generation-intent counts
- best valid score
- final optimizer utility
- capability budget snapshot

The schema supports:

- no-valid-evaluation trajectories
- successful trajectories
- failed trajectories
- mixed trajectories

M8-A does not reconstruct these summaries from campaign artifacts. That is a
later milestone.

## 5. Outer trajectory contract

`OuterGenerationSummary` represents one policy-neutral outer generation
snapshot:

- parent program IDs
- parent utilities
- offspring program ID / utility
- surviving program IDs
- best utility after selection
- optional diversity observation
- raw evidence references

Only one `DiversityObservation` is stored because that object already keeps
exact duplication and semantic diversity as distinct fields. Maintaining two
parallel diversity snapshots would allow contradictory state.

No outer-selection rule is changed.

## 6. Project research extension: cross-level experience

`CrossLevelExperience` is a project research extension.

It links:

- an outer optimizer
- its detailed inner trajectory
- its final utility
- its outer survival outcome

The future research question is whether Outer HiFo benefits from receiving
information about *how* an optimizer obtained its utility rather than only the
final scalar utility.

Potential later signals include:

- improvement trajectory
- stagnation
- successful candidate/operator behavior
- failed candidate/operator behavior
- generation intent
- generate/evaluate usage
- failure taxonomy
- duplication/diversity evidence
- provider/token usage
- final utility
- survival

M8-A only makes this linkage representable. It does not implement cross-level
credit assignment.

## 7. Explicitly deferred research decisions

The following remain intentionally unresolved after M8-A:

- exact \(C_{\text{prog}}\) update rule
- exact \(C_{\text{stag}}\) update rule
- improvement tolerance
- semantic-diversity representation
- semantic-diversity metric
- embedding model, if any
- Explore threshold
- Exploit threshold
- Balance threshold
- verbal-gradient generation policy
- exact effectiveness definition
- usage-penalty equation
- recency-bonus equation / decay
- aggregate credit equation
- reuse threshold
- pruning threshold
- insight ranking policy
- cross-level credit equation
- auxiliary HiFo provider-call budget

These must not be silently inferred from field names.

## 8. Deterministic serialization

`to_json_dict` recursively converts contracts to JSON-compatible values.

Properties:

- enum values use stable string values
- tuples become arrays
- dataclass fields are emitted in stable sorted order
- mapping keys must be strings
- non-finite floats are rejected
- no timestamp is inserted
- no random identifier is generated
- source objects are not mutated

This serialization is representation only. It is not HiFo inference.

## 9. Future equal-budget ablations

Planned comparison family:

- Full MoH
- Inner HiFo
- Outer HiFo
- Dual HiFo Independent
- Dual HiFo Cross-Level

Future scientific comparisons should control at least:

- base model
- task instances
- root seeds
- outer topology
- inner capability limits
- task-evaluation budget

Any additional HiFo cost must be explicitly accounted for:

- provider requests
- provider attempts
- prompt/input tokens
- completion/output tokens

M8-A introduces zero provider requests.

## 10. Non-interference requirement

When HiFo is not integrated into runtime code, the frozen Full-MoH baseline
remains unchanged.

M8-A adds only schema, tests, and design documentation.

No existing Full-MoH source, experiment configuration, prompt, runner,
selection logic, accounting logic, or frozen baseline document is modified.
