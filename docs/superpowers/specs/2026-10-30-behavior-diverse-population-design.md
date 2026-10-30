# Behavior-diverse heuristic populations

Status: Ready for review before implementation planning.

## Requested outcome

Implement proposal 6: retain both high-quality and behaviorally distinct heuristic
programs in the inner-loop population. The user explicitly selected this proposal
and permits changes without preserving legacy code compatibility.

Success means candidates with complementary per-instance validation performance
can survive even when their mean gap is worse than another candidate's. The best
mean-gap candidate must always survive. Improved final solution quality is a
research hypothesis, not an implementation acceptance claim.

## Scope

Apply the policy to training subtask populations and deployment target populations.
The outer optimizer population keeps its current score-based policy. Do not change
the meta objective, budget policy, or optimizer comparison protocol. Do not revive
the reverted unified-training-evaluator project. Extend each existing evaluation
path only enough to supply validated behavior descriptors.

## Alternatives and decision

Code-text distance is cheap but does not measure solving behavior. A fixed grid of
behavior niches requires arbitrary bin boundaries. Use quality elites plus greedy
max-min distance over validation gap vectors: it is small, deterministic, and
directly tied to the proposed research question.

## Behavior descriptor

Each successfully evaluated heuristic has an ordered vector of nonnegative finite
gap percentages, one per validation instance. Its identity includes problem size,
mode, dataset SHA-256, ordered instance indices, seed, and solver settings.
Compare descriptors only within the same identity. Test results never participate.

Validate vector length, statuses, finite values, identity, and consistency of the
vector mean with the reported score using numerical tolerance. Any instance
failure makes the candidate ineligible for the diversity-enabled population;
failure penalty values must not become novelty rewards. Do not synthesize a vector
from a scalar mean or accept a descriptor supplied by generated optimizer code.

EvalResult gains the per-instance gap vector. The deployment evaluator parses it
from the existing JSON instances field. Training requests a unique result JSON
file through eval.py's existing --result-path option and validates it before
registering the descriptor. A shared pure parser/descriptor validator prevents
the two paths from interpreting behavior differently. Generated code continues
to execute only in timeout-limited subprocesses.

## Survivor selection

Configure population.diversity_fraction, default 0.3, in training and deployment.
Validate 0 <= fraction < 1. For capacity K, reserve floor(K * fraction) diversity
slots and keep K minus that count as quality elites. Capacity must be positive.
If rounding yields zero diversity slots, selection is entirely quality-based.

Deduplicate identical cleaned source code. Rank quality by mean gap ascending,
then cleaned-source SHA-256 ascending. Retain the quality elites first. For each
remaining slot, choose the remaining candidate maximizing its minimum distance
to already selected survivors. Break equal-distance ties by the quality order.

Distance is root-mean-square coordinate difference after applying log1p to each
gap percentage. This dampens extreme magnitudes while preserving absolute
performance differences; do not independently normalize each candidate vector.
Perform selection over the current survivors plus the newly evaluated candidate,
not an unbounded historical archive. Sequential arrival order can affect survivors;
identical input order and seed must reproduce selections.

Equal mean gap is not grounds for removing a behaviorally distinct candidate.
Return survivors sorted by quality so index 0 and get_best_solution retain their
meaning. Existing rank-based parent sampling remains; diverse survivors remain
available to it. A fraction of zero provides a quality-only ablation with the same
validity and tie-breaking rules.

## Evaluation-to-population flow

The parent process registers every fully successful candidate evaluated through
the inner utility callback, not only the optimizer's final returned candidate.
Without this, diversity selection would never see most search candidates.

Associate code, measured mean score, descriptor, and evaluation identity in parent
state. On final optimizer return, reuse a matching registered evaluation or evaluate
that code before admission; do not use its supplied score or behavior metadata.
This verification is scoped to heuristic admission, not outer optimizer ranking.

Initialization and deployment callbacks use the same admission policy. Generated
optimizers receive copies of the population as today; child mutations cannot
modify authoritative parent descriptors. New metadata must remain serializable
through the existing subprocess interface and snapshots.

## Persistence and configuration

Persist survivor descriptors and identity with population snapshots. Loading into
a diversity-enabled population requires matching evaluation identity and valid
descriptors. Training must reevaluate legacy or mismatched candidates on current
validation instances before admitting them; never silently use stale scores.
If none survive, use normal seed generation. No automatic deletion of old files.

Record the selection fraction and policy version in run/checkpoint metadata so
experiments can distinguish diversity runs from quality-only runs. Checkpoint
elite references remain references; target descriptors are measured on target
validation data. Configuration plumbing includes deploy.py and InferenceRunner.

## Tests required before implementation

- A lower-quality but complementary vector survives alongside the best elite;
  a redundant vector loses the diversity slot.
- Equal-mean distinct vectors can coexist; duplicate code cannot consume slots.
- Elite preservation, capacity 1, fraction 0, fractional rounding, invalid capacity
  or fraction, exact-distance ties, and repeatability for fixed insertion order.
- Missing vectors, wrong lengths, reordered indices, mismatched datasets/seeds/
  solver settings, NaN/Infinity, negative gaps, inconsistent means, test mode,
  and partial instance failures cannot enter the diverse population.
- JSON round-trip preserves descriptors and selection behavior; old snapshots
  require reevaluation before admission rather than borrowing stored scores.
- FakeLLM training and inference tests evaluate several complementary candidates
  and return only one; parent selection still sees every valid evaluated candidate.
- A generated optimizer's claimed score and child population mutation cannot
  override measured parent behavior. Candidate failure permits later success.
- Population snapshots contain policy and behavior metadata; final test evaluation
  does not alter validation selection. Existing subprocess and checkpoint tests pass.

## Implementation boundaries and verification

Use Python 3.12 and uv. FakeLLM is mandatory for tests involving LLM generation;
no unit test calls external LLM APIs. Preserve provider-independent abstractions.
Keep the user's current MoH/.gitignore edits intact. Run focused regression tests
red before implementation, green afterward, then uv run ruff check . and
uv run pytest -q. Baseline: 95 tests and 10 subtests pass; Ruff has 35 existing
findings. Report remaining findings without hiding them or expanding this feature
into an unrelated cleanup.

## Research validation after implementation

Compare fraction 0 and 0.3 with the same seeds, initial candidates, validation
instances, LLM configuration, and evaluation budget. Record best gap, population
pairwise behavior distance, and per-size outcomes. Use held-out test only after
selection. Real LLM experiments are separate from unit validation and no quality
improvement is claimed until measured.
