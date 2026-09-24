# Mini-MoH Project Plan — Build with Codex

## 0. Goal

Build a minimal but real end-to-end implementation inspired by **MoH (Meta-Optimization of Heuristics)**.

Target architecture:

```text
                         OUTER LOOP

Optimizer Population
        │
        ▼
  Meta-Optimizer + LLM
        │
        ▼
   New Optimizers
        │
        └──────────────────────────────┐
                                       │
                         INNER LOOP    │
                                       │
Optimizer O_i                          │
        │                              │
        ├── Task 1                     │
        │     │                        │
        │     ▼                        │
        │  Heuristic Population        │
        │     │                        │
        │     ▼                        │
        │  Search / Generate           │
        │     │                        │
        │     ▼                        │
        │  Best Heuristic              │
        │     │                        │
        │     ▼                        │
        │   Utility                    │
        │                              │
        ├── Task 2                     │
        │                              │
        └── Task N                     │
              │                        │
              ▼                        │
      Aggregate Optimizer Utility ─────┘
```

The implementation should make the following distinction explicit:

```text
LLM != Optimizer

LLM:
    GPT / DeepSeek / etc.

Heuristic:
    code that solves the task

Heuristic-Optimizer:
    program that searches for better heuristics

Meta-Optimizer:
    program that searches for better Heuristic-Optimizers
```

---

# 1. Tech Stack

Use:

- Python 3.12
- `uv`
- `pytest`
- `pytest-cov`
- `ruff`
- `numpy`
- `pydantic`
- `PyYAML`
- `openai`
- JSONL experiment logs
- subprocess-based execution for generated code

Do **not** add a database initially.

---

# 2. Initialize Repository

```bash
mkdir mini-moh
cd mini-moh

git init

uv init --python 3.12

uv add numpy pydantic pyyaml openai
uv add --dev pytest pytest-cov ruff

mkdir -p \
    src/moh/core \
    src/moh/problems \
    src/moh/llm \
    src/moh/execution \
    src/moh/optimizers \
    src/moh/prompts \
    configs \
    tests \
    outputs
```

Launch Codex:

```bash
codex
```

Inside Codex:

```text
/init
```

---

# 3. Target Repository Structure

```text
mini-moh/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── smoke.yaml
│   └── real_llm.yaml
│
├── src/
│   └── moh/
│       ├── __init__.py
│       ├── main.py
│       │
│       ├── core/
│       │   ├── models.py
│       │   ├── populations.py
│       │   └── protocols.py
│       │
│       ├── problems/
│       │   ├── base.py
│       │   └── tsp.py
│       │
│       ├── llm/
│       │   ├── base.py
│       │   ├── fake.py
│       │   └── openai_client.py
│       │
│       ├── execution/
│       │   ├── heuristic_runner.py
│       │   └── sandbox.py
│       │
│       ├── optimizers/
│       │   ├── inner.py
│       │   ├── seed_optimizers.py
│       │   └── meta.py
│       │
│       ├── prompts/
│       │   ├── heuristic_generation.py
│       │   └── optimizer_generation.py
│       │
│       ├── evaluation.py
│       └── logging.py
│
└── tests/
    ├── test_tsp.py
    ├── test_heuristic_runner.py
    ├── test_inner_loop.py
    └── test_outer_loop.py
```

---

# 4. Milestone 1 — Core Abstractions

## Goal

Create the basic domain objects without any real LLM calls.

Core entities:

```python
Heuristic
OptimizerProgram
Population
Task
LLMClient
EvaluationResult
```

Suggested `Heuristic` shape:

```python
@dataclass
class Heuristic:
    id: str
    source_code: str
    idea: str | None = None
    utility: float | None = None
    metadata: dict = field(default_factory=dict)
```

Suggested optimizer representation:

```python
@dataclass
class OptimizerProgram:
    id: str
    source_code: str
    utility: float | None = None
    metadata: dict = field(default_factory=dict)
```

## Codex Prompt

```text
We are building a minimal research reproduction of the MoH
(Meta-Optimization of Heuristics) architecture.

Do NOT try to implement the whole paper yet.

Goal for milestone 1:
create a clean Python package with the following abstractions:

1. Heuristic
   - id
   - source_code
   - idea
   - utility
   - metadata

2. OptimizerProgram
   - id
   - source_code
   - utility
   - metadata

3. Task protocol
   - name
   - generate_instances()
   - evaluate_heuristic()

4. LLMClient protocol
   - generate(prompt: str) -> str

5. Population generic container

Use:
- Python 3.12
- dataclasses or pydantic where appropriate
- pytest
- strong typing

Create the package structure under src/moh.

Do not implement actual LLM calls yet.

Add unit tests.
Run all tests before finishing.
```

Run:

```bash
uv run ruff check .
uv run pytest -q
```

Commit:

```bash
git add .
git commit -m "milestone 1: core abstractions"
```

---

# 5. Milestone 2 — Deterministic TSP Evaluator

## Goal

Before using an LLM, guarantee that this works:

```text
heuristic code
    ↓
TSP task
    ↓
utility
```

Use Euclidean TSP.

A heuristic must expose:

```python
def select_next_node(
    current_node,
    unvisited,
    coordinates,
) -> int:
    ...
```

Execution logic:

```text
start at node 0
    ↓
call select_next_node()
    ↓
visit returned node
    ↓
repeat until all nodes visited
    ↓
return to node 0
    ↓
calculate tour length
```

Initial utility:

```python
utility = -mean_tour_length
```

Higher utility is therefore better.

## Codex Prompt

```text
Implement milestone 2: a minimal deterministic TSP problem.

Requirements:

- Generate Euclidean TSP instances from a random seed.
- Support sizes 10, 20, 50 initially.
- Coordinates are numpy arrays.
- A heuristic must expose:

    select_next_node(
        current_node,
        unvisited,
        coordinates
    ) -> int

- Starting city is always city 0.
- Repeatedly call select_next_node until all cities are visited.
- Return to city 0.
- Compute tour length.

Implement two built-in heuristics:
1. nearest_neighbor
2. random_choice with seeded RNG

Define utility such that higher is better.
For now use:

    utility = -mean_tour_length

Add tests verifying:
- generated tours visit every node exactly once
- nearest_neighbor produces valid tours
- same seed produces same score
```

Expected demo:

```bash
uv run python -m moh.main --mode tsp-demo
```

Example output:

```text
TSP20
nearest_neighbor
mean_length: 4.32
utility: -4.32
```

---

# 6. Milestone 3 — Execute Heuristic Source Code

## Goal

Generated heuristics will arrive as source strings.

Architecture:

```text
source code
    ↓
sandbox / subprocess
    ↓
Python function
    ↓
TSP evaluator
    ↓
utility
```

Never execute generated code directly in the main experiment process.

## Codex Prompt

```text
Implement milestone 3: execute heuristic source code.

A Heuristic contains Python source code defining:

def select_next_node(current_node, unvisited, coordinates):
    ...

Create HeuristicRunner.

Responsibilities:
- validate required function exists
- execute heuristic in an isolated subprocess
- enforce a timeout
- capture exceptions
- return a structured EvaluationResult
- failed heuristics receive a very bad utility rather than crashing
  the experiment

For now network isolation does not need to be perfect,
but do NOT eval arbitrary code in the main process.

Add tests for:
- valid heuristic
- syntax error
- infinite loop / timeout
- missing select_next_node
```

Required behavior:

```text
bad generated candidate
    ↓
candidate fails
    ↓
minimum utility
    ↓
experiment continues
```

---

# 7. Milestone 4 — Inner Heuristic Search

## Goal

Implement the right-hand / inner-loop part of MoH.

Start with two manually written optimizers.

### Optimizer A — Best Mutation

```text
Heuristic Population
        ↓
evaluate
        ↓
select best
        ↓
ask LLM to improve best
        ↓
new child
        ↓
evaluate
        ↓
keep top-k
```

### Optimizer B — Crossover

```text
Heuristic Population
        ↓
evaluate
        ↓
select top 2
      ↙     ↘
 parent A   parent B
      ↘     ↙
      LLM crossover
           ↓
         child
           ↓
        evaluate
           ↓
        keep top-k
```

Use a deterministic `FakeLLM` first.

## Codex Prompt

```text
Implement milestone 4: the inner heuristic-search loop.

Create HeuristicOptimizer protocol:

optimize(
    population: Population[Heuristic],
    task: Task,
    llm: LLMClient,
    iterations: int
) -> Population[Heuristic]

Implement two optimizers:

1. BestMutationOptimizer

Each generation:
- evaluate population
- select current best heuristic
- ask LLM to improve it
- evaluate child
- insert child
- retain top-k

2. CrossoverOptimizer

Each generation:
- select top two heuristics
- ask LLM to combine them
- evaluate child
- retain top-k

Create FakeLLM.

FakeLLM must return deterministic predefined heuristic code so tests
do not make network calls.

Add an integration test showing that an optimizer can run 3
generations on TSP10.
```

At this point this flow must work:

```text
H1
H2
H3
 │
 ▼
Optimizer
 │
 ▼
LLM
 │
 ▼
H4
 │
 ▼
Evaluate
 │
 ▼
Population update
```

---

# 8. Milestone 5 — Real LLM Adapter

## Goal

Keep core logic provider-independent.

Only `src/moh/llm/openai_client.py` should contain provider-specific code.

Environment:

```bash
export OPENAI_API_KEY="..."
```

Do not commit API keys.

## Codex Prompt

```text
Implement an OpenAI LLMClient adapter.

Requirements:

- API key comes only from OPENAI_API_KEY.
- Model name is configurable.
- No credentials in source code.
- Keep FakeLLM unchanged for tests.
- Return plain source code after stripping markdown fences.
- Log token usage where available.
- Retry transient failures with bounded retries.
- Add a CLI flag or config option:

    llm.provider=fake
    llm.provider=openai

All existing tests must continue to run without an API key.
```

Example config:

```yaml
seed: 42

llm:
  provider: openai
  model: gpt-5.6-luna

inner:
  population_size: 4
  iterations: 3

tasks:
  sizes:
    - 10
    - 20
```

---

# 9. Heuristic Generation Prompt

Initial prompt template:

```python
HEURISTIC_PROMPT = """
You are designing a constructive heuristic for Euclidean TSP.

Your output MUST contain exactly one Python function:

def select_next_node(
    current_node,
    unvisited,
    coordinates
):
    ...

Rules:
- Return exactly one node contained in unvisited.
- Do not import packages.
- Do not use file IO.
- Do not use network.
- Do not change global state.

Current heuristic:

{parent_code}

Current utility:
{utility}

Propose an improved heuristic.

Return code only.
"""
```

Runtime:

```python
prompt = HEURISTIC_PROMPT.format(
    parent_code=parent.source_code,
    utility=parent.utility,
)

child_source = llm.generate(prompt)

child = Heuristic(
    id=new_id(),
    source_code=child_source,
)

result = evaluator.evaluate(child)
```

---

# 10. Milestone 6 — Multi-Task Optimizer Evaluation

## Goal

Evaluate the **optimizer**, not just a heuristic.

An optimizer should run independently on several task sizes.

Example:

```text
Optimizer O1

TSP10 → best heuristic utility = 0.61
TSP20 → best heuristic utility = 0.47
TSP50 → best heuristic utility = 0.31

aggregate utility = 0.46
```

Important:

Each task must start with a **fresh heuristic population**.

## Codex Prompt

```text
Implement milestone 6: evaluate one heuristic optimizer across
multiple tasks.

Introduce OptimizerEvaluator.

For each optimizer:

for task in tasks:
    initialize a fresh heuristic population
    run the optimizer
    take its best resulting heuristic
    store task utility

Then compute:

optimizer_utility =
    weighted mean of task utilities

Return:

OptimizerEvaluation(
    optimizer_id,
    task_results,
    weighted_utility
)

Support task sizes:
10, 20, 50.

Use deterministic seeds.

Add tests proving that each optimizer starts each task with an
independent heuristic population.
```

Simplified implementation:

```python
def evaluate_optimizer(optimizer, tasks, llm):
    task_utilities = []

    for task in tasks:
        population = seed_population()

        final_population = optimizer.optimize(
            population=population,
            task=task,
            llm=llm,
            iterations=3,
        )

        best = final_population.best()

        task_utilities.append(best.utility)

    return weighted_mean(task_utilities)
```

---

# 11. Optimizer Population

At this point:

```python
optimizer_population = [
    BestMutationOptimizer(),
    CrossoverOptimizer(),
]
```

Conceptually:

```text
Optimizer Population

O1            O2
│             │
▼             ▼
Tasks         Tasks
│             │
▼             ▼
0.51          0.59
```

This is the `Optimizer Population` box in the MoH diagram.

It is **not**:

```text
GPT
Llama
DeepSeek
Claude
```

The LLM is a tool used by each optimizer.

---

# 12. Milestone 7 — Constrained Meta-Optimizer

## Goal

Do not generate arbitrary optimizer Python immediately.

Start with an optimizer DSL/spec.

Example:

```python
@dataclass
class OptimizerSpec:
    parent_selection: str
    use_reflection: bool
    generation_operator: str
    survivor_selection: str
    population_size: int
```

Allowed values:

```text
parent_selection:
    random
    best
    tournament

generation_operator:
    mutate
    crossover

reflection:
    true
    false

survivor_selection:
    elitist
    diversity

population_size:
    2..10
```

Example optimizer candidate:

```yaml
parent_selection: tournament
use_reflection: true
generation_operator: crossover
survivor_selection: diversity
population_size: 5
```

Architecture:

```text
current optimizer population
        ↓
optimizer scores
        ↓
Meta-Optimizer
        ↓
LLM
        ↓
new OptimizerSpec
        ↓
validate
        ↓
compile_optimizer(spec)
        ↓
new executable HeuristicOptimizer
```

## Codex Prompt

```text
Implement milestone 7: a constrained meta-optimizer.

Do NOT generate arbitrary Python optimizer code yet.

Create OptimizerSpec with:

- parent_selection:
    random
    best
    tournament

- generation_operator:
    mutate
    crossover

- reflection:
    bool

- survivor_selection:
    elitist
    diversity

- population_size:
    integer from 2 to 10

MetaOptimizer should:

1. receive the current optimizer population and scores
2. serialize optimizer specs and evaluations into a prompt
3. ask the LLM for one new OptimizerSpec as JSON
4. validate it
5. compile it into an executable HeuristicOptimizer

FakeLLM must support deterministic optimizer generation.

Add a complete outer-loop integration test.
```

---

# 13. Milestone 8 — Complete Outer Loop

Core structure:

```python
def outer_loop(cfg, llm):
    optimizers = initial_optimizer_population()

    for generation in range(cfg.outer_iterations):
        evaluations = []

        for optimizer in optimizers:
            evaluation = evaluate_optimizer(
                optimizer=optimizer,
                tasks=cfg.tasks,
                llm=llm,
            )

            evaluations.append(evaluation)

        children = meta_optimizer.generate(
            optimizer_population=optimizers,
            evaluations=evaluations,
            llm=llm,
        )

        optimizers = survivor_selection(
            optimizers + children,
            population_size=cfg.outer_population_size,
        )

    return best_optimizer(optimizers)
```

This is the full diagram:

```text
Optimizer Population
       ↓
Meta-Optimizer
       ↓
new Optimizer candidates
       ↓
run each optimizer on tasks
       ↓
optimizer utilities
       ↓
Update Optimizer Population
       ↓
repeat
```

---

# 14. Smoke Configuration

Do not start with a large search.

Use:

```yaml
seed: 42

outer:
  iterations: 2
  population_size: 2

inner:
  iterations: 2
  population_size: 3

tasks:
  sizes:
    - 10
    - 20

instances_per_task: 3

llm:
  provider: fake
```

Run:

```bash
uv run python -m moh.main \
    --config configs/smoke.yaml
```

Only after this works end-to-end, switch to:

```yaml
llm:
  provider: openai
```

---

# 15. Milestone 9 — Full Optimizer Program Evolution

After the `OptimizerSpec` version is stable, move closer to MoH.

Represent an optimizer as Python source code defining exactly:

```python
def optimize_algorithm(
    population,
    task,
    llm,
    evaluator,
    config,
):
    ...
```

Meta-Optimizer can then generate new optimizer source code.

## Codex Prompt

```text
We now want milestone 9: upgrade OptimizerSpec evolution to
program evolution, closer to the MoH paper.

Represent an optimizer as source code defining exactly:

def optimize_algorithm(
    population,
    task,
    llm,
    evaluator,
    config
):
    ...

MetaOptimizer may generate a new implementation of this function.

Requirements:

- run generated optimizers in a subprocess
- timeout
- structured errors
- optimizer cannot access filesystem/network
- retain source code and evaluation history
- invalid optimizer gets minimum utility
- never crash the outer experiment

Do not remove the OptimizerSpec implementation.
Keep it as a baseline.
```

---

# 16. Add Natural-Language Idea Layer

Instead of:

```text
code
 ↓
new code
```

use:

```text
code
 ↓
reflection / idea
 ↓
new code
```

Example:

```python
idea_prompt = f"""
Current heuristic:

{heuristic.source_code}

Utility:
{heuristic.utility}

Explain one concrete algorithmic improvement.

Do not write code.
"""
```

Then:

```python
idea = llm.generate(idea_prompt)
```

Then:

```python
code_prompt = f"""
Implement this improvement:

{idea}

Current code:

{heuristic.source_code}

Return Python code only.
"""
```

Finally:

```python
new_code = llm.generate(code_prompt)
```

Store both:

```python
Heuristic(
    source_code=new_code,
    idea=idea,
)
```

---

# 17. Logging Design

Every experiment should create:

```text
outputs/
└── 2026-09-23_164500/
    ├── config.yaml
    ├── run.json
    ├── events.jsonl
    │
    ├── optimizers/
    │   ├── o001.py
    │   ├── o002.py
    │   └── ...
    │
    └── heuristics/
        ├── h001.py
        ├── h002.py
        └── ...
```

Example event:

```json
{
  "event": "heuristic_evaluated",
  "outer_generation": 1,
  "optimizer": "o2",
  "task": "tsp20",
  "heuristic": "h14",
  "utility": -4.83
}
```

Recommended events:

```text
experiment_started
outer_generation_started
optimizer_evaluation_started
inner_generation_started
llm_called
heuristic_generated
heuristic_evaluated
optimizer_evaluated
optimizer_generated
population_updated
experiment_finished
```

---

# 18. Test Hierarchy

Keep tests layered.

```text
Unit tests
    ↓
TSP

Unit tests
    ↓
Heuristic execution

Unit tests
    ↓
LLM parser

Integration
    ↓
Inner loop

Integration
    ↓
Optimizer evaluation

Integration
    ↓
Outer loop

Smoke test
    ↓
FakeLLM full experiment
```

After every milestone:

```bash
uv run ruff check .
uv run pytest -q
```

---

# 19. Recommended AGENTS.md

```markdown
# Project

This repository implements a minimal,
research-oriented reproduction of
Meta-Optimization of Heuristics (MoH).

## Architecture

There are two optimization levels.

Inner loop:

HeuristicOptimizer
-> searches Heuristic programs.

Outer loop:

MetaOptimizer
-> searches HeuristicOptimizer programs.

Do not confuse LLMClient with an optimizer.

## Engineering rules

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

## Before completing a task

Run:

uv run ruff check .
uv run pytest -q
```

---

# 20. Git Workflow

Use one commit per milestone.

```text
m1 core abstractions
m2 tsp task
m3 sandboxed heuristic execution
m4 inner-loop search
m5 real llm adapter
m6 multi-task optimizer evaluation
m7 constrained meta-optimizer
m8 outer-loop experiment
m9 optimizer program generation
m10 logging + reproducibility
```

Do **not** ask Codex:

```text
Implement the entire MoH paper.
```

Instead work incrementally so each abstraction is testable.

---

# 21. Compare Against Official MoH Later

Only after the local mini implementation runs end-to-end:

```bash
git clone https://github.com/yiding-s/MoH.git upstream-moh
```

Then ask Codex:

```text
Compare ../upstream-moh with this repository.

Do not modify either repository yet.

Create docs/architecture-comparison.md describing:

1. Heuristic representation
2. Optimizer representation
3. Meta-optimizer
4. LLM calls
5. population management
6. task evaluation
7. configuration
8. execution model

Map equivalent concepts between both implementations.
```

---

# 22. Definition of Done — V0

Do not call the project complete until all of these are true:

```text
Fake LLM:
    ✓ full outer loop runs end-to-end

Real LLM:
    ✓ can generate heuristic code

Heuristic:
    ✓ executable in subprocess
    ✓ timeout
    ✓ error handling
    ✓ utility calculation

Inner loop:
    ✓ heuristic population evolves

Optimizer:
    ✓ evaluated on more than one TSP size

Outer loop:
    ✓ optimizer population evolves

Experiment:
    ✓ fixed seed
    ✓ source code logged
    ✓ utilities logged
    ✓ candidate failure does not crash run
```

---

# 23. Recommended Implementation Order

Use this exact sequence:

```text
/init

1. Core abstractions
2. Deterministic TSP evaluator
3. Sandboxed heuristic execution
4. FakeLLM + inner heuristic optimizer
5. OpenAI LLM adapter
6. Multi-task optimizer evaluator
7. OptimizerSpec + MetaOptimizer
8. Complete outer loop
9. Generated Python optimizer programs
10. Compare with official MoH
```

The key rule is:

```text
First make this reliable:

heuristic code
    ↓
execute
    ↓
evaluate
    ↓
population update
```

Then add:

```text
optimizer
    ↓
tasks
    ↓
optimizer utility
```

Only then add:

```text
meta-optimizer
    ↓
new optimizer
```

---

# 24. Minimal Mental Model

The whole project can be reduced to:

```python
optimizer_population = initialize_optimizers()

for outer_step in range(T):

    optimizer_scores = []

    for optimizer in optimizer_population:

        task_scores = []

        for task in tasks:

            heuristic_population = initialize_heuristics()

            for inner_step in range(K):

                heuristic_population = optimizer(
                    heuristic_population,
                    task,
                    llm,
                )

            best_heuristic = best(
                heuristic_population
            )

            task_scores.append(
                evaluate(best_heuristic, task)
            )

        optimizer_scores.append(
            aggregate(task_scores)
        )

    new_optimizers = meta_optimizer(
        optimizer_population,
        optimizer_scores,
        llm,
    )

    optimizer_population = select(
        optimizer_population + new_optimizers
    )
```

Interpretation:

```text
Inner level:

Optimizer
    ↓
better Heuristic


Outer level:

Meta-Optimizer
    ↓
better Optimizer
```

The LLM is the code-generation/reasoning engine used by both levels.

It is not itself the optimizer population.
