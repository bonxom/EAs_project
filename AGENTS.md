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

## Required task workflow

Follow this sequence for every task:

**Brainstorm -> Spec -> Plan -> Test -> Implement -> Test again.**

Do not skip or reorder stages. Keep the depth of each stage proportional to
the task; small tasks may use a concise written spec and plan.

1. **Brainstorm:** Understand the request, inspect relevant context, identify
   constraints, and consider suitable approaches. Clarify only unresolved
   questions that materially affect the outcome.
2. **Spec:** Write down the intended behavior, scope, constraints, and verifiable
   acceptance criteria before implementation. Do not invent requirements.
3. **Plan:** Define ordered implementation steps, affected files, and the tests
   or checks that will demonstrate compliance with the spec.
4. **Test before implementation:** For behavior changes or bug fixes, write or
   update meaningful tests first and run them. Confirm that new tests fail for
   the expected missing behavior or bug, rather than an unrelated setup error.
   Use FakeLLM wherever an LLM is involved; never call external LLM APIs in
   unit tests. For documentation-only tasks or changes with no behavior to
   test, define and run appropriate validation checks instead of adding
   artificial unit tests. Record the baseline and any existing failures.
5. **Implement:** Only after the preceding stages, make the changes needed to
   satisfy the spec and tests. Keep changes within the agreed scope. If the
   requirements change, revise the spec, plan, and tests before continuing.
6. **Test again:** Rerun relevant tests or validation checks, then run the
   required repository checks below. Fix regressions introduced by the task.
   Report what changed, what was verified, and any unresolved failures or
   checks that could not run; never claim unrun checks passed.

## Before completing a task

Run both commands from the repository root, including for documentation-only
changes:

```sh
uv run ruff check .
uv run pytest -q
```
