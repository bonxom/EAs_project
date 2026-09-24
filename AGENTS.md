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