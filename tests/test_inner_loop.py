import itertools

import pytest

from moh.core.models import EvaluationResult
from moh.core.specs import OptimizerSpec
from moh.execution.heuristic_runner import HeuristicRunner
from moh.execution.protocol import ExecutionLimits
from moh.llm.base import GenerationError
from moh.llm.fake import FakeLLM
from moh.llm.recording import RecordingLLM
from moh.optimizers.inner import compile_optimizer
from moh.optimizers.seed_optimizers import initial_optimizer_candidates
from moh.problems.tsp import TSPTask


class ControlledRunner:
    def __init__(self, fail_first=0):
        self.calls = 0
        self.fail_first = fail_first

    def evaluate(self, heuristic, task, context):
        self.calls += 1
        if self.calls <= self.fail_first:
            return EvaluationResult(
                heuristic.id, context, "failed", None, (), "test_failure", 1
            )
        return EvaluationResult(heuristic.id, context, "success", -1.0, (1.0,), None, 1)


def run(spec, iterations=3, runner=None, fake=None):
    events = []
    emit = lambda e, p: events.append((e, p))
    client = RecordingLLM(fake or FakeLLM(42), emit, {})
    result = compile_optimizer(spec).optimize(
        task=TSPTask.create(10, 1, 42),
        llm=client,
        runner=runner or ControlledRunner(),
        iterations=iterations,
        root_seed=42,
        namespace="o1",
        emit=emit,
    )
    return result, events


@pytest.mark.parametrize("index", [0, 1])
def test_seed_optimizers(index):
    spec = initial_optimizer_candidates(3)[index].spec
    first, _ = run(spec, runner=HeuristicRunner(ExecutionLimits()))
    assert len(first.population) == 3
    assert first.counts.heuristic_evaluations == 6
    assert first.counts.llm_calls == 3
    assert first == run(spec, runner=HeuristicRunner(ExecutionLimits()))[0]


@pytest.mark.parametrize(
    "p,g,r,s",
    list(
        itertools.product(
            ["best", "random", "tournament"],
            ["mutate", "crossover"],
            [True, False],
            ["elitist", "diversity"],
        )
    ),
)
def test_policy_matrix(p, g, r, s):
    result, events = run(
        OptimizerSpec(
            parent_selection=p,
            generation_operator=g,
            use_reflection=r,
            survivor_selection=s,
            population_size=3,
        )
    )
    assert result.counts.llm_calls == (6 if r else 3)
    generated = [
        payload
        for kind, payload in events
        if kind == "heuristic_generated" and payload["generation"] >= 0
    ]
    assert len(generated) == 3
    assert all(len(x["parents"]) == (2 if g == "crossover" else 1) for x in generated)
    if r:
        requests = [
            x["prompt"].splitlines()[0] for k, x in events if k == "llm_requested"
        ]
        assert requests == ["KIND: reflection", f"KIND: {g}"] * 3
        assert all(x["idea"] for x in generated)


def test_failures_recovery_and_zero(base_spec):
    spec = OptimizerSpec(**base_spec)
    result, _ = run(spec, runner=ControlledRunner(fail_first=3))
    assert result.population[0].evaluation.status == "success"
    result, _ = run(spec, iterations=0)
    assert result.counts.heuristic_evaluations == 3
    assert result.counts.llm_calls == 0
    result, events = run(
        spec, fake=FakeLLM(42, {"mutate": (GenerationError("bad"),) * 3})
    )
    assert result.counts.heuristic_evaluations == 3
    assert result.counts.llm_calls == 3
    assert len([x for k, x in events if k == "generation_failed"]) == 3
    reflected = spec.model_copy(update={"use_reflection": True})
    result, _ = run(
        reflected,
        iterations=1,
        fake=FakeLLM(42, {"reflection": (GenerationError("bad"),)}),
    )
    assert result.counts.llm_calls == 1


def test_failed_code_preserves_parents(base_spec):
    result, events = run(
        OptimizerSpec(**base_spec),
        iterations=1,
        runner=HeuristicRunner(ExecutionLimits()),
        fake=FakeLLM(42, {"mutate": ("bad code !",)}),
    )
    assert all(x.evaluation.status == "success" for x in result.population)
    assert any(
        k == "heuristic_evaluated" and x["status"] == "failed" for k, x in events
    )


def test_invalid_unicode_generation_consumes_iteration(base_spec):
    result, events = run(
        OptimizerSpec(**base_spec),
        iterations=1,
        fake=FakeLLM(42, {"mutate": ("\ud800",)}),
    )
    assert result.counts.heuristic_evaluations == 3
    assert result.counts.llm_calls == 1
    assert any(k == "generation_failed" for k, _ in events)
