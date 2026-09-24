import math

from moh.core.models import OptimizerEvaluation, TaskResult, WorkCounts
from moh.core.seeds import derive_seed
from moh.logging import to_json
from moh.optimizers.inner import compile_optimizer


def normalized_weights(weights, count):
    if (
        not count
        or len(weights) != count
        or any(
            type(w) not in (int, float) or not math.isfinite(w) or w <= 0
            for w in weights
        )
    ):
        raise ValueError("weights must be finite, positive, and match tasks")
    scale = max(weights)
    scaled = [w / scale for w in weights]
    total = math.fsum(scaled)
    return tuple(w / total for w in scaled)


class OptimizerEvaluator:
    def __init__(
        self, tasks, weights, runner, llm_factory, iterations, root_seed, emit
    ):
        self.tasks = tuple(tasks)
        if len({t.id for t in tasks}) != len(tasks):
            raise ValueError("duplicate tasks")
        self.weights = normalized_weights(weights, len(tasks))
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be nonnegative")
        derive_seed(root_seed)
        self.runner, self.llm_factory = runner, llm_factory
        self.iterations, self.root_seed, self.emit = iterations, root_seed, emit

    def evaluate(self, candidate):
        self.emit("optimizer_evaluation_started", {"optimizer_id": candidate.id})
        results, counts = [], WorkCounts()
        for task in self.tasks:
            llm = self.llm_factory((candidate.id, task.id))
            inner = compile_optimizer(candidate.spec).optimize(
                task=task,
                llm=llm,
                runner=self.runner,
                iterations=self.iterations,
                root_seed=self.root_seed,
                namespace=candidate.id,
                emit=self.emit,
            )
            best = next(
                (x for x in inner.population if x.evaluation.status == "success"), None
            )
            results.append(TaskResult(task.id, best, inner.counts))
            counts += inner.counts
        success = all(x.best is not None for x in results)
        utility = (
            math.fsum(
                w * x.best.evaluation.utility
                for w, x in zip(self.weights, results, strict=True)
            )
            if success
            else None
        )
        result = OptimizerEvaluation(
            candidate.id,
            "success" if success else "failed",
            utility,
            tuple(results),
            counts,
            None if success else "task_without_success",
        )
        self.emit("optimizer_evaluated", to_json(result))
        return result
