import math

from moh.core.models import EvaluationResult
from moh.execution.sandbox import run_worker


def make_request(heuristic, task, index, seed):
    return {
        "id": heuristic.id,
        "source": heuristic.source_code,
        "coordinates": task.instances[index].tolist(),
        "seed": seed,
    }


def failed_evaluation(identity, context, lengths, attempts, error):
    return EvaluationResult(
        identity, context, "failed", None, tuple(lengths), error, attempts
    )


class HeuristicRunner:
    def __init__(self, limits):
        self.limits = limits

    def evaluate(self, heuristic, task, context):
        if (
            context.task_id != task.id
            or context.instance_seeds != task.instance_seeds
            or len(task.instances) != len(context.worker_seeds)
        ):
            raise ValueError("task/context mismatch")
        lengths = []
        for index, seed in enumerate(context.worker_seeds):
            result = run_worker(make_request(heuristic, task, index, seed), self.limits)
            if result.status == "failed":
                return failed_evaluation(
                    heuristic.id, context, lengths, index + 1, result.error
                )
            try:
                lengths.append(task.score_tour(index, result.tour))
            except (ValueError, OverflowError):
                return failed_evaluation(
                    heuristic.id, context, lengths, index + 1, "protocol"
                )
        utility = -math.fsum(x / len(lengths) for x in lengths)
        return EvaluationResult(
            heuristic.id,
            context,
            "success",
            utility,
            tuple(lengths),
            None,
            len(lengths),
        )
