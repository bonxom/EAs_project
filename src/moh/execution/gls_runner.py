"""Evaluate heuristic programs with isolated execution and parent-owned scoring."""

import math
from dataclasses import asdict

from moh.core.models import Heuristic, WorkCounts
from moh.core.programs import GapEvaluation, mean_fitness
from moh.core.seeds import derive_seed
from moh.execution.optimizer_protocol import source_size
from moh.execution.process import CandidateFailure, Deadline, encode_frame
from moh.problems.tsp_gls.evaluation import GapOverflow, gap_percent
from moh.problems.tsp_gls.task import GLSTask
from moh.problems.tsp_gls.tour import tour_cost, validate_tour


def _no_callback(operation, payload, deadline):
    raise CandidateFailure('protocol')


def _decode_tour(result, size):
    if not isinstance(result, dict) or set(result) != {'status', 'tour', 'error'}:
        raise CandidateFailure('protocol')
    if result['status'] == 'failed':
        if (result['tour'] is not None or not isinstance(result['error'], str)
                or not 0 < len(result['error']) <= 1024):
            raise CandidateFailure('protocol')
        raise CandidateFailure(result['error'])
    if result['status'] != 'success' or result['error'] is not None:
        raise CandidateFailure('protocol')
    if not isinstance(result['tour'], list):
        raise CandidateFailure('invalid_tour')
    try:
        return validate_tour(result['tour'], size)
    except ValueError as exc:
        raise CandidateFailure('invalid_tour') from exc


class GLSRunner:
    def __init__(self, supervisor, limits, options):
        self.supervisor = supervisor
        self.limits = limits
        self.options = options

    def evaluate(self, heuristic: Heuristic, task: GLSTask, *, split: str,
                 root_seed: int, deadline: Deadline, scope) -> GapEvaluation:
        if split not in ('validation', 'test'):
            raise ValueError('split must be validation or test')
        # Validate trusted seed before converting any candidate failures to records.
        derive_seed(root_seed)
        costs, gaps, tours = [], [], []
        attempts = 0
        error = None
        try:
            if source_size(heuristic.source_code) > self.limits.source_bytes:
                raise CandidateFailure('source_limit')
            for instance in getattr(task, split):
                deadline.check()
                scope.check()
                invocation = Deadline.after(self.limits.timeout_seconds, parent=deadline)
                request = {
                    'source': heuristic.source_code,
                    'distances': instance.distances.tolist(),
                    'options': asdict(self.options),
                    'seed': derive_seed(root_seed, 'gls', task.id, split, instance.id, 'worker'),
                    'task': task.id, 'batch_size': self.limits.batch_size,
                    'source_bytes': self.limits.source_bytes,
                    'request_bytes': self.limits.request_bytes,
                    'result_bytes': self.limits.result_bytes,
                }
                encode_frame(request, self.limits.request_bytes, 'request_limit')
                with self.supervisor.scope(invocation, parent_scope=scope) as child_scope:
                    attempts += 1
                    result = self.supervisor.exchange(
                        'moh.execution.gls_worker', request, _no_callback,
                        limits=self.limits, deadline=invocation, scope=child_scope,
                    )
                tour = _decode_tour(result, task.size)
                cost = tour_cost(instance.distances, tour)
                if not math.isfinite(cost):
                    raise CandidateFailure('invalid_cost')
                # Reference inconsistencies are infrastructure errors, not candidate failures.
                try:
                    gap = gap_percent(cost, instance.optimal_cost)
                except GapOverflow as exc:
                    raise CandidateFailure('invalid_gap') from exc
                costs.append(cost)
                gaps.append(gap)
                tours.append(tour)
        except CandidateFailure as exc:
            error = exc.code
        return GapEvaluation(
            heuristic.id, task.id, split, 'failed' if error else 'success',
            None if error else mean_fitness(gaps),
            tuple(costs), tuple(gaps), tuple(tours), error, WorkCounts(1, attempts, 0),
        )
