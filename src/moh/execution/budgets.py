"""Run-wide accounting; nested invocations share this same budget."""
from threading import RLock

from moh.core.models import WorkCounts
from moh.execution.process import CandidateFailure


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError('count must be a nonnegative integer')


class WorkBudget:
    def __init__(self, max_llm_calls: int, max_heuristic_evaluations: int):
        _count(max_llm_calls)
        _count(max_heuristic_evaluations)
        self.max_llm_calls = max_llm_calls
        self.max_heuristic_evaluations = max_heuristic_evaluations
        self._counts = WorkCounts()
        self._lock = RLock()

    @property
    def counts(self):
        with self._lock:
            return self._counts

    def require_llm(self, count):
        _count(count)
        with self._lock:
            if self._counts.llm_calls + count > self.max_llm_calls:
                raise CandidateFailure('llm_budget')

    def charge_llm(self):
        with self._lock:
            self.require_llm(1)
            self._counts += WorkCounts(llm_calls=1)

    def charge_evaluation(self):
        with self._lock:
            if self._counts.heuristic_evaluations >= self.max_heuristic_evaluations:
                raise CandidateFailure('evaluation_budget')
            self._counts += WorkCounts(heuristic_evaluations=1)

    def charge_instances(self, count):
        _count(count)
        with self._lock:
            self._counts += WorkCounts(instance_attempts=count)

    def delta(self, before: WorkCounts) -> WorkCounts:
        now = self.counts
        return WorkCounts(now.heuristic_evaluations - before.heuristic_evaluations,
                          now.instance_attempts - before.instance_attempts,
                          now.llm_calls - before.llm_calls)
