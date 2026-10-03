"""Parent adapter for generated optimizers; callback fitness stays parent-owned."""
from dataclasses import dataclass

from moh.core.programs import OptimizerProgram
from moh.execution.optimizer_protocol import OptimizerWorkerResult, validate_finish
from moh.execution.process import CandidateFailure, Deadline


@dataclass(frozen=True)
class OptimizerRequest:
    program: OptimizerProgram
    snapshot: dict
    task: str
    function_format: str
    seed: int
    batch_size: int


class OptimizerRunner:
    def __init__(self, supervisor, limits):
        self.supervisor = supervisor
        self.limits = limits

    def run(self, request, dispatch, *, deadline, scope):
        invocation = Deadline.after(self.limits.timeout_seconds, parent=deadline)
        try:
            if len(request.program.source_code.encode('utf-8')) > self.limits.source_bytes:
                raise CandidateFailure('source_limit')
            if type(request.batch_size) is not int or not 0 < request.batch_size <= self.limits.batch_size:
                raise CandidateFailure('batch_limit')
            if type(request.seed) is not int or not 0 <= request.seed < 2**32:
                raise ValueError('seed must be an integer in [0, 2**32)')
            payload = {'source': request.program.source_code, 'snapshot': request.snapshot,
                       'task': request.task, 'function_format': request.function_format,
                       'seed': request.seed, 'batch_size': request.batch_size,
                       'result_bytes': self.limits.result_bytes,
                       'request_bytes': self.limits.request_bytes,
                       'source_bytes': self.limits.source_bytes}
            result = self.supervisor.exchange(
                'moh.execution.optimizer_worker', payload, dispatch,
                limits=self.limits, deadline=invocation, scope=scope,
            )
            return OptimizerWorkerResult(**validate_finish(result, self.limits))
        except CandidateFailure as exc:
            return OptimizerWorkerResult('failed', error=exc.code)
