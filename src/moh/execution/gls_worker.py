"""Run generated heuristic module and an entire GLS instance in a child only."""

import os
import random
import sys

import numpy as np

from moh.execution.optimizer_worker import RPC
from moh.execution.process import CandidateFailure
from moh.execution.protocol import strict_json
from moh.problems.tsp_gls.solver import GLSOptions, solve_gls


def main():
    fd = int(sys.argv[1])
    frame = sys.stdin.buffer.readline(int(sys.argv[2]) + 1)
    request = strict_json(frame)
    rpc = RPC(fd, request)
    try:
        random.seed(request['seed'])
        np.random.seed(request['seed'])
        namespace = {'__name__': '__generated_heuristic__'}
        exec(compile(request['source'], '<heuristic>', 'exec'), namespace)  # noqa: S102 -- child only
        function = namespace.get('update_edge_distance')
        if not callable(function):
            raise CandidateFailure('missing_function')
        distances = np.asarray(request['distances'], dtype=np.float64)
        tour = solve_gls(distances, function, GLSOptions(**request['options']))
        result = {'status': 'success', 'tour': list(tour), 'error': None}
    except BaseException as exc:  # noqa: BLE001 -- contain candidate SystemExit too
        error = 'syntax' if isinstance(exc, SyntaxError) else (
            exc.code if isinstance(exc, CandidateFailure) else 'exception')
        result = {'status': 'failed', 'tour': None, 'error': error}
    try:
        rpc.emit('finish', result)
    except BaseException:  # noqa: BLE001, S110 -- parent may have closed the pipe
        pass
    os.close(fd)


if __name__ == '__main__':
    main()
