"""Only child processes compile and execute generated optimizer modules."""
import os
import random
import sys

import numpy as np

from moh.execution.optimizer_protocol import PopulationView, validate_finish
from moh.execution.process import CandidateFailure, ProgramLimits, encode_frame
from moh.execution.protocol import strict_json


class RPC:
    def __init__(self, fd, request):
        self.fd = fd
        self.request = request
        self.next_id = 1

    def emit(self, op, payload):
        identity = self.next_id
        self.next_id += 1
        frame = encode_frame({'id': identity, 'op': op, 'payload': payload},
                             self.request['result_bytes'], 'result_limit')
        offset = 0
        while offset < len(frame):
            offset += os.write(self.fd, frame[offset:])
        return identity

    def call(self, op, payload):
        identity = self.emit(op, payload)
        frame = sys.stdin.buffer.readline(self.request['request_bytes'] + 1)
        if not frame.endswith(b'\n') or len(frame) > self.request['request_bytes']:
            raise CandidateFailure('protocol')
        response = strict_json(frame)
        if (not isinstance(response, dict) or set(response) != {'id', 'ok', 'value', 'error'}
                or type(response['id']) is not int or response['id'] != identity
                or type(response['ok']) is not bool):
            raise CandidateFailure('protocol')
        if not response['ok']:
            raise CandidateFailure('callback')
        return response['value']


class LanguageModelProxy:
    def __init__(self, rpc):
        self.rpc = rpc
        self.batch_size = rpc.request['batch_size']

    def prompt(self, expertise, message, temperature=1.0):
        return self.rpc.call('llm_prompt', {'expertise': expertise, 'message': message,
                                          'temperature': temperature})

    def prompt_batch(self, expertise, messages, temperature=1.0):
        return self.rpc.call('llm_batch', {'expertise': expertise, 'messages': messages,
                                         'temperature': temperature})


def main():
    fd = int(sys.argv[1])
    # Initial parent input is bounded before decoding, even though parent is trusted.
    frame = sys.stdin.buffer.readline(int(sys.argv[2]) + 1)
    request = strict_json(frame)
    rpc = RPC(fd, request)
    result = {'status': 'failed', 'idea': None, 'source_code': None,
              'claimed_utility': None, 'error': 'exception'}
    try:
        random.seed(request['seed'])
        np.random.seed(request['seed'])
        namespace = {'__name__': '__generated_optimizer__'}
        exec(compile(request['source'], '<optimizer>', 'exec'), namespace)  # noqa: S102 -- child only
        function = namespace.get('improve_algorithm')
        if not callable(function):
            raise CandidateFailure('missing_function')

        def utility(source_code, idea=None, problem_type=None):
            return rpc.call('evaluate', {'source_code': source_code, 'idea': idea,
                                        'task': request['task'] if problem_type is None else problem_type})

        returned = function(PopulationView(request['snapshot'], request['seed']), utility,
                            LanguageModelProxy(rpc), request['function_format'], request['task'])
        if not isinstance(returned, tuple) or len(returned) != 3:
            raise CandidateFailure('invalid_return')
        idea, source, score = returned
        result = {'status': 'success', 'idea': idea, 'source_code': source,
                  'claimed_utility': score, 'error': None}
        validate_finish(result, ProgramLimits(source_bytes=request['source_bytes']))
    except BaseException as exc:  # noqa: BLE001 -- candidate SystemExit is contained
        error = 'syntax' if isinstance(exc, SyntaxError) else (
            exc.code if isinstance(exc, CandidateFailure) else 'exception')
        result = {'status': 'failed', 'idea': None, 'source_code': None,
                  'claimed_utility': None, 'error': error}
    try:
        rpc.emit('finish', result)
    except BaseException:  # noqa: BLE001, S110 -- parent may have closed pipe
        pass
    os.close(fd)


if __name__ == '__main__':
    main()
