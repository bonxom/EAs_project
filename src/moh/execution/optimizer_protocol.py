"""Strict optimizer RPC schemas and detached worker-side population views."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

from moh.core.program_population import selection_weights
from moh.execution.process import CandidateFailure


@dataclass(frozen=True)
class OptimizerWorkerResult:
    status: str
    idea: str | None = None
    source_code: str | None = None
    claimed_utility: float | None = None
    error: str | None = None


def text(value):
    return isinstance(value, str)


def number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # JSON integers can exceed the finite float range used for fitness.
        return False


def source_size(source):
    try:
        return len(source.encode('utf-8'))
    except UnicodeError as exc:
        raise CandidateFailure('protocol') from exc


def validate_finish(payload, limits):
    if not isinstance(payload, dict) or set(payload) != {
        'status', 'idea', 'source_code', 'claimed_utility', 'error'
    }:
        raise CandidateFailure('protocol')
    if payload['status'] == 'failed':
        if (any(payload[key] is not None for key in ('idea', 'source_code', 'claimed_utility'))
                or not text(payload['error']) or not 0 < len(payload['error']) <= 1024):
            raise CandidateFailure('protocol')
    elif payload['status'] == 'success':
        if (not text(payload['idea']) or not text(payload['source_code'])
                or not payload['source_code'] or not number(payload['claimed_utility'])
                or payload['claimed_utility'] < 0 or payload['error'] is not None):
            raise CandidateFailure('invalid_return')
        if source_size(payload['source_code']) > limits.source_bytes:
            raise CandidateFailure('source_limit')
    else:
        raise CandidateFailure('protocol')
    return payload


def validate_request(envelope, expected_id, request, limits):
    if (not isinstance(envelope, dict) or set(envelope) != {'id', 'op', 'payload'}
            or type(envelope['id']) is not int or envelope['id'] != expected_id
            or not text(envelope['op']) or not isinstance(envelope['payload'], dict)):
        raise CandidateFailure('protocol')
    op, payload = envelope['op'], envelope['payload']
    if op == 'finish':
        return op, payload
    if op == 'evaluate':
        if (set(payload) != {'source_code', 'idea', 'task'}
                or not text(payload['source_code']) or not payload['source_code']
                or payload['idea'] is not None and not text(payload['idea'])
                or payload['task'] != request['task']):
            raise CandidateFailure('protocol')
        if source_size(payload['source_code']) > limits.source_bytes:
            raise CandidateFailure('source_limit')
    elif op in ('llm_prompt', 'llm_batch'):
        key = 'message' if op == 'llm_prompt' else 'messages'
        if (set(payload) != {'expertise', key, 'temperature'}
                or not text(payload['expertise'])
                or payload['temperature'] is not None and (
                    not number(payload['temperature']) or not 0 <= payload['temperature'] <= 2)):
            raise CandidateFailure('protocol')
        if op == 'llm_prompt':
            if not text(payload[key]):
                raise CandidateFailure('protocol')
        elif (not isinstance(payload[key], list) or not payload[key]
              or any(not text(item) for item in payload[key])):
            raise CandidateFailure('protocol')
        elif len(payload[key]) > min(limits.batch_size, request['batch_size']):
            raise CandidateFailure('batch_limit')
    else:
        raise CandidateFailure('protocol')
    return op, payload


class PopulationView:
    def __init__(self, snapshot, seed):
        self._population = copy.deepcopy(snapshot)
        self._rng = random.Random(seed)

    def get_population(self, task):
        return copy.deepcopy(self._population.get(task, []))

    def get_solution_by_index(self, task, index):
        return self.get_population(task)[index]

    def get_best_solution(self, task):
        return self.get_solution_by_index(task, 0)

    def get_subtask_size(self, task):
        return len(self._population.get(task, []))

    def get_random_solution(self, task):
        entries = self.get_population(task)
        if not entries:
            raise ValueError('empty population')
        return self._rng.choices(entries, selection_weights(len(entries), len(entries)), k=1)[0]
