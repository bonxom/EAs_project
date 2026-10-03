"""Validated parent-side callbacks shared by both optimizer levels."""
from moh.execution.optimizer_protocol import source_size
from moh.execution.process import CandidateFailure
from moh.llm.base import GenerationError, LLMRequest


class InvocationDispatcher:
    def __init__(self, task, llm, budget, evaluate, *, max_batch_size):
        self.task, self.llm, self.budget, self.evaluate = task, llm, budget, evaluate
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError('max_batch_size must be positive')
        self.max_batch_size = max_batch_size

    def dispatch(self, operation, payload, deadline):
        deadline.check()
        if not isinstance(payload, dict):
            raise CandidateFailure('protocol')
        if operation == 'evaluate':
            if (set(payload) != {'source_code', 'idea', 'task'}
                    or payload['task'] != self.task
                    or not isinstance(payload['source_code'], str) or not payload['source_code']
                    or payload['idea'] is not None and not isinstance(payload['idea'], str)):
                raise CandidateFailure('protocol')
            source_size(payload['source_code'])
            if payload['idea'] is not None:
                source_size(payload['idea'])
            return self.evaluate(payload['source_code'], payload['idea'], deadline)
        if operation not in ('llm_prompt', 'llm_batch'):
            raise CandidateFailure('protocol')
        key = 'message' if operation == 'llm_prompt' else 'messages'
        if set(payload) != {'expertise', key, 'temperature'}:
            raise CandidateFailure('protocol')
        messages = [payload[key]] if operation == 'llm_prompt' else payload[key]
        if not isinstance(messages, list) or not 0 < len(messages) <= min(
                self.max_batch_size, self.llm.batch_size):
            raise CandidateFailure('batch_limit')
        # Only validation failures are translated here: provider infrastructure
        # errors, including ValueError, must retain their original meaning.
        try:
            for message in messages:
                LLMRequest(payload['expertise'], message, payload['temperature'])
        except (ValueError, UnicodeError) as exc:
            raise CandidateFailure('invalid_llm_request') from exc
        self.budget.require_llm(len(messages))
        responses = []
        for message in messages:
            deadline.check()
            self.budget.charge_llm()
            try:
                responses.append(self.llm.prompt(payload['expertise'], message,
                                                payload['temperature'], deadline=deadline))
            except (GenerationError, UnicodeError) as exc:
                raise CandidateFailure('generation') from exc
        return responses[0] if operation == 'llm_prompt' else responses
