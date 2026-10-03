"""Sequential deadline-aware parent-side program LLM facade."""
from moh.llm.base import LLMRequest, validate_response


class ProgramLLM:
    def __init__(self, client, *, batch_size):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        self.client, self.batch_size = client, batch_size

    def prompt(self, expertise, message, temperature=None, *, deadline):
        deadline.check()
        request = LLMRequest(expertise, message, temperature, deadline.remaining())
        response = self.client.generate_request(request)
        deadline.check()
        return validate_response(response)

    def prompt_batch(self, expertise, messages, temperature=None, *, deadline):
        if not isinstance(messages, list) or not 0 < len(messages) <= self.batch_size:
            raise ValueError('invalid batch size')
        # Validate all entries before making a partial batch call.
        for message in messages:
            LLMRequest(expertise, message, temperature)
        return [self.prompt(expertise, message, temperature, deadline=deadline)
                for message in messages]
