from moh.llm.base import GenerationError


class RecordingLLM:
    def __init__(self, client, emit, scope):
        self.client, self.emit, self.scope = client, emit, dict(scope)
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        call_id = self.calls
        self.emit("llm_requested", {**self.scope, "call_id": call_id, "prompt": prompt})
        try:
            response = self.client.generate(prompt)
        except GenerationError as exc:
            self.emit(
                "llm_failed",
                {**self.scope, "call_id": call_id, "error": str(exc)[:1024]},
            )
            raise
        self.emit(
            "llm_called", {**self.scope, "call_id": call_id, "response": response}
        )
        return response
