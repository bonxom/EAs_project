from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from moh.core.models import Status


class GenerationError(Exception):
    """Expected provider, response, or candidate-generation failure."""


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


type LLMFactory = Callable[[tuple[str, ...]], LLMClient]


@dataclass(frozen=True)
class CallMetadata:
    provider: str
    model: str
    attempt: int
    status: Status
    input_tokens: int | None
    output_tokens: int | None
    error: str | None


type CallObserver = Callable[[CallMetadata], None]


MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 1_048_576


@dataclass(frozen=True)
class LLMRequest:
    expertise: str
    message: str
    temperature: float | None = None
    timeout_seconds: float | None = None

    def __post_init__(self):
        import math

        for text in (self.expertise, self.message):
            if not isinstance(text, str) or not text.strip():
                raise ValueError('request text must be nonempty')
        if len((self.expertise + self.message).encode('utf-8')) > MAX_REQUEST_BYTES:
            raise ValueError('request text exceeds byte limit')
        for name, maximum in (('temperature', 2.0), ('timeout_seconds', None)):
            value = getattr(self, name)
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if value is not None and (
                not finite
                or (value < 0 if name == 'temperature' else value <= 0)
                or (maximum is not None and value > maximum)
            ):
                raise ValueError(f'invalid {name}')


def validate_response(response):
    if not isinstance(response, str) or not response.strip():
        raise GenerationError('empty_response')
    if len(response.encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise GenerationError('response_limit')
    return response
