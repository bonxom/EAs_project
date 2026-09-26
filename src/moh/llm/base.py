from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from moh.core.models import Status


class GenerationError(Exception):
    """Expected provider, response, or candidate-generation failure."""

    def __init__(
        self,
        message: str | None = None,
        last_provider_error: str | None = None,
    ):
        class_code = getattr(type(self), "code", None)
        resolved_code = (
            message
            if message is not None and message != ""
            else class_code or ""
        )
        super().__init__(resolved_code)
        self.code = resolved_code
        self.last_provider_error = last_provider_error


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
