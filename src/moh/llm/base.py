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
