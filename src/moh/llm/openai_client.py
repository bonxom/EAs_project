"""The only provider-specific module; SDK retries are disabled."""

import math
import os
import time

import openai

from moh.llm.base import CallMetadata, GenerationError


def validate_environment():
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY must be configured")


def redact_credentials(text):
    key = os.environ.get("OPENAI_API_KEY", "")
    return text.replace(key, "[REDACTED]") if key else text


class OpenAILLMClient:
    def __init__(
        self, model, timeout_seconds, observer, *, transport=None, sleep=time.sleep
    ):
        validate_environment()
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be explicitly configured")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("provider timeout must be positive and finite")
        self.model, self.timeout, self.observer, self.sleep = (
            model,
            timeout_seconds,
            observer,
            sleep,
        )
        self.owns_transport = transport is None
        self.transport = (
            transport
            if transport is not None
            else openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                timeout=timeout_seconds,
                max_retries=0,
            )
        )

    def generate(self, prompt):
        for attempt in range(1, 4):
            try:
                response = self.transport.responses.create(
                    model=self.model, input=prompt, timeout=self.timeout, store=False
                )
            except openai.APIError as exc:
                transient = isinstance(
                    exc, (openai.APIConnectionError, openai.RateLimitError)
                ) or (isinstance(exc, openai.APIStatusError) and exc.status_code >= 500)
                error = type(
                    exc
                ).__name__  # Never retain headers, bodies, or credentials.
                self.observer(
                    CallMetadata(
                        "openai", self.model, attempt, "failed", None, None, error
                    )
                )
                if not transient or attempt == 3:
                    raise GenerationError(error) from None
                self.sleep(0.25 * attempt)
                continue
            text = getattr(response, "output_text", None)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            model = getattr(response, "model", self.model)
            valid = (
                isinstance(text, str)
                and bool(text.strip())
                and getattr(response, "status", None) == "completed"
            )
            valid = (
                valid
                and isinstance(model, str)
                and all(
                    x is None or (type(x) is int and x >= 0)
                    for x in (input_tokens, output_tokens)
                )
            )
            if not valid:
                self.observer(
                    CallMetadata(
                        "openai",
                        self.model,
                        attempt,
                        "failed",
                        None,
                        None,
                        "malformed_response",
                    )
                )
                raise GenerationError("malformed_response")
            self.observer(
                CallMetadata(
                    "openai",
                    redact_credentials(model),
                    attempt,
                    "success",
                    input_tokens,
                    output_tokens,
                    None,
                )
            )
            return redact_credentials(text)
        raise AssertionError("unreachable")

    def close(self):
        if self.owns_transport:
            self.transport.close()
