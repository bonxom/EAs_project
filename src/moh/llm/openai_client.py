"""The only provider-specific module; SDK retries are disabled."""

import math
import os
import time

import openai

from moh.llm.base import CallMetadata, GenerationError, validate_response


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
        return self._generate(prompt)

    def generate_request(self, request):
        duration = min(self.timeout, request.timeout_seconds or self.timeout)
        return self._generate(request.message, request=request,
                              expires_at=time.monotonic() + duration)

    def _generate(self, prompt, *, request=None, expires_at=None):
        for attempt in range(1, 4):
            try:
                timeout = self.timeout
                if expires_at is not None:
                    timeout = min(timeout, expires_at - time.monotonic())
                    if timeout <= 0:
                        raise GenerationError("timeout")
                params = {"model": self.model, "input": prompt,
                          "timeout": timeout, "store": False}
                if request is not None:
                    params["instructions"] = request.expertise
                    if request.temperature is not None:
                        params["temperature"] = request.temperature
                response = self.transport.responses.create(**params)
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
                delay = 0.25 * attempt
                if expires_at is not None and delay >= expires_at - time.monotonic():
                    raise GenerationError("timeout") from None
                self.sleep(delay)
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
            if request is not None:
                if time.monotonic() >= expires_at:
                    raise GenerationError("timeout")
                validate_response(text)
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
