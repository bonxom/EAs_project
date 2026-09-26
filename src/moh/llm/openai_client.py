"""The only provider-specific module; SDK retries are disabled."""

import math
import os
import time

import openai

from moh.llm.base import CallMetadata, GenerationError
from moh.llm.budget import ProviderAttemptBudgetExceeded


def resolve_base_url() -> str | None:
    return (
        os.environ.get("OPENAI_COMPAT_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
        or None
    )


OPENAI_COMPAT_API_KEY_ENV = "OPENAI_COMPAT_API_KEY"


def resolve_api_key() -> str:
    return os.environ.get(OPENAI_COMPAT_API_KEY_ENV, "").strip()


def validate_environment():
    if not resolve_api_key():
        raise ValueError("OPENAI_COMPAT_API_KEY must be configured")


def redact_credentials(text):
    key = resolve_api_key()
    return text.replace(key, "[REDACTED]") if key else text


def sanitize_provider_error(exc: Exception) -> str:
    if isinstance(exc, openai.APITimeoutError):
        return "provider_timeout"
    if isinstance(exc, openai.APIConnectionError):
        return "provider_connection_error"
    if isinstance(exc, openai.RateLimitError):
        return "provider_rate_limited"
    return type(exc).__name__


class OpenAILLMClient:
    def __init__(
        self,
        model,
        timeout_seconds,
        observer,
        *,
        transport=None,
        sleep=time.sleep,
        attempt_budget=None,
        usage_accountant=None,
        usage_limits=None,
        max_output_tokens=None,
        api_mode="auto",
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
        if (
            max_output_tokens is not None
            and (
                type(max_output_tokens) is bool
                or not isinstance(max_output_tokens, int)
                or max_output_tokens <= 0
            )
        ):
            raise ValueError(
                "max_output_tokens must be a positive integer if provided"
            )
        self.model, self.timeout, self.observer, self.sleep = (
            model,
            timeout_seconds,
            observer,
            sleep,
        )
        self.attempt_budget = attempt_budget
        self.usage_accountant = usage_accountant
        self.usage_limits = usage_limits
        self.max_output_tokens = max_output_tokens
        self.api_mode = api_mode
        self.owns_transport = transport is None
        self.last_provider_error = None
        if transport is not None:
            self.transport = transport
        else:
            base_url = resolve_base_url()
            kwargs = {
                "api_key": resolve_api_key(),
                "timeout": timeout_seconds,
                "max_retries": 0,
            }
            if base_url:
                kwargs["base_url"] = base_url
            self.transport = openai.OpenAI(**kwargs)

    def _fetch_chat_completion(self, prompt):
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.timeout,
        }
        if self.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_output_tokens
        response = self.transport.chat.completions.create(**kwargs)
        text = (
            response.choices[0].message.content
            if getattr(response, "choices", None)
            else None
        )
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "output_tokens", None)

        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if reasoning_tokens is None:
            reasoning_tokens = getattr(usage, "reasoning_tokens", None)
        if reasoning_tokens is None:
            reasoning_tokens = 0

        total_tokens = getattr(usage, "total_tokens", None)
        model = getattr(response, "model", self.model)
        return (
            text,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            total_tokens,
            model,
            True,
        )

    def _fetch_completion(self, prompt):
        if self.api_mode == "responses" or (
            self.api_mode == "auto" and hasattr(self.transport, "responses")
        ):
            try:
                kwargs = {
                    "model": self.model,
                    "input": prompt,
                    "timeout": self.timeout,
                    "store": False,
                }
                if self.max_output_tokens is not None:
                    kwargs["max_output_tokens"] = self.max_output_tokens
                response = self.transport.responses.create(**kwargs)
                try:
                    text = getattr(response, "output_text", None)
                except TypeError:
                    text = None
                    if hasattr(self.transport, "chat"):
                        return self._fetch_chat_completion(prompt)
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", None)
                if input_tokens is None:
                    input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if output_tokens is None:
                    output_tokens = getattr(usage, "completion_tokens", None)

                details = getattr(usage, "output_tokens_details", None)
                if details is None:
                    details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
                if reasoning_tokens is None:
                    reasoning_tokens = getattr(usage, "reasoning_tokens", None)
                if reasoning_tokens is None:
                    reasoning_tokens = 0

                total_tokens = getattr(usage, "total_tokens", None)
                model = getattr(response, "model", self.model)
                status = getattr(response, "status", None)
                return (
                    text,
                    input_tokens,
                    output_tokens,
                    reasoning_tokens,
                    total_tokens,
                    model,
                    status == "completed",
                )
            except (openai.NotFoundError, AttributeError):
                pass
            except openai.APIStatusError as exc:
                if exc.status_code != 404:
                    raise

        return self._fetch_chat_completion(prompt)

    def generate(self, prompt):
        self.last_provider_error = None
        for attempt in range(1, 4):
            if self.usage_accountant is not None:
                self.usage_accountant.check_pre_request_guard(
                    self.usage_limits, self.model
                )
            if self.attempt_budget is not None:
                try:
                    self.attempt_budget.reserve()
                except ProviderAttemptBudgetExceeded as exc:
                    exc.last_provider_error = self.last_provider_error
                    raise
            try:
                (
                    text,
                    input_tokens,
                    output_tokens,
                    reasoning_tokens,
                    total_tokens,
                    model,
                    is_completed,
                ) = self._fetch_completion(prompt)
            except openai.APIError as exc:
                transient = isinstance(
                    exc, (openai.APIConnectionError, openai.RateLimitError)
                ) or (isinstance(exc, openai.APIStatusError) and exc.status_code >= 500)
                error_code = sanitize_provider_error(exc)
                self.last_provider_error = error_code
                self.observer(
                    CallMetadata(
                        "openai", self.model, attempt, "failed", None, None, error_code
                    )
                )
                if not transient or attempt == 3:
                    raise GenerationError(
                        error_code, last_provider_error=error_code
                    ) from None
                self.sleep(0.25 * attempt)
                continue
            valid = isinstance(text, str) and bool(text.strip()) and is_completed
            valid = (
                valid
                and isinstance(model, str)
                and all(
                    x is None or (type(x) is int and x >= 0)
                    for x in (
                        input_tokens,
                        output_tokens,
                        reasoning_tokens,
                        total_tokens,
                    )
                )
            )
            if (
                valid
                and output_tokens is not None
                and reasoning_tokens is not None
                and reasoning_tokens > output_tokens
            ):
                valid = False
            if (
                valid
                and input_tokens is not None
                and output_tokens is not None
                and total_tokens is not None
                and total_tokens != input_tokens + output_tokens
            ):
                valid = False

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

            rec_input = input_tokens if input_tokens is not None else 0
            rec_output = output_tokens if output_tokens is not None else 0
            rec_reasoning = reasoning_tokens if reasoning_tokens is not None else 0

            if self.usage_accountant is not None:
                try:
                    self.usage_accountant.record(
                        model=model,
                        input_tokens=rec_input,
                        output_tokens=rec_output,
                        reasoning_tokens=rec_reasoning,
                    )
                except ValueError:
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
                    raise GenerationError("malformed_response") from None

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
