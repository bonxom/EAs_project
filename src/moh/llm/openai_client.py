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


def _normalize_token_usage_semantics(
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    total_tokens: int | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    if (
        input_tokens is not None
        and output_tokens is not None
        and reasoning_tokens is not None
        and total_tokens is not None
    ):
        if (
            total_tokens == input_tokens + output_tokens
            and reasoning_tokens <= output_tokens
        ):
            return input_tokens, output_tokens, reasoning_tokens, total_tokens

        if (
            reasoning_tokens > 0
            and total_tokens == input_tokens + output_tokens + reasoning_tokens
        ):
            return (
                input_tokens,
                output_tokens + reasoning_tokens,
                reasoning_tokens,
                total_tokens,
            )

    return input_tokens, output_tokens, reasoning_tokens, total_tokens


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
        last_provider_error = None
        for attempt in range(1, 4):
            if self.usage_accountant is not None:
                self.usage_accountant.check_pre_request_guard(
                    self.usage_limits, self.model
                )
            if self.attempt_budget is not None:
                try:
                    self.attempt_budget.reserve()
                except ProviderAttemptBudgetExceeded as exc:
                    exc.last_provider_error = last_provider_error
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
                wire_input_tokens = input_tokens
                wire_output_tokens = output_tokens
                wire_reasoning_tokens = reasoning_tokens
                wire_total_tokens = total_tokens
            except openai.APIError as exc:
                transient = isinstance(
                    exc, (openai.APIConnectionError, openai.RateLimitError)
                ) or (isinstance(exc, openai.APIStatusError) and exc.status_code >= 500)
                error_code = sanitize_provider_error(exc)
                last_provider_error = error_code
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
            def raise_malformed(
                reason: str,
                current_attempt: int = attempt,
                obs_in: int | None = None,
                obs_out: int | None = None,
                obs_reas: int | None = None,
                obs_tot: int | None = None,
            ):
                self.observer(
                    CallMetadata(
                        "openai",
                        self.model,
                        current_attempt,
                        "failed",
                        None,
                        None,
                        "malformed_response",
                    )
                )
                raise GenerationError(
                    "malformed_response",
                    malformed_response_reason=reason,
                    observed_input_tokens=obs_in,
                    observed_output_tokens=obs_out,
                    observed_reasoning_tokens=obs_reas,
                    observed_total_tokens=obs_tot,
                )

            if text is None:
                raise_malformed("missing_text")
            elif not isinstance(text, str):
                raise_malformed("invalid_text_type")
            elif not text.strip():
                raise_malformed("empty_text")

            if not isinstance(model, str):
                raise_malformed("invalid_model")

            if not is_completed:
                raise_malformed("incomplete_response")

            if type(input_tokens) is bool or (
                input_tokens is not None and not isinstance(input_tokens, int)
            ):
                raise_malformed("invalid_input_tokens_type")
            elif input_tokens is not None and input_tokens < 0:
                raise_malformed("negative_input_tokens")

            if type(output_tokens) is bool or (
                output_tokens is not None and not isinstance(output_tokens, int)
            ):
                raise_malformed("invalid_output_tokens_type")
            elif output_tokens is not None and output_tokens < 0:
                raise_malformed("negative_output_tokens")

            if type(reasoning_tokens) is bool or (
                reasoning_tokens is not None and not isinstance(reasoning_tokens, int)
            ):
                raise_malformed("invalid_reasoning_tokens_type")
            elif reasoning_tokens is not None and reasoning_tokens < 0:
                raise_malformed("negative_reasoning_tokens")

            if type(total_tokens) is bool or (
                total_tokens is not None and not isinstance(total_tokens, int)
            ):
                raise_malformed("invalid_total_tokens_type")
            elif total_tokens is not None and total_tokens < 0:
                raise_malformed("negative_total_tokens")

            (
                input_tokens,
                output_tokens,
                reasoning_tokens,
                total_tokens,
            ) = _normalize_token_usage_semantics(
                input_tokens, output_tokens, reasoning_tokens, total_tokens
            )

            if (
                output_tokens is not None
                and reasoning_tokens is not None
                and reasoning_tokens > output_tokens
            ):
                raise_malformed(
                    "reasoning_exceeds_output",
                    obs_in=wire_input_tokens,
                    obs_out=wire_output_tokens,
                    obs_reas=wire_reasoning_tokens,
                    obs_tot=wire_total_tokens,
                )

            if (
                input_tokens is not None
                and output_tokens is not None
                and total_tokens is not None
                and total_tokens != input_tokens + output_tokens
            ):
                raise_malformed(
                    "inconsistent_total_tokens",
                    obs_in=wire_input_tokens,
                    obs_out=wire_output_tokens,
                    obs_reas=wire_reasoning_tokens,
                    obs_tot=wire_total_tokens,
                )

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
                    raise GenerationError(
                        "malformed_response",
                        malformed_response_reason="usage_accountant_rejected",
                    ) from None

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
