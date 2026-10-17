import concurrent.futures
import json
import logging
import os
import time
from datetime import datetime
from random import random
from threading import Lock
from uuid import uuid4

logger = logging.getLogger(__name__)
#   lm = hydra.utils.instantiate(cfg.llm_client)
#   lm.prompt(expertise, message, temperature=0.7)
#   lm.prompt_batch(expertise, [msg1, msg2, msg3])


class ProviderBudget:
    """Shared provider request/attempt budget across LLM roles (heuristic + meta)."""

    def __init__(self, max_requests: int | None = None, max_attempts: int | None = None) -> None:
        if max_requests is not None and max_requests <= 0:
            raise ValueError("max_requests must be positive or None")
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive or None")
        self.max_requests = max_requests
        self.max_attempts = max_attempts
        self.requests = 0
        self.attempts = 0
        self.exhausted_reason: str | None = None
        self._lock = Lock()

    def try_request(self) -> bool:
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                self.exhausted_reason = f"max_requests={self.max_requests}"
                return False
            self.requests += 1
            return True

    def try_attempt(self) -> bool:
        with self._lock:
            if self.max_attempts is not None and self.attempts >= self.max_attempts:
                self.exhausted_reason = f"max_attempts={self.max_attempts}"
                return False
            self.attempts += 1
            return True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests": self.requests,
                "attempts": self.attempts,
                "max_requests": self.max_requests,
                "max_attempts": self.max_attempts,
                "exhausted_reason": self.exhausted_reason,
            }


class BaseClient:
    def __init__(self, model: str, temperature: float = 1.0, batch_size: int = 5,
                 cache_dir: str | None = None, budget: ProviderBudget | None = None,
                 max_retries: int = 10) -> None:
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.budget = budget
        self.max_retries = max(1, int(max_retries))
        self._call_counter = 0
        self._cache_lock = Lock()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        raise NotImplementedError

    def _single_call(self, messages: list[dict], temperature: float) -> str:
        """Single API call with retry and shared provider budget, returns content string."""
        time.sleep(random() * 0.5)
        if self.budget is not None and not self.budget.try_request():
            logger.warning(
                "Provider request budget exhausted (%s); skipping call",
                self.budget.exhausted_reason,
            )
            return "Error: provider request budget exhausted"
        for attempt in range(self.max_retries):
            if self.budget is not None and not self.budget.try_attempt():
                logger.warning(
                    "Provider attempt budget exhausted (%s); skipping call",
                    self.budget.exhausted_reason,
                )
                return "Error: provider attempt budget exhausted"
            try:
                choices = self._chat_completion_api(messages, temperature, n=1)
                return choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        logger.error("All retry attempts exhausted")
        return "Error: API call failed after retries"

    def _log_to_cache(self, messages: list[dict], temperature: float, response: str):
        if not self.cache_dir:
            return
        with self._cache_lock:
            self._call_counter += 1
            call_id = self._call_counter
        record = {
            "id": call_id,
            "timestamp": datetime.now().astimezone().isoformat(),
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
            "response": response,
        }
        filename = f"{call_id:04d}_{uuid4().hex}.json"
        path = os.path.join(self.cache_dir, filename)
        with open(path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def prompt(self, expertise: str, message: str, temperature: float | None = None) -> str:
        """
        Single prompt call
        Args:
            expertise: system role content
            message: user message content
            temperature: sampling temperature (defaults to self.temperature)
        """
        temperature = self.temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": expertise},
            {"role": "user", "content": message},
        ]
        response = self._single_call(messages, temperature)
        self._log_to_cache(messages, temperature, response)
        return response

    def prompt_batch(self, expertise: str, message_batch: list[str], temperature: float | None = None) -> list[str]:
        """
        Parallel batch prompt call.
        Args:
            expertise: system role content (shared across batch)
            message_batch: list of user messages
            temperature: sampling temperature (defaults to self.temperature)
        """
        temperature = self.temperature if temperature is None else temperature

        if not message_batch:
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(message_batch)) as executor:
            futures = {
                executor.submit(self.prompt, expertise, msg, temperature): i
                for i, msg in enumerate(message_batch)
            }
            results = [None] * len(message_batch)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results
