import http.client
import json
import time
from urllib.parse import urlparse

# HTTP statuses worth retrying; anything else is a configuration, auth or
# quota problem that will not resolve by immediately retrying.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        # Support a comma-separated fallback list, e.g.
        # "ag/gemini-3.8-flash-medium,ag/gemini-3.5-flash-low". When a model
        # is unavailable (503/404/5xx) the next one in the list is tried.
        self.models = [m.strip() for m in str(model_LLM).split(",") if m.strip()] or [model_LLM]
        self.model_index = 0
        self.debug_mode = debug_mode
        self.n_trial = 6                 # attempts per model when a single model is configured
        self.retry_attempts_per_model = 2  # attempts per model when falling back across models
        self.n_passes = 2                # full passes over the model list before giving up
        self.retry_base_delay = 5
        self.retry_max_delay = 60
        self.last_error = None
        self.host, self.path, self.scheme = self._parse_endpoint(api_endpoint)

    def _sleep_before_retry(self, retry_after, attempt):
        """Exponential backoff for transient failures (429/5xx).

        Honors a Retry-After header when the server sends one.
        """
        delay = min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except (TypeError, ValueError):
                pass
        if delay > 0:
            print(f"   retrying in {delay:g}s ...")
            time.sleep(delay)

    @staticmethod
    def _parse_endpoint(api_endpoint):
        """Split an endpoint into (host, path, scheme).

        Accepts a bare host ("api.deepseek.com"), a full chat-completions URL
        ("https://host/v1/chat/completions"), or an OpenAI-compatible base URL
        ("http://localhost:20128/v1"). A base URL is completed to
        "/chat/completions"; a bare host falls back to the default
        chat-completions path. The scheme is preserved so plain-HTTP gateways
        (e.g. a local router) are not forced through TLS.
        """
        endpoint = (api_endpoint or "").strip()
        if "://" not in endpoint:
            endpoint = "https://" + endpoint

        parsed = urlparse(endpoint)
        scheme = parsed.scheme or "https"
        host = parsed.netloc or parsed.path
        path = parsed.path.rstrip("/")

        if not path:
            path = "/compatible-mode/v1/chat/completions"
        elif not path.endswith("/chat/completions"):
            # Treat the given path as an OpenAI-compatible base URL
            # (e.g. "/v1") and append the chat-completions route.
            path += "/chat/completions"

        if parsed.query:
            path += "?" + parsed.query

        return host, path, scheme

    def _headers(self):
        return {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": 1,
        }

    @staticmethod
    def _extract_content(json_data, raw):
        """Pull the assistant text out of a response.

        Handles both a normal JSON body (choices[0].message.content) and an
        SSE stream (lines starting with "data:", ending with "data: [DONE]")
        that some gateways return even when streaming was not requested.
        """
        # Normal (non-streaming) JSON body.
        if isinstance(json_data, dict):
            choices = json_data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                delta = first.get("delta") or {}
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    return delta["content"]

        # Server-sent events (streaming) body.
        if raw:
            text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
            if "data:" in text:
                parts = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        obj = json.loads(chunk)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not isinstance(choices, list) or not choices:
                        continue
                    first = choices[0] or {}
                    delta = first.get("delta") or {}
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        parts.append(delta["content"])
                    else:
                        message = first.get("message") or {}
                        if isinstance(message, dict) and isinstance(message.get("content"), str):
                            parts.append(message["content"])
                joined = "".join(parts)
                if joined:
                    return joined

        return None

    def _request_model(self, model, prompt_content, max_attempts):
        """POST one request with `model`, retrying transient failures.

        Returns (content, outcome) where outcome is one of:
          "ok"    - a usable completion was returned
          "hard"  - a non-recoverable error (auth/config); stop entirely
          "model" - this model is unavailable (503/5xx/404); try the next one
        """
        payload = json.dumps(
            {
                "model": model,
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
                # Ask for a single JSON response. Some gateways (e.g. 9router)
                # stream chunks by default; this makes the intent explicit.
                "stream": False,
            }
        )
        headers = self._headers()

        for attempt in range(1, max_attempts + 1):
            data = b""
            try:
                if self.scheme == "http":
                    conn = http.client.HTTPConnection(self.host)
                else:
                    conn = http.client.HTTPSConnection(self.host)
                conn.request("POST", self.path, payload, headers)
                res = conn.getresponse()
                data = res.read()

                # Error responses are not always JSON (e.g. an empty api_key
                # makes the server return a plain-text 401 body). Parse
                # leniently so we surface the real status instead of masking it
                # with a JSONDecodeError.
                try:
                    json_data = json.loads(data)
                except (json.JSONDecodeError, ValueError):
                    json_data = None

                if res.status == 200:
                    content = self._extract_content(json_data, data)
                    if content is not None:
                        return content, "ok"
                    self.last_error = self._describe_error(res.status, json_data, data)
                    print(self.last_error)
                    if attempt < max_attempts:
                        self._sleep_before_retry(res.getheader("Retry-After"), attempt)
                    continue

                self.last_error = self._describe_error(res.status, json_data, data)
                print(self.last_error)
                if res.status in (401, 403):
                    return None, "hard"
                if res.status not in _RETRYABLE_STATUS:
                    return None, "model"
                if attempt < max_attempts:
                    self._sleep_before_retry(res.getheader("Retry-After"), attempt)
                continue
            except Exception as e:
                self.last_error = f">> LLM API request failed ({type(e).__name__}): {e}"
                if self.debug_mode:
                    print(self.last_error)
                    print(f"   endpoint: {self.host}{self.path}")
                    print(f"   response: {data[:500]}")
                if attempt < max_attempts:
                    self._sleep_before_retry(None, attempt)
                continue

        return None, "model"

    def get_response(self, prompt_content):
        single = len(self.models) == 1
        max_attempts = self.n_trial if single else self.retry_attempts_per_model
        n_passes = 1 if single else self.n_passes
        self.last_error = None

        for _ in range(n_passes):
            for offset in range(len(self.models)):
                idx = (self.model_index + offset) % len(self.models)
                model = self.models[idx]
                response, outcome = self._request_model(model, prompt_content, max_attempts)

                if outcome == "ok":
                    self.model_index = idx  # stick with the model that worked
                    return response
                if outcome == "hard":
                    return None
                if len(self.models) > 1:
                    nxt = self.models[(idx + 1) % len(self.models)]
                    print(f"   model '{model}' unavailable, falling back to '{nxt}'")

        return None

    @staticmethod
    def _describe_error(status, json_data, raw):
        """Build a readable message from an error response returned by the API."""
        message = None
        error = json_data.get("error") if isinstance(json_data, dict) else None
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error

        if message is None and isinstance(json_data, list) and json_data:
            first = json_data[0]
            if isinstance(first, dict) and isinstance(first.get("error"), dict):
                message = first["error"].get("message")

        if message is None:
            message = raw[:300].decode("utf-8", "replace")

        hint = ""
        if status == 429:
            hint = (" [rate limit / quota exhausted - wait for the quota to reset "
                    "or use a model with a higher limit]")
        elif status in (401, 403):
            hint = " [check llm_api_key]"
        elif status == 404:
            hint = " [check llm_api_endpoint and llm_model]"

        return f">> LLM API error (HTTP {status}): {message}{hint}"
