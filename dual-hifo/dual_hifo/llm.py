"""OpenAI-compatible LLM client, an offline mock, and a logging wrapper."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Dict, List, Optional

from .run_logger import RunLogger


class ChatClient:
    model = "unknown"
    temperature = 0.8

    def prompt(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        raise NotImplementedError

    def prompt_batch(self, system: str, users: List[str],
                     temperature: Optional[float] = None) -> List[str]:
        return [self.prompt(system, user, temperature) for user in users]


class OpenAICompatibleClient(ChatClient):
    def __init__(self, *, model: str, base_url: str, api_key: str,
                 temperature: float = 0.8, timeout: int = 180, max_retries: int = 3):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def prompt(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        temperature = self.temperature if temperature is None else temperature
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=payload, headers=headers, method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")


MOCK_HEURISTICS = [
    '''```python
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    if unvisited_nodes.size == 0:
        return int(destination_node)
    return int(unvisited_nodes[int(np.argmin(distance_matrix[current_node][unvisited_nodes]))])
```''',
    '''```python
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    if unvisited_nodes.size == 0:
        return int(destination_node)
    best, best_score = int(unvisited_nodes[0]), float("inf")
    for node in unvisited_nodes:
        others = unvisited_nodes[unvisited_nodes != node]
        future = np.min(distance_matrix[node][others]) if others.size else distance_matrix[node][destination_node]
        score = 0.7 * distance_matrix[current_node][node] + 0.3 * future
        if score < best_score:
            best, best_score = int(node), score
    return best
```''',
    '''```python
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    if unvisited_nodes.size == 0:
        return int(destination_node)
    order = np.argsort(distance_matrix[current_node][unvisited_nodes])
    top = order[:min(3, len(order))]
    pick = top[(current_node + len(unvisited_nodes)) % len(top)]
    return int(unvisited_nodes[int(pick)])
```''',
]

MOCK_IMPROVER = '''# {Mock improver: ask the LLM for a new TSP heuristic and score it.}
def improve_algorithm(population, utility, language_model, function_format, task):
    parent = population.get_random_solution(task)
    prompt = (
        "[TSP_HEURISTIC] Improve this heuristic.\\n"
        + parent["best_sol"] + "\\n" + function_format
    )
    response = language_model.prompt("TSP expert.", prompt)
    code = extract_code(response)
    idea = extract_idea(response) or "mock candidate"
    return idea, code, utility(code, idea, task)
'''


class MockClient(ChatClient):
    model = "mock-llm"

    def __init__(self, temperature: float = 0.8):
        self.temperature = temperature
        self._heuristic_index = 0
        self._improver_index = 0

    def prompt(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        if "[META_IMPROVER]" in user:
            self._improver_index += 1
            return (f"{{Mock improver {self._improver_index}.}}\n\n"
                    f"```python\n{MOCK_IMPROVER}```")
        if "[TSP_HEURISTIC]" in user:
            code = MOCK_HEURISTICS[self._heuristic_index % len(MOCK_HEURISTICS)]
            self._heuristic_index += 1
            return f"{{Mock heuristic {self._heuristic_index}.}}\n\n{code}"
        return "- Balance immediate cost with future connectivity."


class LoggedLLM:
    def __init__(self, client: ChatClient, logger: RunLogger, loop: str,
                 tag_prefix: str = "llm"):
        self.client = client
        self.logger = logger
        self.loop = loop
        self.tag_prefix = tag_prefix
        self.calls = 0

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "unknown")

    @property
    def temperature(self) -> float:
        return getattr(self.client, "temperature", 0.8)

    def prompt(self, system: str, user: str, temperature: Optional[float] = None,
               tag: Optional[str] = None) -> str:
        self.calls += 1
        tag = tag or f"{self.tag_prefix}_{self.calls}"
        temperature = self.temperature if temperature is None else temperature
        try:
            response = self.client.prompt(system, user, temperature=temperature)
        except Exception as exc:
            self.logger.log_llm(loop=self.loop, tag=tag, model=self.model,
                                temperature=temperature, system=system, user=user,
                                response="", error=str(exc))
            raise
        self.logger.log_llm(loop=self.loop, tag=tag, model=self.model,
                            temperature=temperature, system=system, user=user,
                            response=response)
        return response

    def prompt_batch(self, system: str, users: List[str],
                     temperature: Optional[float] = None) -> List[str]:
        return [self.prompt(system, user, temperature=temperature) for user in users]


def build_client(cfg: dict, mock: bool) -> ChatClient:
    if mock:
        return MockClient(temperature=cfg.get("temperature", 0.8))

    api_key = str(cfg.get("api_key") or "").strip()
    if not api_key:
        for name in (cfg.get("api_key_env", "LLM_API_KEY"), "LLM_API_KEY",
                     "GEMINI_API_KEY", "OPENAI_API_KEY"):
            if name and os.environ.get(name):
                api_key = os.environ[name].strip()
                break
    if not api_key:
        raise RuntimeError(
            "No API key. Set LLM_API_KEY in .env, or pass --api-key. "
            "Use --mock for an offline test."
        )

    return OpenAICompatibleClient(
        model=cfg.get("model", "ag/gemini-3.7-flash-low"),
        base_url=cfg.get("base_url", "http://localhost:20128/v1"),
        api_key=api_key,
        temperature=cfg.get("temperature", 0.8),
        timeout=cfg.get("timeout", 180),
        max_retries=cfg.get("max_retries", 3),
    )
