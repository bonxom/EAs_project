import json
from types import SimpleNamespace

from utils.llm_client.base import BaseClient


class FakeLLM(BaseClient):
    def _chat_completion_api(self, messages, temperature, n=1):
        content = messages[-1]["content"]
        return [SimpleNamespace(message=SimpleNamespace(content=content))]


def test_cache_keeps_responses_from_clients_sharing_a_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.llm_client.base.time.sleep", lambda _: None)
    clients = [FakeLLM("fake", cache_dir=str(tmp_path)) for _ in range(2)]
    clients[0].prompt("expert", "first")
    clients[1].prompt("expert", "second")
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert sorted(record["response"] for record in records) == ["first", "second"]


def test_batch_cache_retains_all_responses_and_zero_temperature(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.llm_client.base.time.sleep", lambda _: None)
    llm = FakeLLM("fake", temperature=1, cache_dir=str(tmp_path))
    messages = [f"response-{i}" for i in range(20)]
    assert llm.prompt_batch("expert", messages, temperature=0) == messages
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(records) == 20
    assert {record["id"] for record in records} == set(range(1, 21))
    assert all(record["temperature"] == 0 for record in records)


def test_empty_batch_returns_empty_list():
    assert FakeLLM("fake").prompt_batch("expert", []) == []
