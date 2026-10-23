"""Tests run without an API key by using a fake LLM."""
import pytest
from fastapi.testclient import TestClient

from app import main
from app.conversation import ConversationManager
from app.llm import LLMError


class FakeLLM:
    """Records every call; replies with a summary for summary requests."""

    def __init__(self, fail_on: set[int] | None = None):
        self.calls = []
        self.fail_on = fail_on or set()

    def generate(self, system_prompt, messages):
        self.calls.append((system_prompt, [dict(m) for m in messages]))
        if len(self.calls) in self.fail_on:
            raise LLMError("boom", 503)
        if "conversation summaries" in system_prompt:
            return "SUMMARY: student is Vivek, preparing for placements."
        return f"reply {len(self.calls)}"


def test_history_is_sent_with_each_request():
    llm = FakeLLM()
    m = ConversationManager(llm, max_messages=20, keep_recent=6)
    conv, _ = m.chat(None, "My name is Vivek")
    m.chat(conv.id, "I like graphs")
    m.chat(conv.id, "What's my name?")
    sent = llm.calls[-1][1]
    assert sent[0]["content"] == "My name is Vivek"
    assert sent[-1]["content"] == "What's my name?"
    assert len(sent) == 5  # 2 full turns + new message


def test_conversations_are_isolated():
    llm = FakeLLM()
    m = ConversationManager(llm, max_messages=20, keep_recent=6)
    a, _ = m.chat(None, "I am A")
    m.chat(None, "I am B")
    m.chat(a.id, "who am I?")
    contents = [x["content"] for x in llm.calls[-1][1]]
    assert "I am B" not in contents


def test_truncation_keeps_last_n():
    llm = FakeLLM()
    m = ConversationManager(llm, max_messages=6, keep_recent=4, strategy="truncate")
    conv = None
    for i in range(10):
        conv, _ = m.chat(conv.id if conv else None, f"msg {i}")
        assert len(conv.messages) <= 6
    assert conv.summary == ""
    assert len(llm.calls) == 10  # no extra summary calls


def test_summarization_prepends_summary():
    llm = FakeLLM()
    m = ConversationManager(llm, max_messages=6, keep_recent=4, strategy="summarize")
    conv = None
    for i in range(5):
        conv, _ = m.chat(conv.id if conv else None, f"msg {i}")
    assert conv.summary.startswith("SUMMARY")
    m.chat(conv.id, "what do you remember?")
    chat_calls = [c for c in llm.calls if "conversation summaries" not in c[0]]
    system_prompt, sent = chat_calls[-1]
    assert sent[-1]["content"] == "what do you remember?"
    assert "SUMMARY: student is Vivek" in system_prompt
    assert "Mentor" in system_prompt  # persona kept


def test_summary_failure_falls_back_to_truncation():
    # 4th chat reply is call 4 -> 8 msgs > 6 -> summary request is call 5
    llm = FakeLLM(fail_on={5})
    m = ConversationManager(llm, max_messages=6, keep_recent=4, strategy="summarize")
    conv = None
    for i in range(4):
        conv, _ = m.chat(conv.id if conv else None, f"msg {i}")
    assert len(conv.messages) == 4
    assert conv.summary == ""


def test_failed_call_does_not_corrupt_history():
    llm = FakeLLM(fail_on={2})
    m = ConversationManager(llm, max_messages=20, keep_recent=6)
    conv, _ = m.chat(None, "first")
    with pytest.raises(LLMError):
        m.chat(conv.id, "second")
    assert len(conv.messages) == 2  # failed turn not stored, can retry


def test_api_returns_clean_error(monkeypatch):
    llm = FakeLLM(fail_on={1})
    monkeypatch.setattr(main, "_manager", ConversationManager(llm, 12, 6))
    client = TestClient(main.app)
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 503
    assert r.json() == {"error": "boom"}


def test_api_multi_turn(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(main, "_manager", ConversationManager(llm, 12, 6))
    client = TestClient(main.app)
    r1 = client.post("/api/chat", json={"message": "hi"}).json()
    r2 = client.post("/api/chat", json={"message": "again", "conversation_id": r1["conversation_id"]}).json()
    assert r2["turns"] == 2
    assert r2["messages_in_context"] == 4
