"""
Data-sovereignty test: running generation on a fake document chunk must
never make a real network connection off the host.

Real sockets are blocked for the whole test (autouse fixture below), so if
any code path bypassed the requests.post mock — a bug, a forgotten patch,
a library silently falling back to a raw connection — the test fails
loudly instead of quietly reaching a real server.
"""

import socket

import pytest

from rag import config
from rag.generate import answer
from rag.retrieve import RetrievedChunk


@pytest.fixture(autouse=True)
def block_real_sockets(monkeypatch):
    """Make any real outbound connection attempt during this test raise,
    instead of silently going out over the network."""

    def _blocked_connect(self, address):
        raise AssertionError(f"outbound network connection attempted: {address}")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)


class _FakeResponse:
    """Minimal stand-in for requests.Response — just what generate.py reads."""

    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_answer_only_calls_the_local_gateway(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return _FakeResponse({
            "choices": [{"message": {"content": "réponse factice [1]"}}],
            "usage": {"total_tokens": 12},
        })

    # generate.answer() must go through requests.post, and only through it —
    # block_real_sockets above catches anything that doesn't.
    monkeypatch.setattr("rag.generate.requests.post", fake_post)

    fake_chunk = RetrievedChunk(
        text="Ceci est un chunk de document factice pour le test.",
        score=0.99,
        source_file="fake.pdf",
        page_number=1,
        chunk_index=0,
    )

    result = answer("Question de test ?", [fake_chunk])

    assert result["answer"] == "réponse factice [1]"
    # The only call made must target the configured local gateway —
    # never a hardcoded or leaked external URL.
    assert calls == [f"{config.LITELLM_URL}/v1/chat/completions"]
