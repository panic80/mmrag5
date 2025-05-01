"""Tests covering the high-severity fixes recently implemented."""

from __future__ import annotations

import importlib
import os
import types
from uuid import UUID


import pytest


# ---------------------------------------------------------------------------
# Token validation hardening
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_token_env(monkeypatch):
    """Ensure token env vars are cleared before every test."""

    for var in ("SLASH_TOKEN", "SLASH_TOKEN_INJECT", "SLASH_TOKEN_ASK"):
        monkeypatch.delenv(var, raising=False)


def _make_test_client(monkeypatch, token_env: str | None):
    """Return Flask test client with optional token env set; stub threads."""

    # (Re)load the server module so that it picks up monkey-patched env vars.
    if "server" in globals():
        importlib.reload(globals()["server"])  # type: ignore[arg-type]
    else:
        import server  # noqa: F401

    # Set env var if provided
    if token_env is not None:
        # Set both generic and command-specific token so that requests work
        for var in ("SLASH_TOKEN", "SLASH_TOKEN_INJECT"):
            monkeypatch.setenv(var, token_env)

    import server as _srv  # import after env applied

    # Prevent background threads from spawning during tests
    class _DummyThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    monkeypatch.setattr(_srv.threading, "Thread", _DummyThread)

    return _srv.app.test_client()


def test_missing_token_returns_503(monkeypatch):
    client = _make_test_client(monkeypatch, token_env=None)
    resp = client.post("/inject", data={"token": "anything", "command": "/inject"})
    assert resp.status_code == 503
    assert b"slash token not set" in resp.data.lower()


def test_invalid_token_returns_403(monkeypatch):
    client = _make_test_client(monkeypatch, token_env="secret")
    resp = client.post("/inject", data={"token": "wrong", "command": "/inject"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Deterministic UUID generation – no collisions & reproducible
# ---------------------------------------------------------------------------


def test_deterministic_uuid_generation(monkeypatch):
    from ingest_rag import embed_and_upsert, Document

    # Dummy OpenAI client producing predictable embeddings
    class _Embeddings:
        @staticmethod
        def create(model, input):  # noqa: D401
            class _Resp:
                data = [types.SimpleNamespace(embedding=[0.0] * 10) for _ in input]

            return _Resp()

    dummy_openai = types.SimpleNamespace(embeddings=_Embeddings())

    # Dummy Qdrant client capturing points
    class _DummyClient:
        def __init__(self):
            self.captured = []

        def upsert(self, collection_name, points):  # noqa: D401
            self.captured.extend(points)

    client = _DummyClient()

    docs = [Document(content="hello world", metadata={"idx": 1})]

    # First insert – capture ID
    embed_and_upsert(client, "test", docs, dummy_openai, batch_size=1, deterministic_id=True)
    pid_first = client.captured[0].id

    # Second insert with *identical* payload should repeat the ID
    client.captured.clear()
    embed_and_upsert(client, "test", docs, dummy_openai, batch_size=1, deterministic_id=True)
    pid_second = client.captured[0].id

    assert pid_first == pid_second  # reproducible

    # Now change *chunk_text* (content) – ID must change
    client.captured.clear()
    docs2 = [Document(content="different text", metadata={"idx": 1})]
    embed_and_upsert(client, "test", docs2, dummy_openai, batch_size=1, deterministic_id=True)
    pid_diff = client.captured[0].id

    assert pid_first != pid_diff

    # Ids must be valid UUID strings
    UUID(pid_first)
    UUID(pid_diff)