"""Provider factory tests — no network calls."""

from __future__ import annotations

import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from fast_rag.config import Config
from fast_rag.providers import make_llm


def test_ollama_factory_returns_chatollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_RAG_PROVIDER", "ollama")
    llm = make_llm(Config.from_env())
    assert isinstance(llm, ChatOllama)


def test_bailian_factory_returns_chatopenai_with_dashscope_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_RAG_PROVIDER", "bailian")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    llm = make_llm(Config.from_env())
    assert isinstance(llm, ChatOpenAI)
    assert "dashscope" in str(llm.openai_api_base)


def test_bailian_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_RAG_PROVIDER", "bailian")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        make_llm(Config.from_env())
