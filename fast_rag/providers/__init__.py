"""Provider factories: dispatch LLM and Embedding construction by Config.provider."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from fast_rag.config import Config

from . import bailian, ollama


def make_llm(cfg: Config) -> BaseChatModel:
    if cfg.provider == "ollama":
        return ollama.make_llm(cfg)
    if cfg.provider == "bailian":
        return bailian.make_llm(cfg)
    raise ValueError(
        f"未知 provider: {cfg.provider!r}(支持 ollama / bailian)"
    )


def make_embeddings(cfg: Config) -> Embeddings:
    if cfg.provider == "ollama":
        return ollama.make_embeddings(cfg)
    if cfg.provider == "bailian":
        return bailian.make_embeddings(cfg)
    raise ValueError(
        f"未知 provider: {cfg.provider!r}(支持 ollama / bailian)"
    )


__all__ = ["make_llm", "make_embeddings"]
