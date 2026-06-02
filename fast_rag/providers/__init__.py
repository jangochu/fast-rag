"""Provider factories: dispatch LLM and Embedding construction by Config.provider.

Providers are lazy-imported so e.g. choosing bailian never pulls torch/sentence-transformers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

from fast_rag.config import Config

_SUPPORTED = ("ollama", "bailian")


def _load(provider: str):
    if provider not in _SUPPORTED:
        raise ValueError(
            f"未知 provider: {provider!r}(支持 {' / '.join(_SUPPORTED)})"
        )
    return importlib.import_module(f"fast_rag.providers.{provider}")


def make_llm(cfg: Config) -> "BaseChatModel":
    return _load(cfg.provider).make_llm(cfg)


def make_embeddings(cfg: Config) -> "Embeddings":
    return _load(cfg.provider).make_embeddings(cfg)


__all__ = ["make_llm", "make_embeddings"]
