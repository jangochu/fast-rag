"""Ollama provider: ChatOllama + HuggingFaceEmbeddings (local)."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from fast_rag.config import Config


def make_llm(cfg: Config) -> ChatOllama:
    return ChatOllama(model=cfg.ollama.llm_model, base_url=cfg.ollama.url)


def make_embeddings(cfg: Config) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=cfg.ollama.embed_model)
