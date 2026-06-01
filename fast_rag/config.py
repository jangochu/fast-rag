"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    embed_model: str
    llm_model: str
    ollama_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            embed_model=os.getenv(
                "FAST_RAG_EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            llm_model=os.getenv("FAST_RAG_LLM_MODEL", "qwen2.5:7b"),
            ollama_url=os.getenv("FAST_RAG_OLLAMA_URL", "http://localhost:11434"),
            chunk_size=int(os.getenv("FAST_RAG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("FAST_RAG_CHUNK_OVERLAP", "50")),
            top_k=int(os.getenv("FAST_RAG_TOP_K", "4")),
        )
