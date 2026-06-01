"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class OllamaSettings:
    llm_model: str
    embed_model: str
    url: str


@dataclass(frozen=True)
class BailianSettings:
    api_key: str | None
    llm_model: str
    embed_model: str
    base_url: str


@dataclass(frozen=True)
class Config:
    provider: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    ollama: OllamaSettings
    bailian: BailianSettings

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        ollama = OllamaSettings(
            llm_model=os.getenv("FAST_RAG_OLLAMA_LLM", "qwen2.5:7b"),
            embed_model=os.getenv(
                "FAST_RAG_OLLAMA_EMBED",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            url=os.getenv("FAST_RAG_OLLAMA_URL", "http://localhost:11434"),
        )
        bailian = BailianSettings(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            llm_model=os.getenv("FAST_RAG_BAILIAN_LLM", "qwen-plus"),
            embed_model=os.getenv("FAST_RAG_BAILIAN_EMBED", "text-embedding-v3"),
            base_url=os.getenv(
                "FAST_RAG_BAILIAN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        return cls(
            provider=os.getenv("FAST_RAG_PROVIDER", "ollama").lower(),
            chunk_size=int(os.getenv("FAST_RAG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("FAST_RAG_CHUNK_OVERLAP", "50")),
            top_k=int(os.getenv("FAST_RAG_TOP_K", "4")),
            ollama=ollama,
            bailian=bailian,
        )
