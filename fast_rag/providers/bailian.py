"""Bailian / DashScope provider via OpenAI-compatible interface."""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from fast_rag.config import Config


def _require_api_key(cfg: Config) -> str:
    if not cfg.bailian.api_key:
        raise ValueError(
            "provider=bailian 需要设置环境变量 DASHSCOPE_API_KEY"
        )
    return cfg.bailian.api_key


def make_llm(cfg: Config) -> ChatOpenAI:
    api_key = _require_api_key(cfg)
    return ChatOpenAI(
        model=cfg.bailian.llm_model,
        api_key=api_key,
        base_url=cfg.bailian.base_url,
    )


def make_embeddings(cfg: Config) -> OpenAIEmbeddings:
    api_key = _require_api_key(cfg)
    return OpenAIEmbeddings(
        model=cfg.bailian.embed_model,
        api_key=api_key,
        base_url=cfg.bailian.base_url,
        # DashScope expects raw strings; default OpenAI path sends tiktoken int arrays.
        check_embedding_ctx_length=False,
        # text-embedding-v3 caps single request at 25 inputs.
        chunk_size=10,
    )
