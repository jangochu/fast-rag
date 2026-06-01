"""Smoke test: ingest + retrieve round-trip without touching the LLM.

This test downloads the embedding model (~400MB) on first run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fast_rag.ingest import build_index
from fast_rag.rag import load_retriever


def test_ingest_then_retrieve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAST_RAG_PROVIDER", "ollama")
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "idx"
    data_dir.mkdir()

    (data_dir / "a.md").write_text(
        "RAG 是一种检索增强生成技术,先检索再让大模型回答。", encoding="utf-8"
    )
    (data_dir / "b.md").write_text(
        "FAISS 是 Facebook 开源的向量检索库,适合本地大规模相似度搜索。",
        encoding="utf-8",
    )

    n_chunks = build_index(data_dir, index_dir)
    assert n_chunks >= 2

    retriever = load_retriever(index_dir)
    docs = retriever.invoke("什么是 RAG?")

    assert docs, "retriever returned no docs"
    assert any("检索增强" in d.page_content for d in docs)
