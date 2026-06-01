"""Load the FAISS index and run a retrieval-augmented chain via the configured provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import Config
from .providers import make_embeddings, make_llm

PROMPT_TEMPLATE = """你是一个基于检索内容回答问题的助手。请严格根据下列上下文回答问题。
如果上下文里没有答案,直接说"根据已有资料无法回答"。回答语言与问题保持一致。

上下文:
{context}

问题: {question}

回答:"""


def load_vector_store(index_dir: Path | str, config: Config | None = None) -> FAISS:
    cfg = config or Config.from_env()
    embeddings = make_embeddings(cfg)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_retriever(
    index_dir: Path | str,
    k: int | None = None,
    config: Config | None = None,
):
    cfg = config or Config.from_env()
    vs = load_vector_store(index_dir, cfg)
    return vs.as_retriever(search_kwargs={"k": k or cfg.top_k})


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def answer(
    question: str,
    index_dir: Path | str,
    k: int | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """Answer a question using retrieved context. Returns {answer, sources}."""
    cfg = config or Config.from_env()
    retriever = load_retriever(index_dir, k, cfg)
    docs = retriever.invoke(question)

    llm = make_llm(cfg)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    answer_text = chain.invoke(
        {"context": _format_docs(docs), "question": question}
    )

    sources = [d.metadata.get("source", "<unknown>") for d in docs]
    return {"answer": answer_text, "sources": sources}
