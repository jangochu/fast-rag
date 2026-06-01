"""Build a FAISS index from .md / .txt files under a data directory."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config


def build_index(
    data_dir: Path | str,
    index_dir: Path | str,
    rebuild: bool = False,
    config: Config | None = None,
) -> int:
    """Build a FAISS index from .md/.txt files under data_dir, save to index_dir.

    Returns the number of chunks indexed.
    """
    cfg = config or Config.from_env()
    data_dir = Path(data_dir)
    index_dir = Path(index_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)

    documents = []
    for ext in ("md", "txt"):
        loader = DirectoryLoader(
            str(data_dir),
            glob=f"**/*.{ext}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No .md or .txt files found under {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embed_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))

    return len(chunks)
