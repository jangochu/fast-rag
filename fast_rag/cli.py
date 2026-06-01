"""Command-line entry point: `python -m fast_rag.cli {ingest,ask}`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .ingest import build_index
from .rag import answer

DEFAULT_DATA_DIR = Path("data")
DEFAULT_INDEX_DIR = Path("index")


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    print(f"Building index from {args.data_dir} → {args.index_dir} ...")
    if args.rebuild:
        print("  (--rebuild: existing index will be cleared)")
    n_chunks = build_index(
        args.data_dir,
        args.index_dir,
        rebuild=args.rebuild,
        config=cfg,
    )
    print(f"Indexed {n_chunks} chunks. Done.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    if not args.index_dir.exists():
        print(
            f"索引未找到 ({args.index_dir}),请先运行 'python -m fast_rag.cli ingest'",
            file=sys.stderr,
        )
        return 2

    try:
        result = answer(args.question, args.index_dir, k=args.k, config=cfg)
    except Exception as e:
        msg = str(e).lower()
        if any(token in msg for token in ("connection refused", "connecterror", "max retries", "failed to connect")):
            print(
                f"无法连接 Ollama ({cfg.ollama_url}),请先运行 'ollama serve'",
                file=sys.stderr,
            )
            return 3
        if "model" in msg and any(token in msg for token in ("not found", "does not exist", "try pulling")):
            print(
                f"模型 '{cfg.llm_model}' 不存在,请运行 'ollama pull {cfg.llm_model}'",
                file=sys.stderr,
            )
            return 4
        raise

    print("答案:")
    print(result["answer"])
    if args.show_sources:
        print()
        print("参考来源:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fast-rag",
        description="A simple, fully-local RAG demo (FAISS + sentence-transformers + Ollama).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Build FAISS index from data/")
    p_ingest.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_ingest.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_ingest.add_argument(
        "--rebuild", action="store_true", help="Clear existing index first"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question against the index")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_ask.add_argument("--k", type=int, default=None, help="Override top-k")
    p_ask.add_argument(
        "--no-sources",
        dest="show_sources",
        action="store_false",
        help="Suppress reference sources output",
    )
    p_ask.set_defaults(func=cmd_ask, show_sources=True)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
