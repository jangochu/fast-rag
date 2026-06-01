# Fast-RAG 样例文档

## 关于本项目

Fast-RAG 是一个最小可运行的 RAG (Retrieval-Augmented Generation) 演示工程。它把
本地 Markdown / 文本文件切分、嵌入并写入 FAISS 索引,之后通过命令行向本地
Ollama 上托管的大语言模型提问,模型会基于检索到的上下文作答。

整个流程不依赖任何外部 API:embedding 模型来自 HuggingFace 上的
sentence-transformers,LLM 由 Ollama 在本机运行。第一次启动会下载 embedding
模型(约 400MB),之后所有操作完全离线。

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines a vector
retriever with a large language model. Instead of asking the LLM to recall
facts from its training data, RAG first fetches the most relevant passages
from a private corpus and feeds them into the prompt. The model then answers
grounded in the retrieved context.

This pattern reduces hallucination, lets you update knowledge without
fine-tuning the model, and works well for FAQ bots, internal documentation
search, and code-base Q&A.

## 工程组成

- **ingest**:把 `data/` 下的文档切片、嵌入,写到 `index/`
- **ask**:从 `index/` 召回 top-k 片段,交给 Ollama LLM 作答
- **配置**:所有可调参数通过环境变量覆盖,详见 `.env.example`
