# Fast-RAG Demo 设计文档

- 日期: 2026-06-01
- 状态: 已通过头脑风暴确认,待进入实施计划

## 目标

在空工程下用 Python 实现一个**全本地、简单、可复现**的 RAG demo。强调"跑得起、看得懂、可改写",非生产级。

## 关键决策与权衡

| 维度 | 选择 | 理由 |
|---|---|---|
| 模型栈 | 全本地: sentence-transformers + Ollama | 零 API key、零费用、纯离线可演示 |
| 数据格式 | 纯文本 / Markdown | loader 最简,demo 首选 |
| 交互 | CLI (argparse) | 调试/集成最方便,不引入 typer/click |
| 向量库 | FAISS (本地落盘) | 高性能、依赖单一、社区成熟 |
| 语言 | 中英混合 | 用多语言 embedding 兼顾 |
| LLM | Ollama,默认 qwen2.5:7b,可 env 覆盖 | 不锁死,演示和实战都灵活 |
| 代码组织 | LangChain 封装 + 模块化包结构 | 用户偏好生态丰富、后续扩展易 |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (~400MB) | 比 bge-m3 (~2.3GB) 轻,demo 体量合适 |

## 整体架构与数据流

```
[ingest 流程 - 离线一次性]
  data/*.md|*.txt
      ↓ DirectoryLoader
  Documents
      ↓ RecursiveCharacterTextSplitter (chunk=500, overlap=50)
  Chunks
      ↓ HuggingFaceEmbeddings (多语言模型)
  Vectors
      ↓ FAISS.from_documents(...).save_local()
  index/  (faiss.index + docstore.pkl)

[ask 流程 - 每次问答]
  用户问题
      ↓
  FAISS.load_local() → as_retriever(k=4)
      ↓ 召回 top-4 chunks
  Prompt 模板 ("根据下列上下文回答...{context}\n问题: {question}")
      ↓ LCEL: retriever | prompt | ChatOllama | StrOutputParser
  ChatOllama (默认 qwen2.5:7b,可 env 覆盖)
      ↓
  CLI 输出: 答案 + 来源文件名列表
```

要点:
- ingest 与 ask 完全解耦,索引落盘后重启不丢
- 用 LCEL (`|` 管道) 写 chain,符合 LangChain 0.3 推荐写法
- 检索结果同时透传 `source` 元数据,答案后附"参考文档"清单

## 目录结构与模块职责

```
fast-rag/
├── README.md              # 启动 Ollama + 拉模型 + 运行的 3 步说明
├── pyproject.toml         # 依赖声明
├── .env.example           # 环境变量样例
├── .gitignore             # 忽略 index/、.env、模型缓存
├── data/                  # 用户放 .md/.txt (自带 1-2 个样例)
│   └── sample.md
├── index/                 # FAISS 落盘目录 (git 忽略)
├── fast_rag/
│   ├── __init__.py
│   ├── config.py          # 读 env,集中管理:模型名、路径、chunk 参数
│   ├── ingest.py          # build_index(data_dir, index_dir) -> None
│   ├── rag.py             # load_chain(index_dir) -> LCEL Runnable
│   └── cli.py             # argparse 两个子命令: ingest / ask
└── tests/
    └── test_smoke.py      # mock LLM,验证 ingest + retriever 通路
```

模块边界:

| 模块 | 单一职责 | 不做的事 |
|---|---|---|
| `config.py` | 把 env 收成 dataclass,所有默认值在这里 | 不调用任何模型 |
| `ingest.py` | 读文件 → 切分 → 嵌入 → 写 FAISS | 不处理问答 |
| `rag.py` | 装配 retriever + prompt + LLM 成 chain | 不写 CLI 解析 |
| `cli.py` | 解析 argv,调上面三个 | 不写业务逻辑 |

### .env.example

```
FAST_RAG_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
FAST_RAG_LLM_MODEL=qwen2.5:7b
FAST_RAG_OLLAMA_URL=http://localhost:11434
FAST_RAG_CHUNK_SIZE=500
FAST_RAG_CHUNK_OVERLAP=50
FAST_RAG_TOP_K=4
```

## CLI 设计

```bash
# 1. 建索引 (首次或数据更新后)
python -m fast_rag.cli ingest
# 可选: --data-dir ./data  --index-dir ./index  --rebuild

# 2. 提问
python -m fast_rag.cli ask "什么是 RAG?"
# 可选: --k 6  --show-sources/--no-sources
```

典型输出:

```
答案:
RAG (Retrieval-Augmented Generation) 是一种把外部知识检索结果...

参考来源:
  [1] data/intro.md
  [2] data/architecture.md
  [3] data/intro.md
```

首次运行旅程(README 写清这 4 步):
1. `ollama serve` 启动服务,`ollama pull qwen2.5:7b` 拉模型
2. `uv sync` 或 `pip install -e .` 装依赖
3. `python -m fast_rag.cli ingest` 建索引(首次会下载 embedding 模型 ~400MB)
4. `python -m fast_rag.cli ask "你的问题"`

## 刻意不做 (YAGNI)

- ❌ 交互式 REPL / 对话历史 — 单轮问答足够 demo
- ❌ 流式输出 — CLI 一次性打印更简洁
- ❌ 增量 ingest — `--rebuild` 直接清掉重建,简单可靠
- ❌ 多用户 / 会话管理 — 不在 demo 范围
- ❌ Reranker / MMR / Hybrid Search — 留作后续扩展

## 错误处理

只覆盖用户最容易踩的 3 个坑,每个一句人话提示:

| 场景 | 检测时机 | 提示 |
|---|---|---|
| Ollama 没启动 | `ask` 调 LLM 时捕 `ConnectionError` | `无法连接 Ollama (http://localhost:11434),请先运行 'ollama serve'` |
| 模型未拉取 | Ollama 返回 model not found | `模型 'qwen2.5:7b' 不存在,请运行 'ollama pull qwen2.5:7b'` |
| 索引不存在 | `ask` 启动检查 `index/` 目录 | `索引未找到,请先运行 'python -m fast_rag.cli ingest'` |

其他异常(磁盘满、文件不可读)直接冒泡,demo 不必伪装健壮。

## 测试策略

只写一个 smoke test,验证关键管道不断:

```python
# tests/test_smoke.py
def test_ingest_then_retrieve(tmp_path):
    (tmp_path / "a.md").write_text("RAG 是一种检索增强生成技术")
    (tmp_path / "b.md").write_text("FAISS 是 Facebook 开源的向量库")

    build_index(data_dir=tmp_path, index_dir=tmp_path / "idx")

    retriever = load_retriever(tmp_path / "idx")
    docs = retriever.invoke("什么是 RAG?")
    assert any("检索增强" in d.page_content for d in docs)
```

不写的测试:
- ❌ 不真调 Ollama(CI 没 GPU/模型)
- ❌ 不做答案质量评测(超出 demo)
- ✅ embedding 真跑,首次会下载 400MB,本地可接受

## 关键依赖

- `langchain` >= 0.3, `langchain-community`, `langchain-huggingface`, `langchain-ollama`
- `sentence-transformers` (会自动带 `torch`)
- `faiss-cpu`
- `python-dotenv` (可选,读 `.env`)
- `pytest` (dev)

## 后续可演进的方向(不在本次范围)

- 引入 reranker (bge-reranker-base) 提升精排
- 加 `--stream` 流式输出
- Gradio Web UI
- 切换 vector store 到 Chroma 以支持元数据过滤
- ingest 增量更新 + 文件 hash 去重
