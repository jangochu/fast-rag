# Fast-RAG

一个**全本地、最小可运行**的 RAG (Retrieval-Augmented Generation) 演示工程。
零 API key,零费用 — 用 sentence-transformers 做嵌入,FAISS 存向量,
Ollama 在本机跑大语言模型。

## 它做了什么

把 `data/` 下的 `.md` / `.txt` 文件切片、嵌入并写入 FAISS 索引;之后通过
命令行问问题,模型基于检索到的上下文作答,并附上参考文档来源。

## 快速开始(4 步)

### 1. 启动 Ollama 并拉一个模型

```bash
ollama serve                 # 另开终端,后台运行
ollama pull qwen2.5:7b       # 中文友好的 7B 模型,约 5GB
```

### 2. 安装依赖

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# 或者用普通 pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. 建索引

```bash
python -m fast_rag.cli ingest
# 首次运行会从 HuggingFace 下载 embedding 模型 (~400MB)
```

### 4. 提问

```bash
python -m fast_rag.cli ask "样例文档讲了什么?"
```

输出示例:

```
答案:
样例文档介绍了 Fast-RAG 项目 ......

参考来源:
  [1] data/sample.md
  [2] data/sample.md
```

## 配置(环境变量)

所有可调参数都在 `.env.example`,复制成 `.env` 后按需修改:

| 变量 | 默认 | 说明 |
|---|---|---|
| `FAST_RAG_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace embedding 模型名 |
| `FAST_RAG_LLM_MODEL` | `qwen2.5:7b` | Ollama 上的模型名 |
| `FAST_RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `FAST_RAG_CHUNK_SIZE` | `500` | 切片字符数 |
| `FAST_RAG_CHUNK_OVERLAP` | `50` | 切片重叠字符数 |
| `FAST_RAG_TOP_K` | `4` | 召回片段数量 |
| `HF_ENDPOINT` | (未设置) | HuggingFace 下载镜像,如 `https://hf-mirror.com` |

> 若 `huggingface.co` 不通(常见于国内网络),设置
> `export HF_ENDPOINT=https://hf-mirror.com` 后重试。

## 命令参考

```bash
# 建索引
python -m fast_rag.cli ingest [--data-dir DIR] [--index-dir DIR] [--rebuild]

# 提问
python -m fast_rag.cli ask "你的问题" [--index-dir DIR] [--k N] [--no-sources]
```

## 项目结构

```
fast-rag/
├── data/             # 放你的 .md / .txt 知识源
├── index/            # FAISS 索引落盘 (git 忽略)
├── fast_rag/
│   ├── config.py     # env → dataclass
│   ├── ingest.py     # 加载 → 切分 → 嵌入 → 写 FAISS
│   ├── rag.py        # 加载索引 → LCEL chain → 答案 + 来源
│   └── cli.py        # argparse 入口
└── tests/
    └── test_smoke.py # ingest + retriever 通路测试
```

## 测试

```bash
pytest -q
# 首次会下载 embedding 模型,确保网络可达 HuggingFace 或已设置 HF_ENDPOINT
```

测试不会真调 Ollama,只验证 ingest + retriever 路径。

## 常见错误

| 现象 | 处理 |
|---|---|
| `索引未找到` | 先跑 `python -m fast_rag.cli ingest` |
| `无法连接 Ollama` | 启动 `ollama serve` |
| `模型 'xxx' 不存在` | 运行 `ollama pull <model>` |
| HuggingFace 下载失败/超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 后重试 |

## 刻意不做(YAGNI)

- 交互式 REPL / 对话历史
- 流式输出
- 增量 ingest(用 `--rebuild` 整建)
- 多用户 / 会话管理
- Reranker、MMR、Hybrid Search

详见 `docs/plans/2026-06-01-fast-rag-demo-design.md`。
