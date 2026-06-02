# Fast-RAG

一个**最小可运行**的 RAG (Retrieval-Augmented Generation) 演示工程。
默认全本地:sentence-transformers 嵌入 + FAISS 向量库 + Ollama 本地 LLM。
也支持一键切换到阿里云**百炼 (DashScope)** 跑远程 LLM 与 Embedding,
详见下方 **切换 Provider** 章节。

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

所有可调参数都在 `.env.example`,复制成 `.env` 后按需修改。
配置按 provider 分段;`FAST_RAG_PROVIDER` 主开关决定使用哪一套。

**主开关**

| 变量 | 默认 | 说明 |
|---|---|---|
| `FAST_RAG_PROVIDER` | `ollama` | `ollama`(本地) 或 `bailian`(阿里云百炼) |

**Ollama 段(本地)**

| 变量 | 默认 | 说明 |
|---|---|---|
| `FAST_RAG_OLLAMA_LLM` | `qwen2.5:7b` | Ollama 上的模型名 |
| `FAST_RAG_OLLAMA_EMBED` | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace embedding 模型名 |
| `FAST_RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `HF_ENDPOINT` | (未设置) | HuggingFace 下载镜像,如 `https://hf-mirror.com` |

**百炼 / DashScope 段(远程)**

| 变量 | 默认 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | (必填) | 阿里云百炼 API key,沿用官方约定 |
| `FAST_RAG_BAILIAN_LLM` | `qwen-plus` | 百炼上的 LLM 模型名 |
| `FAST_RAG_BAILIAN_EMBED` | `text-embedding-v3` | 百炼上的 embedding 模型名 |
| `FAST_RAG_BAILIAN_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容接口地址 |

**公共段**

| 变量 | 默认 | 说明 |
|---|---|---|
| `FAST_RAG_CHUNK_SIZE` | `500` | 切片字符数 |
| `FAST_RAG_CHUNK_OVERLAP` | `50` | 切片重叠字符数 |
| `FAST_RAG_TOP_K` | `4` | 召回片段数量 |

> 若 `huggingface.co` 不通(常见于国内网络),设置
> `export HF_ENDPOINT=https://hf-mirror.com` 后重试。

## 切换 Provider

LLM 与 Embedding **联动切换**(不支持交叉组合)。索引目录会按 provider
自动隔离到 `index/ollama/` 与 `index/bailian/`,切换无需手动清空。

**用本地 Ollama(默认)**

```bash
export FAST_RAG_PROVIDER=ollama
python -m fast_rag.cli ingest        # → index/ollama/
python -m fast_rag.cli ask "..."
```

**用阿里云百炼**

```bash
export FAST_RAG_PROVIDER=bailian
export DASHSCOPE_API_KEY=sk-xxxx
python -m fast_rag.cli ingest        # → index/bailian/
python -m fast_rag.cli ask "..."
```

> 不同 provider 的 embedding 维度不同,FAISS 索引本身不可跨 provider 复用,
> 因此默认目录自动隔离;`--index-dir` 显式覆盖时不再派生。

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
| `provider=bailian 需要设置环境变量 DASHSCOPE_API_KEY` | `export DASHSCOPE_API_KEY=sk-xxx` |
| `百炼 API 调用失败: ...` | 查看原始异常信息(常见为鉴权失败 / 限流 / 网络) |
| HuggingFace 下载失败/超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 后重试 |
| macOS `OMP: Error #15: Initializing libomp.dylib...` | 包初始化已自动设 `KMP_DUPLICATE_LIB_OK=TRUE`;若仍报错请确认是从 `python -m fast_rag.cli` 入口启动 |

## 刻意不做(YAGNI)

- 交互式 REPL / 对话历史
- 流式输出
- 增量 ingest(用 `--rebuild` 整建)
- 多用户 / 会话管理
- Reranker、MMR、Hybrid Search

详见 `docs/plans/2026-06-01-fast-rag-demo-design.md`。
