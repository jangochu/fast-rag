# Fast-RAG Demo 实施计划

参考设计: `docs/plans/2026-06-01-fast-rag-demo-design.md`

## 阶段 1: 项目骨架 (~10 分钟)

- [ ] 创建 `pyproject.toml`,声明依赖:`langchain>=0.3`, `langchain-community`, `langchain-huggingface`, `langchain-ollama`, `sentence-transformers`, `faiss-cpu`, `python-dotenv`;dev 依赖 `pytest`
- [ ] 创建 `.gitignore` (忽略 `index/`, `.env`, `__pycache__/`, `.venv/`, `*.egg-info/`, HuggingFace 缓存)
- [ ] 创建 `.env.example`,包含设计文档中列出的 6 个 env 变量
- [ ] 创建 `data/sample.md`,放 1-2 段中英文混合内容用于首跑验证
- [ ] 创建 `fast_rag/__init__.py` 和 `tests/__init__.py`

## 阶段 2: 核心模块 (~25 分钟)

- [ ] `fast_rag/config.py`:`@dataclass Config`,`Config.from_env()` 读 env 并填默认值
- [ ] `fast_rag/ingest.py`:`build_index(data_dir, index_dir, rebuild=False)`
  - `DirectoryLoader` + `TextLoader` 加载 `*.md` 和 `*.txt`
  - `RecursiveCharacterTextSplitter` 按 config 切分
  - `HuggingFaceEmbeddings` 嵌入
  - `FAISS.from_documents().save_local()` 落盘
  - `rebuild=True` 时先清空 `index_dir`
- [ ] `fast_rag/rag.py`
  - `load_retriever(index_dir, k)` → FAISS retriever
  - `build_chain(retriever, llm_model, ollama_url)` → LCEL `retriever | prompt | ChatOllama | StrOutputParser`
  - `answer(question)` 返回 `{"answer": str, "sources": list[str]}`
- [ ] `fast_rag/cli.py`:argparse 两个子命令
  - `ingest [--data-dir] [--index-dir] [--rebuild]`
  - `ask <question> [--k] [--no-sources]`
  - 入口:`if __name__ == "__main__": main()`,且支持 `python -m fast_rag.cli`

## 阶段 3: 错误处理 (~10 分钟)

- [ ] `ask` 启动检查 `index/` 存在,否则报"索引未找到,请先运行 'python -m fast_rag.cli ingest'"
- [ ] 捕获 Ollama `ConnectionError`,提示启动 `ollama serve`
- [ ] 捕获 Ollama model-not-found,提示 `ollama pull <model>`

## 阶段 4: 测试 (~10 分钟)

- [ ] `tests/test_smoke.py`:`test_ingest_then_retrieve`
  - 用 `tmp_path` 写两个临时 md
  - 调 `build_index`
  - 调 `load_retriever` 后 `invoke` 一个问题
  - 断言召回结果包含期望关键字
- [ ] 本地跑 `pytest -q` 通过(首次会下载 embedding ~400MB)

## 阶段 5: 文档 (~10 分钟)

- [ ] `README.md`:含 4 步快速启动、env 变量说明、示例 ask/ingest 命令、"刻意不做"清单
- [ ] 在 README 顶部加最小化的"什么是 RAG / 本工程做了什么"两段

## 验收清单

- [ ] `python -m fast_rag.cli ingest` 成功生成 `index/` 目录
- [ ] `python -m fast_rag.cli ask "样例文档讲了什么?"` 返回含答案 + 来源的输出
- [ ] `pytest -q` 全绿
- [ ] 故意停掉 `ollama serve` 后再 `ask`,看到友好提示而非堆栈
- [ ] 删掉 `index/` 后 `ask`,看到"先 ingest"提示

## 评审 (实施后回写)

### 实际改动

- 阶段 1-5 全部按计划落地,未偏离设计
- 阶段 3 "错误处理" 直接合并到 `cli.py` 实现,未独立成文件
- 包管理用 `hatchling` + uv,Python 锁 3.12(系统 3.14 缺 torch wheel)

### 已通过的本地验收

- ✅ 模块全部 import 通过 (config / ingest / rag / cli)
- ✅ `python -m fast_rag.cli --help` 及两个子命令 `--help` 完整
- ✅ 删 `index/` 后 `ask` 报 "索引未找到,请先运行 ..." (exit 2)

### 受网络限制未完成的验收

- ⏸ `pytest -q`:首次需从 HuggingFace 下载 embedding 模型 (~400MB),
  本地 `huggingface.co` 连接被重置,镜像 `hf-mirror.com` 下载也超出 10s
  限时。已在 `.env.example` 和 README 中给出 `HF_ENDPOINT` 镜像建议
- ⏸ `ingest` 真跑:同上,依赖 embedding 模型
- ⏸ `ask` 真跑:除 embedding 外还需本地 Ollama + qwen2.5:7b
- ⏸ Ollama 停掉的友好提示:需先有可用的 ingest

> 用户网络方便时,执行下列三步即可补全验收:
> ```
> export HF_ENDPOINT=https://hf-mirror.com
> .venv/bin/pytest -q
> .venv/bin/python -m fast_rag.cli ingest
> .venv/bin/python -m fast_rag.cli ask "样例文档讲了什么?"
> ```

### 与设计的偏差

- 无功能性偏差
- 新增了 `HF_ENDPOINT` 配置项以应对国内网络场景,设计文档未覆盖此点
  (已在 `.env.example` 注释中体现,设计文档可后续追加备注)
