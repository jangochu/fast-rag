# LLM Provider 切换设计 (Ollama / 阿里云百炼)

- 日期: 2026-06-01
- 状态: 头脑风暴已通过,待进入实施
- 前置: `docs/plans/2026-06-01-fast-rag-demo-design.md`

## 目标

让用户通过一个环境变量在 **本地 Ollama** 与 **阿里云百炼 (DashScope)**
之间切换 RAG 后端,LLM 与 Embedding 同步切换。零交叉组合、零跨索引污染。

## 关键决策

| 维度 | 选择 | 理由 |
|---|---|---|
| 切换范围 | LLM + Embedding 联动 | 用户明确需要,不做交叉组合(YAGNI) |
| 百炼接入 | OpenAI 兼容接口 | 用 `langchain-openai` 一套类同时覆盖 chat 和 embedding,无需 `dashscope` SDK |
| 切换机制 | 仅 env `FAST_RAG_PROVIDER` | 单一开关,简洁 |
| 代码组织 | `fast_rag/providers/` 子包按 provider 拆文件 | 异处隔离彻底,后续加 provider 不动其他文件 |
| 索引隔离 | 默认按 provider 自动分子目录 (`index/ollama/`, `index/bailian/`) | 切换无忧,体验最佳;`--index-dir` 仍可显式覆盖 |
| 依赖增量 | 仅新增 `langchain-openai>=0.2` | 复用现有 LangChain 0.3 生态 |

## 整体架构

```
config.py
   └─ Config.provider ∈ {"ollama", "bailian"}
   └─ 嵌套子配置:OllamaSettings / BailianSettings
        ↓
providers/__init__.py
   ├─ make_llm(cfg)        → providers/<provider>.make_llm(cfg)
   └─ make_embeddings(cfg) → providers/<provider>.make_embeddings(cfg)
        ↓
ingest.py / rag.py
   - HuggingFaceEmbeddings(...) → make_embeddings(cfg)
   - ChatOllama(...)            → make_llm(cfg)
```

工厂返回值遵循 LangChain 标准接口 (`BaseChatModel` / `Embeddings`),
下游 `ingest.py` 和 `rag.py` 不感知 provider。

## 文件结构变化

```
fast_rag/
├── config.py                    # 改:加 provider + 嵌套两套子配置
├── ingest.py                    # 改:embedding 走 make_embeddings(cfg)
├── rag.py                       # 改:LLM 与 embedding 走工厂
├── cli.py                       # 改:索引目录默认按 provider 派生
└── providers/                   # 新增
    ├── __init__.py              # 工厂分派
    ├── ollama.py                # ChatOllama + HuggingFaceEmbeddings
    └── bailian.py               # ChatOpenAI + OpenAIEmbeddings (DashScope base_url)
```

## 配置

### 完整环境变量清单

```bash
# 主开关
FAST_RAG_PROVIDER=ollama          # 或 bailian

# Ollama (本地)
FAST_RAG_OLLAMA_LLM=qwen2.5:7b
FAST_RAG_OLLAMA_EMBED=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
FAST_RAG_OLLAMA_URL=http://localhost:11434

# 百炼 / DashScope (OpenAI 兼容)
DASHSCOPE_API_KEY=sk-xxxx         # 沿用阿里官方约定
FAST_RAG_BAILIAN_LLM=qwen-plus
FAST_RAG_BAILIAN_EMBED=text-embedding-v3
FAST_RAG_BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 公共
FAST_RAG_CHUNK_SIZE=500
FAST_RAG_CHUNK_OVERLAP=50
FAST_RAG_TOP_K=4
# HF_ENDPOINT=https://hf-mirror.com   # 仅 ollama provider 用
```

### Config dataclass 形态

```python
@dataclass(frozen=True)
class OllamaSettings:
    llm_model: str
    embed_model: str
    url: str

@dataclass(frozen=True)
class BailianSettings:
    api_key: str | None       # 允许为 None,在 provider=bailian 时才校验
    llm_model: str
    embed_model: str
    base_url: str

@dataclass(frozen=True)
class Config:
    provider: str             # "ollama" | "bailian"
    chunk_size: int
    chunk_overlap: int
    top_k: int
    ollama: OllamaSettings
    bailian: BailianSettings
```

`Config.from_env()` 始终读出两套子配置(都有合理默认值),不论当前 provider 是谁。

## 索引目录策略

- 默认索引目录由 provider 派生:`index/{provider}/`
- 在 `cli.py` 里:`default_index_dir = Path("index") / cfg.provider`
- 用户显式 `--index-dir` 时直接用,不再派生
- 切换 provider 后,各 provider 走自己的子目录,**互不污染**
- 索引目录不存在时复用已有的"索引未找到"提示,无新增分支

**为什么不用 manifest 校验:** 不同 embedding 模型维度不同,FAISS 索引
本身就不可跨 provider 复用。要么写 manifest 校验报错(多文件、多代码、
出错才发现),要么自动隔离(目录就一层,无声化)。选后者。

## 错误处理

新增 2 类,沿用既有"一句话人话提示"风格:

| 场景 | 检测时机 | 提示 |
|---|---|---|
| `provider=bailian` 但 `DASHSCOPE_API_KEY` 缺失 | `providers/bailian.py` 工厂构造时 | `provider=bailian 需要设置环境变量 DASHSCOPE_API_KEY` |
| 百炼调用失败 (鉴权/限流/网络) | `cli.cmd_ask` 异常分支 | 在原异常前加前缀 `百炼 API 调用失败: ...`,保留原因 |

不做的:
- ❌ 不映射 401 / 429 / 5xx 具体语义,demo 透传即可

既有的"索引未找到 / Ollama 未启动 / 模型未拉"三类提示保持不变。

## 测试

新增 `tests/test_providers.py`,3 个用例,都不真调网络:

```python
def test_ollama_factory_returns_chatollama(monkeypatch):
    monkeypatch.setenv("FAST_RAG_PROVIDER", "ollama")
    cfg = Config.from_env()
    llm = make_llm(cfg)
    assert isinstance(llm, ChatOllama)

def test_bailian_factory_returns_chatopenai_with_dashscope_base(monkeypatch):
    monkeypatch.setenv("FAST_RAG_PROVIDER", "bailian")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    cfg = Config.from_env()
    llm = make_llm(cfg)
    assert isinstance(llm, ChatOpenAI)
    assert "dashscope" in str(llm.openai_api_base)

def test_bailian_without_api_key_raises(monkeypatch):
    monkeypatch.setenv("FAST_RAG_PROVIDER", "bailian")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        make_llm(Config.from_env())
```

既有 `test_smoke.py` 仅加 `monkeypatch.setenv("FAST_RAG_PROVIDER", "ollama")`
确保不被外部 env 干扰。

## 实施改造点清单

1. `pyproject.toml`:加 `langchain-openai>=0.2,<0.4`
2. `.env.example`:用新清单整段替换
3. `fast_rag/config.py`:重构为嵌套 dataclass
4. `fast_rag/providers/`:新建 `__init__.py` / `ollama.py` / `bailian.py`
5. `fast_rag/ingest.py`:embedding 改走 `make_embeddings(cfg)`
6. `fast_rag/rag.py`:LLM 和 embedding 都改走工厂
7. `fast_rag/cli.py`:默认 `index_dir = Path("index") / cfg.provider`;加百炼错误前缀映射
8. `tests/test_providers.py`:新增 3 个工厂测试
9. `tests/test_smoke.py`:固定 provider 为 ollama
10. `README.md`:新增"切换 provider"章节,提及两套 env

## 后续可演进 (不在本次范围)

- 第三个 provider (OpenAI 官方 / Azure / 智谱)
- LLM 与 embedding 解耦切换 (`FAST_RAG_LLM_PROVIDER`, `FAST_RAG_EMBED_PROVIDER`)
- provider 级别的 timeout / retry 配置
