# Lessons

## OpenAI 兼容接口 ≠ OpenAI 接口

DashScope 等"OpenAI 兼容"接口只覆盖**请求/响应 schema 的核心子集**,
LangChain 客户端的若干默认行为会失败:

- `OpenAIEmbeddings` 默认 `check_embedding_ctx_length=True`,
  会用 tiktoken 把文本预编码成 **整数 token 数组**再 POST。
  OpenAI 自家接受,DashScope 抛 400 `contents is neither str nor list of str`。
  → 接非官方 OpenAI 兼容接口时必须 `check_embedding_ctx_length=False`。
- 各家 embedding 接口都有 batch size 上限(DashScope v3: 25)。
  `OpenAIEmbeddings` 默认 `chunk_size=1000`,需要按 provider 收紧。

**通用原则**: 任何 "OpenAI 兼容" provider 接入后必须真跑一次,
本地 mock/类型检查无法暴露这类协议级差异。

## macOS faiss + torch = libomp 二次初始化 abort

`faiss-cpu` 和 `torch`(经 `sentence-transformers`) 在 macOS 上各自静态
链接 `libomp.dylib`。同进程 import 两者时 OpenMP runtime 的 sanity
check 抛 `OMP: Error #15` 然后 abort。

修复必须在**任何子模块 import torch/faiss 之前**:
```python
# package __init__.py 最早处
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```
放在 `cli.py` 或子模块都太晚 —— Python import 顺序会先解析依赖。

## 子包入口的"急切 import"会拖累所有路径

```python
# providers/__init__.py
from . import ollama, bailian   # ❌ 急切
```
即便用户选 bailian,也会触发 ollama.py 顶部的 `from langchain_huggingface
import HuggingFaceEmbeddings` → 拉入 torch + sentence-transformers
(数百 MB,启动慢,顺带触发上一条的 libomp 问题)。

→ 多 provider 工厂模式默认走 `importlib.import_module(name)` 懒加载。
