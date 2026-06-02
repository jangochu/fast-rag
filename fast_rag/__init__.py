"""Fast-RAG: a minimal local-or-remote RAG demo."""

# macOS: faiss-cpu and torch (via sentence-transformers) each ship libomp,
# which trips OpenMP's "multiple runtime initialized" guard at import time.
# Setting this MUST happen before any submodule imports torch or faiss.
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
