"""
All tunable settings for the RAG pipeline, in one place.

Every value is overridable by an environment variable, so experiments
can be run by changing env vars only — no code edits, no risk of code
drift between runs.
"""

import os


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# --- Chunking -----------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 800)        # characters per chunk
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 100)  # characters shared between consecutive chunks

# --- Retrieval ------------------------------------------------------------
TOP_K = _int("TOP_K", 5)

# Minimum top-chunk cosine similarity for a question to be considered
# covered by the indexed documents (see router.py). Below this, the
# question is routed OUT_OF_SCOPE instead of answered. Depends on the
# corpus: re-run eval/calibrate_threshold.py whenever the documents change.
# Deep-learning course corpus (20 golden + 10 off-topic): golden >= 0.550,
# off-topic <= 0.484, "how are you ?" = 0.442 — 0.52 sits between the two.
# (The previous CV corpus used 0.44, which lets 4/10 off-topic through here.)
MIN_SCORE = _float("MIN_SCORE", 0.52)

# --- Models (served through the LiteLLM gateway, not called directly) -----
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
LLM_MODEL = os.environ.get("LLM_MODEL", "rag-chat-local")

# --- Gateway ----------------------------------------------------------------
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")  # LITELLM_MASTER_KEY, passed in at runtime

# --- Vector DB --------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "pfe_docs")

# Embedding vector size for bge-m3 (checked against Qdrant collection config
# and against live embedding calls at ingestion/retrieval time — fail loudly
# if they ever disagree, e.g. after switching EMBED_MODEL).
EMBED_DIM = _int("EMBED_DIM", 1024)
