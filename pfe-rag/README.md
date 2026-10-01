# pfe-rag — the RAG layer

A small FastAPI service that answers questions from your documents, with
sources, using only local models. It speaks the OpenAI protocol, so Open WebUI
lists it as a model (`pfe-rag`). It runs next to [`../ai-stack`](../ai-stack/README.md)
on the shared Docker network `ai-net`.

## How it works

**1. Indexing** — `rag/ingest.py`, run once and again when documents change

```
docs/*.pdf|docx|txt|md ─▶ text per page ─▶ clean ─▶ chunks ─▶ bge-m3 vectors ─▶ Qdrant
```

- Text is extracted page by page (the page number is kept for citations);
  pages with fewer than 50 characters are skipped.
- Chunks of at most 800 characters, 100 characters of overlap, cut between
  paragraphs or sentences — never inside a word.
- Embeddings are computed in batches of 32 through LiteLLM (1024 dimensions,
  checked on every call).
- Each chunk is stored with its text, file and page. Its id is a SHA-256 hash
  of its content, so re-indexing replaces instead of duplicating.

**2. Answering** — `rag/api.py`, for every question

```
question ─▶ greeting? ─▶ bge-m3 ─▶ Qdrant top-5 ─▶ best score ≥ MIN_SCORE ?
                                                   ├─ no  ─▶ fixed refusal (no LLM call)
                                                   └─ yes ─▶ prompt ─▶ LLM ─▶ answer + sources
```

| Route | When | LLM called? |
|---|---|---|
| `GREETING` | the whole message is a known greeting (hello, bonjour, salam…) | no |
| `OUT_OF_SCOPE` | best similarity score `< MIN_SCORE` | no |
| `DOCUMENT_QUESTION` | best similarity score `≥ MIN_SCORE` | yes |

The prompt (`rag/generate.py`) tells the model to answer **only** from the
numbered passages, to reply "Je ne sais pas" when the answer is not there, and
to cite the passage numbers. The answer is followed by the list of sources
(file and page), numbered like the passages.

Internal requests from Open WebUI (title or tag generation, starting with
`### Task:`) are answered empty, so they never trigger the RAG pipeline.

**3. Data sovereignty**

- `rag/startup_checks.py`: at startup, the API asks LiteLLM (`/model/info`)
  which provider really serves `LLM_MODEL` and `EMBED_MODEL`, and refuses to
  start if it is not `ollama`.
- `tests/test_data_sovereignty.py`: blocks every real socket and checks that
  generation only calls the LiteLLM gateway.

## Project layout

```
pfe-rag/
├── rag/
│   ├── config.py           all settings, overridable by environment variables
│   ├── ingest.py           indexing pipeline
│   ├── retrieve.py         question embedding + Qdrant search
│   ├── router.py           greeting / document question / out of scope
│   ├── generate.py         prompt + LLM call through LiteLLM
│   ├── startup_checks.py   sovereignty guard
│   └── api.py              FastAPI app (OpenAI-compatible) + request log
├── eval/
│   ├── golden_dataset.json   reference questions (question, expected answer, pages)
│   ├── calibrate_threshold.py
│   └── evaluate_retrieval.py
├── tests/test_data_sovereignty.py
├── docs/                   your documents (not versioned)
├── Dockerfile
└── docker-compose.yml      rag-api + qdrant, on the external network ai-net
```

## Configuration

Every setting is an environment variable (defaults in `rag/config.py`):

| Variable | Default | Role |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 100 | chunk size and overlap (characters) |
| `TOP_K` | 5 | number of passages retrieved |
| `MIN_SCORE` | 0.52 | routing threshold — **recalibrate it for your documents** |
| `EMBED_MODEL` / `EMBED_DIM` | `bge-m3` / 1024 | embedding model (LiteLLM name) and vector size |
| `LLM_MODEL` | `rag-chat-local` | generation model (LiteLLM name) |
| `LITELLM_URL` / `LITELLM_API_KEY` | `http://litellm:4000` / — | gateway (key from `.env`) |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://qdrant:6333` / `pfe_docs` | vector database |

## Usage

```bash
docker compose up -d --build                                   # rag-api :8001 + qdrant :6333
docker exec rag-api python -m rag.ingest --docs docs --reset   # (re)index docs/

docker exec rag-api python -m rag.retrieve "What is dropout?"  # passages + scores, no LLM
docker exec rag-api python -m rag.generate "What is dropout?"  # full answer in the terminal
curl -s localhost:8001/ask -H 'Content-Type: application/json' -d '{"question": "What is dropout?"}'
curl -s localhost:8001/health                                  # collection and number of points
```

In Open WebUI, add the connection `http://rag-api:8000/v1` (Admin → Settings →
Connections) and the model `pfe-rag` appears. Every request is logged in
`eval/requests.jsonl` (route, score, latency, sources, model).

## Evaluation

`eval/golden_dataset.json` holds the reference questions, each with the
expected answer and the pages where it is found. The provided set targets the
Deep Learning course used for the demo (not included) — **write your own for
your documents**, in the same format.

```bash
docker exec rag-api python -m eval.calibrate_threshold   # choose MIN_SCORE
docker exec rag-api python -m eval.evaluate_retrieval    # hit@1, hit@k, MRR
docker run --rm -v "$PWD:/w" -w /w python:3.11-slim \
  sh -c "pip install -q -r rag/requirements-dev.txt && pytest"   # sovereignty test
```

- `calibrate_threshold.py` scores the reference questions and 10 off-topic
  ones, then prints, for thresholds from 0.20 to 0.60, the share of reference
  questions accepted and of off-topic questions wrongly accepted.
- `evaluate_retrieval.py` checks whether the retrieved chunks come from the
  expected pages: **hit@1** (first chunk), **hit@k** (any of the k chunks) and
  **MRR** (mean of 1/rank of the first correct chunk).

The threshold depends on the documents: on the demo course, reference
questions scored ≥ 0.550 and off-topic ones ≤ 0.484, hence `MIN_SCORE = 0.52`.
Re-run the calibration whenever the corpus changes.
