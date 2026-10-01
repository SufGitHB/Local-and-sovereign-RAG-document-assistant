"""
FastAPI service exposing the RAG pipeline.

POST /ask     {"question": str, "top_k": int = 5} -> generate.answer(...) dict
GET  /health  -> collection name and number of stored chunks

Also speaks the OpenAI-compatible chat protocol (GET /v1/models,
POST /v1/chat/completions) so it can be added as a plain "Connection" in
Open WebUI — exactly like the existing litellm connection — and show up
as a selectable model, with no separate pipelines service needed.

Every request is logged to a JSONL file: timestamp, question, chosen
route and top retrieval score (see router.py), latency, retrieved
sources, model used.
"""

import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from . import config
from .generate import answer as generate_answer
from .retrieve import search
from .router import Route, classify_by_score, is_greeting
from .startup_checks import assert_local_models

app = FastAPI(title="pfe-rag")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _check_data_sovereignty() -> None:
    """Refuse to serve requests if the configured chat/embedding models
    don't resolve to a local provider — see startup_checks.py."""
    assert_local_models()

LOG_PATH = Path(os.environ.get("REQUEST_LOG_PATH", "eval/requests.jsonl"))

# Static reply for GREETING — no LLM call needed for "hi"/"bonjour".
GREETING_REPLY = (
    "Bonjour ! Je suis l'assistant RAG de ce projet : je réponds à partir "
    "des documents indexés. Posez-moi une question sur leur contenu."
)

# Returned as-is for OUT_OF_SCOPE — no LLM call, no general-knowledge
# fallback: this is a document assistant, not a general chatbot. States
# both facts explicitly: (1) not covered by the index, (2) what this
# assistant actually does, so the refusal isn't mistaken for a bug.
OUT_OF_SCOPE_REPLY = (
    "Cette question ne correspond à aucun contenu des documents indexés.\n\n"
    "Cet assistant est un assistant **documentaire** : il répond uniquement "
    "aux questions couvertes par les documents qui lui ont été fournis, "
    "jamais à partir de connaissances générales. Reformulez votre question "
    "en la rapprochant du contenu de ces documents, ou consultez une autre "
    "source pour une question générale."
)


class AskRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K


def _log_request(
    question: str, latency_ms: float, sources: list[str], route: Route, top_score: float | None
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "question": question,
        "route": route.value,
        "top_score": top_score,
        "latency_ms": latency_ms,
        "sources": sources,
        "model": config.LLM_MODEL,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _handle_question(question: str, top_k: int = config.TOP_K) -> tuple[dict, Route, float | None]:
    """Route the question, then dispatch to the matching handler.
    Returns (result, route, top_score) — result has the same shape as
    generate.answer()'s, so both API endpoints below can log and format
    it the same way regardless of the route taken. top_score is None for
    GREETING (retrieval never runs)."""
    if is_greeting(question):
        result = {"answer": GREETING_REPLY, "sources": [], "latency_ms": 0.0, "tokens": 0}
        return result, Route.GREETING, None

    # Retrieval runs first for everything else — a real document question
    # needs it anyway, and its own top score is what decides the route.
    chunks = search(question, top_k=top_k)
    top_score = chunks[0].score if chunks else 0.0
    route = classify_by_score(top_score)

    if route == Route.OUT_OF_SCOPE:
        result = {"answer": OUT_OF_SCOPE_REPLY, "sources": [], "latency_ms": 0.0, "tokens": 0}
    else:  # DOCUMENT_QUESTION
        result = generate_answer(question, chunks)

    return result, route, top_score


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    start = time.monotonic()
    result, route, top_score = _handle_question(req.question, top_k=req.top_k)
    total_latency_ms = (time.monotonic() - start) * 1000

    _log_request(req.question, total_latency_ms, result["sources"], route, top_score)
    return result


@app.get("/health")
def health() -> dict:
    client = QdrantClient(url=config.QDRANT_URL)
    if not client.collection_exists(config.QDRANT_COLLECTION):
        return {"collection": config.QDRANT_COLLECTION, "exists": False, "points_count": 0}

    info = client.get_collection(config.QDRANT_COLLECTION)
    return {
        "collection": config.QDRANT_COLLECTION,
        "exists": True,
        "points_count": info.points_count,
    }


# --- OpenAI-compatible surface, for Open WebUI ------------------------------

MODEL_ID = "pfe-rag"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False


@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "pfe-rag"}]}


def _format_content(result: dict) -> str:
    """Answer plus a rendered sources block, so citations are visible in
    the chat UI without a separate sources panel."""
    content = result["answer"]
    if result["sources"]:
        content += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in result["sources"])
    return content


def _is_internal_task_prompt(content: str) -> bool:
    """Open WebUI sends its own background prompts (chat title, tags,
    suggested follow-ups) to whichever model is currently selected,
    each one marked with a "### Task:" header. These aren't real user
    questions — running the full retrieve+generate pipeline on them just
    burns 20-40s per background task for no benefit."""
    return content.lstrip().startswith("### Task:")


def _empty_completion(completion_id: str, created: int, stream: bool):
    if stream:
        def event_stream():
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    question = req.messages[-1].content  # RAG here is single-turn: no conversation history sent upstream
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if _is_internal_task_prompt(question):
        return _empty_completion(completion_id, created, req.stream)

    result, route, top_score = _handle_question(question)
    _log_request(question, result["latency_ms"], result["sources"], route, top_score)

    content = _format_content(result)

    if req.stream:
        # No real token streaming — the answer is already complete by the
        # time generate.answer() returns. Sent as one SSE chunk so Open
        # WebUI's streaming client (the default) still gets a valid response.
        def event_stream():
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            done = {
                "id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": result["tokens"], "total_tokens": result["tokens"]},
    }
