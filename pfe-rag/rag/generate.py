"""
Generation: (question, retrieved chunks) -> cited answer, via LiteLLM.

Calls the LLM through the LiteLLM gateway (config.LITELLM_URL), never
Ollama directly — keeps the existing gateway as the single point for
tracing, keys, and model swaps.
"""

import time

import requests

from . import config
from .retrieve import RetrievedChunk

# Kept separate from the logic below so the prompt can be edited without
# touching the code that builds/sends it.
PROMPT = """\
Answer the question using ONLY the numbered context blocks below.
If the answer is not contained in the context, reply exactly "لا أعرف" / "Je ne sais pas".
Always cite the source number(s) you used, e.g. [1].

{context_blocks}

Question: {question}
"""


def build_context_blocks(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered blocks, each labelled with its source file."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.source_file}, page {c.page_number})\n{c.text}")
    return "\n\n".join(blocks)


def answer(question: str, chunks: list[RetrievedChunk]) -> dict:
    """Build the prompt, call the LLM via LiteLLM (config.LLM_MODEL), and
    return {"answer": str, "sources": list[str], "latency_ms": float,
    "tokens": int}."""
    context_blocks = build_context_blocks(chunks)
    prompt = PROMPT.format(context_blocks=context_blocks, question=question)

    start = time.monotonic()
    resp = requests.post(
        f"{config.LITELLM_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.LITELLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    latency_ms = (time.monotonic() - start) * 1000

    if "error" in data:
        raise RuntimeError(f"Generation call failed: {data['error']}")

    text = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)
    # Numbered to match the [i] citations in the prompt, so a citation in
    # the answer can be mapped back to the real file it came from.
    sources = [f"[{i}] {c.source_file} (page {c.page_number})" for i, c in enumerate(chunks, start=1)]

    return {
        "answer": text,
        "sources": sources,
        "latency_ms": latency_ms,
        "tokens": tokens,
    }


if __name__ == "__main__":
    import argparse

    from .retrieve import search

    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question end-to-end.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    retrieved = search(args.question, top_k=args.top_k)
    result = answer(args.question, retrieved)

    print(result["answer"])
    print()
    print("Sources:")
    for s in result["sources"]:
        print(" ", s)
    print(f"\n({result['latency_ms']:.0f} ms, {result['tokens']} tokens)")
