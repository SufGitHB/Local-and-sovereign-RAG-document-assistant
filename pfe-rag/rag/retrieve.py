"""
Retrieval: question -> top-k chunks from Qdrant.

No LLM call happens here — this module is deliberately isolated so
retrieval quality can be inspected on its own (see CLI below). Most RAG
failures come from bad retrieval, not bad generation.
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from . import config
from .ingest import embed_batch


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source_file: str
    page_number: int
    chunk_index: int


def embed_question(question: str) -> list[float]:
    """Embed the question with the same model used at ingestion time.
    embed_batch already asserts the vector dimension matches
    config.EMBED_DIM — fails loudly if it doesn't (e.g. EMBED_MODEL was
    changed without re-ingesting)."""
    return embed_batch([question])[0]


def search(question: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
    """Embed the question, query Qdrant, return the top_k chunks with
    their score and metadata."""
    client = QdrantClient(url=config.QDRANT_URL)
    vector = embed_question(question)

    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            text=r.payload["text"],
            score=r.score,
            source_file=r.payload["source_file"],
            page_number=r.payload["page_number"],
            chunk_index=r.payload["chunk_index"],
        )
        for r in results
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect retrieval for one question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    results = search(args.question, top_k=args.top_k)
    for r in results:
        print(f"[{r.score:.3f}] {r.source_file} (page {r.page_number}, chunk {r.chunk_index})")
        print(r.text[:200].replace("\n", " "))
        print()
