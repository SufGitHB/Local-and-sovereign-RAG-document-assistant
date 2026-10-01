"""
Ingestion pipeline: documents -> chunks -> embeddings -> Qdrant.

Walks ./docs, extracts text from .pdf/.docx/.txt/.md, cleans and chunks it,
embeds each chunk via the LiteLLM gateway (bge-m3), and stores it in Qdrant
with metadata (source_file, page_number, chunk_index).
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from pypdf import PdfReader
from docx import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from . import config

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MIN_PAGE_CHARS = 50


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int
    chunk_index: int


def load_documents(docs_dir: Path) -> list[Path]:
    """Walk docs_dir recursively, return paths to supported files
    (.pdf, .docx, .txt, .md)."""
    return sorted(
        p for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def extract_text(path: Path) -> list[tuple[int, str]]:
    """Extract text from one document. Returns a list of (page_number, text)
    — page_number is 1 for non-paginated formats (.txt, .md)."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    if suffix == ".docx":
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return [(1, text)]

    if suffix in (".txt", ".md"):
        return [(1, path.read_text(encoding="utf-8", errors="replace"))]

    raise ValueError(f"Unsupported file type: {path}")


def clean_text(text: str) -> str:
    """Collapse whitespace. Caller drops the page if the cleaned text is
    shorter than 50 characters."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries
    over mid-sentence cuts."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Fallback: if the source had no blank-line paragraphs (e.g. cleaned
    # single-line PDF text), treat sentences as the splittable unit instead.
    if len(paragraphs) <= 1:
        paragraphs = [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", text) if s.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current} {para}".strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            # Overlap: carry the tail of the previous chunk into the next one,
            # starting at a word boundary rather than a raw character offset.
            tail_start = current.find(" ", max(len(current) - chunk_overlap, 0))
            tail = current[tail_start:].strip() if tail_start != -1 else ""
            current = f"{tail} {para}".strip() if tail else para
        else:
            current = para

        # A single paragraph longer than chunk_size: hard-split it, but at
        # the last space before the limit — never mid-word.
        while len(current) > chunk_size:
            cut = current.rfind(" ", 0, chunk_size)
            if cut == -1:  # one unbroken token longer than chunk_size — give up gracefully
                cut = chunk_size
            chunks.append(current[:cut])
            # Snap the overlap carry to a word boundary too — a raw
            # character offset here can land mid-word (e.g. "Chang|e ...").
            overlap_start = max(cut - chunk_overlap, 0)
            boundary = current.find(" ", overlap_start, cut)
            if boundary != -1:
                overlap_start = boundary + 1
            current = current[overlap_start:].lstrip()

    if current:
        chunks.append(current)

    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the LiteLLM gateway (config.EMBED_MODEL).
    Batched into one HTTP request — never one request per chunk."""
    resp = requests.post(
        f"{config.LITELLM_URL}/v1/embeddings",
        headers={
            "Authorization": f"Bearer {config.LITELLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": config.EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Embedding call failed: {data['error']}")

    embeddings = [item["embedding"] for item in data["data"]]
    for emb in embeddings:
        if len(emb) != config.EMBED_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: got {len(emb)}, "
                f"expected config.EMBED_DIM={config.EMBED_DIM}. "
                "Did EMBED_MODEL change without updating EMBED_DIM?"
            )
    return embeddings


def chunk_id(source_file: str, chunk_index: int, text: str) -> str:
    """Deterministic ID (content hash) so re-running ingest is idempotent
    and never creates duplicates. Qdrant point IDs must be uint64 or UUID,
    so this returns a UUID derived from the hash."""
    import uuid

    digest = hashlib.sha256(f"{source_file}:{chunk_index}:{text}".encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(config.QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Store chunks + embeddings + metadata in the Qdrant collection."""
    points = [
        PointStruct(
            id=chunk_id(c.source_file, c.chunk_index, c.text),
            vector=emb,
            payload={
                "text": c.text,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
            },
        )
        for c, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)


def run(docs_dir: Path, reset: bool = False) -> None:
    """Full ingestion run. If reset, drop the Qdrant collection first.
    Prints a summary table at the end: file, pages, chunks created."""
    client = QdrantClient(url=config.QDRANT_URL)

    if reset and client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
    ensure_collection(client)

    files = load_documents(docs_dir)
    if not files:
        print(f"No supported documents found in {docs_dir}")
        return

    summary = []  # (file, pages, chunks)
    all_chunks: list[Chunk] = []

    for path in tqdm(files, desc="Extracting"):
        pages = extract_text(path)
        file_chunk_count = 0
        chunk_index = 0

        for page_number, raw_text in pages:
            cleaned = clean_text(raw_text)
            if len(cleaned) < MIN_PAGE_CHARS:
                continue
            for piece in chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                all_chunks.append(Chunk(
                    text=piece,
                    source_file=path.name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
                file_chunk_count += 1

        summary.append((path.name, len(pages), file_chunk_count))

    # Batch embeddings — one HTTP call per batch, not per chunk.
    BATCH_SIZE = 32
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding + storing"):
        batch = all_chunks[i:i + BATCH_SIZE]
        embeddings = embed_batch([c.text for c in batch])
        upsert_chunks(client, batch, embeddings)

    print("\nfile                                pages   chunks")
    print("-" * 55)
    for name, pages, chunks in summary:
        print(f"{name:<35} {pages:>5}   {chunks:>6}")
    print("-" * 55)
    print(f"{'TOTAL':<35} {sum(s[1] for s in summary):>5}   {sum(s[2] for s in summary):>6}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--reset", action="store_true", help="Drop the collection first.")
    args = parser.parse_args()
    run(args.docs, reset=args.reset)
