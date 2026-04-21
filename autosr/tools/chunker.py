"""
Text chunking and BM25-based semantic filtering for extraction.
Splits parsed PDF text into overlapping chunks and retrieves
the most relevant chunks for each extraction context.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    source: str = "body",
) -> List[dict]:
    """
    Split text into overlapping chunks with sentence-boundary awareness.

    Returns list of dicts: {"text", "source", "start_char", "end_char"}
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_str = text[start:end]

        # Try to break at a sentence boundary within the last 20% of the chunk
        if end < text_len:
            boundary_start = int(chunk_size * 0.8)
            boundary_zone = chunk_str[boundary_start:]
            match = re.search(r"[.!?]\s", boundary_zone)
            if match:
                end = start + boundary_start + match.end()
                chunk_str = text[start:end]

        stripped = chunk_str.strip()
        if stripped:
            chunks.append({
                "text": stripped,
                "source": source,
                "start_char": start,
                "end_char": end,
            })

        # Ensure forward progress: advance by at least 1 character
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
        if start >= text_len:
            break

    return chunks


def chunk_document(
    markdown_text: str,
    tables: List[str],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> Tuple[List[dict], List[dict]]:
    """
    Chunk a full parsed document into body chunks and table chunks.

    Returns:
        (body_chunks, table_chunks)
    Body chunks are from the main text; table chunks are standalone (one per table).
    """
    body_chunks = chunk_text(markdown_text, chunk_size, overlap, source="body")

    table_chunks = []
    for table_md in tables:
        stripped = table_md.strip()
        if stripped:
            table_chunks.append({
                "text": stripped,
                "source": "table",
                "start_char": 0,
                "end_char": len(stripped),
            })

    return body_chunks, table_chunks


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    body_chunks: List[dict],
    query: str,
    top_k: int = 15,
) -> List[dict]:
    """
    Use BM25 to retrieve the top-k most relevant body-text chunks for a query.
    """
    from rank_bm25 import BM25Okapi

    if not body_chunks:
        return []

    corpus = [c["text"].lower().split() for c in body_chunks]
    query_tokens = query.lower().split()

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    scored_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [body_chunks[i] for i in scored_indices]


def build_context_chunks(
    body_chunks: List[dict],
    table_chunks: List[dict],
    field_names_and_descs: List[Tuple[str, str]],
    top_k: int = 15,
) -> List[dict]:
    """
    Build the final context chunk list for an extraction call:
    1. Combine all field names+descriptions into a single BM25 query
    2. Retrieve top-k body chunks by relevance
    3. Always include ALL table chunks
    4. Deduplicate by text content

    Returns: deduplicated list of chunk dicts.
    """
    # Build combined query
    query_parts = []
    for name, desc in field_names_and_descs:
        query_parts.append(name)
        if desc:
            query_parts.append(desc)
    query = " ".join(query_parts)

    # BM25 on body chunks
    relevant_body = retrieve_relevant_chunks(body_chunks, query, top_k=top_k)

    # Merge with all table chunks, deduplicate
    seen = set()
    merged = []
    for chunk in relevant_body + table_chunks:
        text_hash = hash(chunk["text"])
        if text_hash not in seen:
            seen.add(text_hash)
            merged.append(chunk)

    return merged


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_chunks_with_citations(chunks: List[dict]) -> str:
    """
    Format chunks as numbered XML citation blocks for LLM consumption.

    Example output:
        <source id="1"><content>Some text from the paper...</content></source>
        <source id="2"><content>A table from the paper...</content></source>
    """
    if not chunks:
        return "(No relevant content found)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f'<source id="{i}"><content>{chunk["text"]}</content></source>'
        )
    return "\n\n".join(parts)
