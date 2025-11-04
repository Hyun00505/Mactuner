"""Retrieval-Augmented Generation (RAG) pipeline utilities.

This module provides helpers to load reference documents, split them into
chunks, build a FAISS vector index, and perform retrieval-augmented question
answering by delegating language-model generation to the existing ``chat``
function.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # Optional dependency for PDF parsing
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional
    fitz = None  # type: ignore

try:  # Optional dependency for embeddings
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional
    SentenceTransformer = None  # type: ignore

try:  # Optional dependency for vector search
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional
    faiss = None  # type: ignore

# Type alias for embedding functions that accept either a string or a sequence
# of strings and return a numpy array of embeddings (shape: [n, d]).
EmbeddingFn = Callable[[Sequence[str]], np.ndarray]


@dataclass
class RAGIndex:
    """Container that groups the FAISS index with its backing chunks."""

    index: "faiss.IndexFlatIP"
    chunks: List[str]


def load_document(path: str) -> str:
    """Load a text or PDF document and return its textual content."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"❌ 문서를 찾을 수 없습니다: {path}")

    suffix = resolved.suffix.lower()
    if suffix == ".pdf":
        if fitz is None:
            raise ImportError(
                "PyMuPDF(fitz)가 설치되어 있지 않습니다. `pip install pymupdf` 후 다시 시도하세요."
            )
        logger.info("Loading PDF document: %s", path)
        text_parts: List[str] = []
        with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts)

    if suffix in {".txt", ".md"}:
        logger.info("Loading text document: %s", path)
        return resolved.read_text(encoding="utf-8")

    raise ValueError("❌ 지원되지 않는 파일 형식입니다 (지원: .pdf, .txt, .md)")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split raw text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be zero or positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    logger.info("Chunked text into %d segments (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)
    return chunks


def create_default_embed_fn(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingFn:
    """Instantiate a sentence-transformer model and return an embedding callable."""

    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers 라이브러리가 필요합니다. `pip install sentence-transformers`로 설치하세요."
        )
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    def _embed(texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = list(texts)
        embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(embeddings, dtype=np.float32)

    return _embed


def _call_embed_fn(embed_fn: EmbeddingFn, texts: Sequence[str]) -> np.ndarray:
    """Call the embedding function while guarding against scalar outputs."""

    embeddings = embed_fn(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)
    return embeddings


def build_rag_index(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    embed_fn: Optional[EmbeddingFn] = None,
) -> RAGIndex:
    """Create chunks and build a FAISS cosine-similarity index."""

    if faiss is None:
        raise ImportError("faiss가 설치되어 있지 않습니다. `pip install faiss-cpu` 후 다시 시도하세요.")

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("문서에서 추출된 청크가 없습니다. chunk_size/overlap을 조정해보세요.")

    embed_fn = embed_fn or create_default_embed_fn()
    embeddings = _call_embed_fn(embed_fn, chunks)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity via dot product on normalized vectors
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors (dim=%d)", embeddings.shape[0], embeddings.shape[1])
    return RAGIndex(index=index, chunks=chunks)


def rag_query(
    query: str,
    rag_index: RAGIndex,
    embed_fn: EmbeddingFn,
    llm_chat_fn: Callable[..., Tuple[str, Any]],
    top_k: int = 3,
    system_prompt: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None,
    prompt_template: Optional[str] = None,
    **chat_kwargs: Any,
) -> Dict[str, Any]:
    """Run a RAG query returning the response, updated history, and retrieved chunks."""

    if faiss is None:
        raise ImportError("faiss가 설치되어 있지 않습니다. `pip install faiss-cpu` 후 다시 시도하세요.")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    history = list(history) if history else []
    normalized_top_k = min(top_k, len(rag_index.chunks))

    query_vec = _call_embed_fn(embed_fn, [query])
    faiss.normalize_L2(query_vec)
    scores, indices = rag_index.index.search(query_vec, normalized_top_k)

    retrieved: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        chunk_text_value = rag_index.chunks[idx]
        retrieved.append({"chunk": chunk_text_value, "score": float(score), "rank": rank})
        context_parts.append(f"[Chunk {rank}]\n{chunk_text_value}")

    context_block = "\n\n".join(context_parts)
    if not prompt_template:
        prompt_template = (
            "You are a helpful assistant. Use ONLY the provided context chunks to answer the question.\n"
            "If the answer is not present in the context, respond that the information is unavailable.\n"
            "\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    prompt = prompt_template.format(context=context_block, question=query)

    response, new_history = llm_chat_fn(
        message=prompt,
        history=history,
        system_prompt=system_prompt,
        **chat_kwargs,
    )

    result: Dict[str, Any] = {
        "response": response,
        "history": new_history,
        "retrieved_chunks": retrieved,
        "prompt": prompt,
    }
    return result


__all__ = [
    "EmbeddingFn",
    "RAGIndex",
    "load_document",
    "chunk_text",
    "create_default_embed_fn",
    "build_rag_index",
    "rag_query",
]
