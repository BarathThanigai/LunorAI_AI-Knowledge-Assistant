"""Sentence-transformer embeddings for the LunorAI RAG pipeline."""

from __future__ import annotations

from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model: SentenceTransformer | None = None
_model_lock = Lock()


def get_embedding_model() -> SentenceTransformer:
    """Return a shared SentenceTransformer model instance."""
    global _embedding_model

    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return normalized sentence embeddings for a list of texts as float32."""
    model = get_embedding_model()

    if not texts:
        dim = int(model.get_sentence_embedding_dimension())
        return np.empty((0, dim), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array.astype(np.float32, copy=False)


def embed_query(text: str) -> np.ndarray:
    """Return a single normalized query embedding as a 1D float32 vector."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    model = get_embedding_model()
    embedding = model.encode(
        [text],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    array = np.asarray(embedding, dtype=np.float32)
    return array.reshape(-1).astype(np.float32, copy=False)


__all__ = ["get_embedding_model", "embed_texts", "embed_query"]
