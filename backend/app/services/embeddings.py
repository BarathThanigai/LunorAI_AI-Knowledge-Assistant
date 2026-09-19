"""NVIDIA API embeddings for the LunorAI RAG pipeline."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "nvidia/nemotron-3-embed-1b"


@lru_cache(maxsize=1)
def get_embedding_model() -> str:
    """Return the configured NVIDIA embedding model name."""
    return MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    """Return a shared NVIDIA API client."""
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    )

    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def _embed(
    texts: list[str],
    input_type: str,
) -> np.ndarray:
    """Generate normalized embeddings using NVIDIA's embedding API."""
    if not texts:
        return np.empty(
            (0, 2048),
            dtype=np.float32,
        )

    if input_type not in {"query", "passage"}:
        raise ValueError(
            "input_type must be either 'query' or 'passage'."
        )

    client = get_embedding_client()

    response = client.embeddings.create(
        model=MODEL_NAME,
        input=texts,
        extra_body={
            "input_type": input_type,
            "truncate": "END",
        },
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    array = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if array.ndim == 1:
        array = array.reshape(1, -1)

    # Normalize so FAISS IndexFlatIP behaves as cosine similarity.
    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    array = array / norms

    return array.astype(
        np.float32,
        copy=False,
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return normalized passage embeddings for document chunks."""
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings.")

    if any(
        not isinstance(text, str)
        for text in texts
    ):
        raise TypeError(
            "Every item in texts must be a string."
        )

    return _embed(
        texts,
        input_type="passage",
    )


def embed_query(text: str) -> np.ndarray:
    """Return a normalized embedding for a user query."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "text must be a non-empty string."
        )

    embedding = _embed(
        [text],
        input_type="query",
    )

    return embedding[0]


def get_embedding_dimension() -> int:
    """Return the embedding dimension used by the NVIDIA model."""
    return 2048


__all__ = [
    "get_embedding_model",
    "get_embedding_client",
    "get_embedding_dimension",
    "embed_texts",
    "embed_query",
]