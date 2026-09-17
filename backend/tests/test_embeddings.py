from __future__ import annotations

import numpy as np
import pytest

from app.services.embeddings import embed_query, embed_texts, get_embedding_model


def test_embedding_multiple_texts() -> None:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A simple sentence for semantic similarity checks.",
    ]

    embeddings = embed_texts(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.dtype == np.float32


def test_embedding_single_query() -> None:
    query = "How do large language models work?"
    embedding = embed_query(query)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.dtype == np.float32


def test_empty_input_returns_float32_empty_array() -> None:
    embeddings = embed_texts([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0
    assert embeddings.dtype == np.float32


def test_expected_embedding_dimensionality() -> None:
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()

    embeddings = embed_texts(["one sentence", "two sentences"])
    assert embeddings.shape[1] == dim


def test_model_reuse() -> None:
    first_model = get_embedding_model()
    second_model = get_embedding_model()

    assert first_model is second_model


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        embed_query("   ")
