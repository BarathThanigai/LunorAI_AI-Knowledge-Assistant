"""Retrieval service for the LunorAI RAG pipeline."""

from __future__ import annotations

from typing import Any

from app.services.embeddings import embed_query
from app.services.vector_store import VectorStore


class Retriever:
    """Retrieve the most relevant document chunks for a user question."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k chunks relevant to the question."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = embed_query(question)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results


def retrieve_chunks(
    question: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Convenience function for retrieving relevant document chunks."""
    retriever = Retriever()
    return retriever.retrieve(
        question=question,
        top_k=top_k,
    )


__all__ = [
    "Retriever",
    "retrieve_chunks",
]