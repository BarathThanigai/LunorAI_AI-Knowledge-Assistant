"""RAG orchestration service for LunorAI."""

from __future__ import annotations

from typing import Any

from app.services.llm import generate_answer
from app.services.retriever import Retriever


class RAGService:
    """Combine retrieval and LLM generation into a RAG pipeline."""

    def __init__(
        self,
        retriever: Retriever | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and generate a grounded answer."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question must be a non-empty string."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        results = self.retriever.retrieve(
            question=question,
            top_k=top_k,
        )

        if not results:
            return {
                "answer": (
                    "I couldn't find that information "
                    "in the knowledge base."
                ),
                "sources": [],
            }

        context_parts: list[str] = []

        for index, result in enumerate(results, start=1):
            context_parts.append(
                f"""Source {index}
Document: {result["source"]}
Page: {result["page"]}
Similarity: {result["score"]:.3f}

{result["text"]}"""
            )

        context = "\n\n---\n\n".join(context_parts)

        answer = generate_answer(
            question=question,
            context=context,
        )

        sources = [
            {
                "document": result["source"],
                "page": result["page"],
                "score": round(
                    float(result["score"]),
                    3,
                ),
            }
            for result in results
        ]

        return {
            "answer": answer,
            "sources": sources,
        }


def answer_question(
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Convenience function for answering a question with RAG."""

    service = RAGService()

    return service.answer(
        question=question,
        top_k=top_k,
    )


__all__ = [
    "RAGService",
    "answer_question",
]