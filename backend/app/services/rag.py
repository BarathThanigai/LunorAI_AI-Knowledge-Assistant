"""RAG orchestration service for LunorAI."""

from __future__ import annotations

from typing import Any

from app.services.llm import generate_answer
from app.services.retriever import Retriever


FALLBACK_ANSWER = (
    "I couldn't find that information in the knowledge base."
)

# Retrieve a few extra candidates, then only send the strongest
# relevant chunks to the LLM.
RETRIEVAL_TOP_K = 8
MAX_CONTEXT_CHUNKS = 5
MIN_SIMILARITY = 0.20


class RAGService:
    """Retrieve relevant knowledge and generate grounded answers."""

    def __init__(
        self,
        retriever: Retriever | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()

    def _build_context(
        self,
        results: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build grounded LLM context from sufficiently relevant chunks."""

        relevant_results = [
            result
            for result in results
            if float(result["score"]) >= MIN_SIMILARITY
        ]

        relevant_results = relevant_results[:MAX_CONTEXT_CHUNKS]

        context_parts: list[str] = []

        for index, result in enumerate(relevant_results, start=1):
            page = result.get("page")

            page_text = (
                str(page)
                if page is not None
                else "N/A"
            )

            context_parts.append(
                f"""Source {index}
Document: {result["source"]}
Page: {page_text}
Similarity: {float(result["score"]):.3f}

{result["text"]}"""
            )

        return (
            "\n\n---\n\n".join(context_parts),
            relevant_results,
        )

    def answer(
        self,
        question: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> dict[str, Any]:
        """Answer a question using only retrieved knowledge."""

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

        context, relevant_results = self._build_context(results)

        sources = [
            {
                "document": result["source"],
                "page": result.get("page"),
                "score": round(float(result["score"]), 3),
            }
            for result in relevant_results
        ]

        # Nothing sufficiently relevant was retrieved.
        if not relevant_results:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
            }

        answer = generate_answer(
            question=question,
            context=context,
        )

        return {
            "answer": answer,
            "sources": sources,
        }


def answer_question(
    question: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """Convenience function for answering a question."""

    service = RAGService()

    return service.answer(
        question=question,
        top_k=top_k,
    )


__all__ = [
    "RAGService",
    "answer_question",
    "FALLBACK_ANSWER",
]
