"""Chat API endpoints for LunorAI."""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.services.rag import answer_question


router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
)


class ChatRequest(BaseModel):
    """Request body for a knowledge-base question."""

    question: str = Field(
        min_length=1,
        max_length=2000,
    )


@router.post("")
def chat(
    request: ChatRequest,
) -> dict:
    """Answer a question using the RAG pipeline."""

    try:
        result = answer_question(
            question=request.question,
            top_k=5,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate answer: {exc}",
        ) from exc