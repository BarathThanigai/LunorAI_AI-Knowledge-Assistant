"""Document ingestion service for the LunorAI RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.document_processor import process_pdf
from app.services.vector_store import VectorStore


class IngestionService:
    """Process PDF documents and add their chunks to the vector store."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()

    def ingest_pdf(
        self,
        pdf_path: str | Path,
    ) -> dict[str, Any]:
        """Process a PDF and add its chunks to the vector store."""

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        chunks = process_pdf(path)

        if not chunks:
            raise ValueError(
                "No text could be extracted from the PDF."
            )

        self.vector_store.add_chunks(chunks)

        return {
            "source": path.name,
            "chunks_added": len(chunks),
            "total_chunks": self.vector_store.count,
        }


def ingest_pdf(
    pdf_path: str | Path,
) -> dict[str, Any]:
    """Convenience function for ingesting a PDF."""
    service = IngestionService()
    return service.ingest_pdf(pdf_path)


__all__ = [
    "IngestionService",
    "ingest_pdf",
]