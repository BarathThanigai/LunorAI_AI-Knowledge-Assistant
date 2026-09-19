"""Document ingestion service for the LunorAI RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.document_processor import (
    SUPPORTED_EXTENSIONS,
    process_document,
)
from app.services.vector_store import VectorStore


class IngestionService:
    """Process supported documents and add their chunks to the vector store."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()

    def ingest_document(
        self,
        document_path: str | Path,
        source_name: str | None = None,
    ) -> dict[str, Any]:
        """Process a supported document and add its chunks to the vector store."""

        path = Path(document_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported types: {supported}"
            )

        chunks = process_document(path)

        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        metadata_source = source_name if source_name else path.name

        for chunk in chunks:
            chunk["source"] = metadata_source

        self.vector_store.add_chunks(chunks)

        return {
            "source": metadata_source,
            "file_type": path.suffix.lower(),
            "chunks_added": len(chunks),
            "total_chunks": self.vector_store.count,
        }


def ingest_document(
    document_path: str | Path,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Convenience function for ingesting a supported document."""

    service = IngestionService()

    return service.ingest_document(
        document_path=document_path,
        source_name=source_name,
    )


# Backwards-compatible PDF helper.
def ingest_pdf(
    pdf_path: str | Path,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible helper for PDF ingestion."""

    return ingest_document(
        document_path=pdf_path,
        source_name=source_name,
    )


__all__ = [
    "IngestionService",
    "ingest_document",
    "ingest_pdf",
]
