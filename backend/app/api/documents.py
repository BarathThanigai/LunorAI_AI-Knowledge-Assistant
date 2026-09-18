"""Document management API endpoints for LunorAI."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStore


router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


ROOT_DIR = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = (
    ROOT_DIR
    / "data"
    / "documents"
)

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
) -> dict:
    """Upload and ingest a PDF document."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    original_filename = Path(
        file.filename
    ).name

    # Remove any previously uploaded copy
    # of the same filename.
    existing_files = list(
        DOCUMENTS_DIR.glob(
            f"*_{original_filename}"
        )
    )

    for existing_file in existing_files:
        existing_file.unlink()

    document_id = uuid4().hex

    stored_filename = (
        f"{document_id}_{original_filename}"
    )

    file_path = (
        DOCUMENTS_DIR
        / stored_filename
    )

    try:
        # Save the uploaded PDF.
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        vector_store = VectorStore()

        ingestion_service = IngestionService(
            vector_store=vector_store,
        )

        result = ingestion_service.ingest_pdf(
            pdf_path=file_path,
            source_name=original_filename,
        )

        return {
            "id": document_id,
            "filename": original_filename,
            "chunks_added": result[
                "chunks_added"
            ],
            "total_chunks": result[
                "total_chunks"
            ],
            "message": (
                "Document uploaded and indexed "
                "successfully."
            ),
        }

    except Exception as exc:
        # Remove the physical file if
        # ingestion fails.
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "Document ingestion failed: "
                f"{exc}"
            ),
        ) from exc

    finally:
        await file.close()


@router.get("")
def list_documents() -> dict:
    """List uploaded PDF documents."""

    documents = []

    for path in sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    ):
        filename = path.name

        # Stored files use:
        #
        # UUID_original_filename.pdf
        #
        # Extract the original filename
        # for the API response.
        if "_" in filename:
            document_id, original_filename = (
                filename.split("_", 1)
            )
        else:
            document_id = filename
            original_filename = filename

        documents.append(
            {
                "id": document_id,
                "filename": original_filename,
            }
        )

    return {
        "documents": documents,
    }


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
) -> dict:
    """Delete a document and its indexed chunks."""

    matching_files = list(
        DOCUMENTS_DIR.glob(
            f"{document_id}_*.pdf"
        )
    )

    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    file_path = matching_files[0]

    stored_filename = file_path.name

    if "_" in stored_filename:
        _, original_filename = (
            stored_filename.split("_", 1)
        )
    else:
        original_filename = stored_filename

    try:
        vector_store = VectorStore()

        # FAISS metadata uses the
        # original filename.
        vector_store.delete_document(
            original_filename,
        )

        # Delete the physical uploaded file.
        file_path.unlink()

        return {
            "message": (
                "Document deleted successfully."
            ),
            "filename": original_filename,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Document deletion failed: "
                f"{exc}"
            ),
        ) from exc
