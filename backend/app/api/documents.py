from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.document_processor import SUPPORTED_EXTENSIONS
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/documents", tags=["Documents"])

ROOT_DIR = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload and index a supported document."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    original_filename = Path(file.filename).name
    extension = Path(original_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. "
            f"Supported types: {supported}",
        )

    # Remove an existing uploaded copy with the same original filename.
    existing_files = list(
        DOCUMENTS_DIR.glob(f"*_{original_filename}")
    )

    for existing_file in existing_files:
        existing_file.unlink()

    document_id = uuid4().hex
    stored_filename = f"{document_id}_{original_filename}"
    file_path = DOCUMENTS_DIR / stored_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        vector_store = VectorStore()
        ingestion_service = IngestionService(
            vector_store=vector_store
        )

        result = ingestion_service.ingest_document(
            document_path=file_path,
            source_name=original_filename,
        )

        return {
            "id": document_id,
            "filename": original_filename,
            "file_type": extension,
            "chunks_added": result["chunks_added"],
            "total_chunks": result["total_chunks"],
            "message": "Document uploaded and indexed successfully.",
        }

    except Exception as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc

    finally:
        await file.close()


@router.get("")
def list_documents() -> dict:
    """List all uploaded supported documents."""

    documents = []

    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if not path.is_file():
            continue

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        stored_filename = path.name

        if "_" in stored_filename:
            document_id, original_filename = stored_filename.split(
                "_", 1
            )
        else:
            document_id = stored_filename
            original_filename = stored_filename

        documents.append(
            {
                "id": document_id,
                "filename": original_filename,
                "file_type": extension,
            }
        )

    return {"documents": documents}


@router.delete("/{document_id}")
def delete_document(document_id: str) -> dict:
    """Delete an uploaded document and its indexed chunks."""

    matching_files = []

    for extension in SUPPORTED_EXTENSIONS:
        matching_files.extend(
            DOCUMENTS_DIR.glob(
                f"{document_id}_*{extension}"
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
        _, original_filename = stored_filename.split("_", 1)
    else:
        original_filename = stored_filename

    try:
        vector_store = VectorStore()
        vector_store.delete_document(original_filename)

        file_path.unlink()

        return {
            "message": "Document deleted successfully.",
            "filename": original_filename,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {exc}",
        ) from exc