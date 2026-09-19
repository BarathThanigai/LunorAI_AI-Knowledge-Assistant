"""Document extraction and chunking for the LunorAI RAG pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf
from docx import Document


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize spaces/tabs inside lines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _split_long_text(text: str, max_length: int) -> list[str]:
    """Split long text into sentence/whitespace-aware pieces."""
    if len(text) <= max_length:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)

    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_length:
            words = sentence.split()

            current_words: list[str] = []

            for word in words:
                candidate = " ".join(current_words + [word])

                if len(candidate) <= max_length:
                    current_words.append(word)
                else:
                    if current_words:
                        pieces.append(" ".join(current_words))

                    current_words = [word]

            if current_words:
                pieces.append(" ".join(current_words))

            current = ""
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= max_length:
            current = candidate
        else:
            if current:
                pieces.append(current)

            current = sentence

    if current:
        pieces.append(current)

    return pieces


def chunk_text(
    text: str,
    *,
    source: str,
    page: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Convert extracted text into overlapping, boundary-aware chunks."""

    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be greater than or equal to zero "
            "and smaller than chunk_size."
        )

    normalized = _normalize_text(text)

    paragraphs = re.split(r"\n\s*\n", normalized)

    units: list[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        units.extend(_split_long_text(paragraph, chunk_size))

    chunks: list[dict[str, Any]] = []
    current = ""

    for unit in units:
        unit = unit.strip()

        if not unit:
            continue

        candidate = f"{current}\n\n{unit}".strip()

        if current and len(candidate) > chunk_size:
            chunks.append(
                {
                    "text": current,
                    "source": source,
                    "page": page,
                    "chunk_id": "",
                }
            )

            overlap_text = current[-chunk_overlap:].strip()
            current = f"{overlap_text}\n\n{unit}".strip()
        else:
            current = candidate

    if current:
        chunks.append(
            {
                "text": current,
                "source": source,
                "page": page,
                "chunk_id": "",
            }
        )

    for index, chunk in enumerate(chunks, start=1):
        page_part = f"_{page}" if page is not None else ""
        chunk["chunk_id"] = f"{Path(source).stem}{page_part}_{index}"

    return chunks


def process_pdf(path: str | Path) -> list[dict[str, Any]]:
    """Extract text from a PDF page-by-page and create chunks."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    chunks: list[dict[str, Any]] = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            page_chunks = chunk_text(
                text,
                source=path.name,
                page=page_number,
            )

            chunks.extend(page_chunks)

    return chunks


def process_docx(path: str | Path) -> list[dict[str, Any]]:
    """Extract paragraphs from a Word document and create chunks."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    document = Document(path)

    paragraphs: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)

    return chunk_text(
        full_text,
        source=path.name,
        page=None,
    )


def process_text(path: str | Path) -> list[dict[str, Any]]:
    """Read a plain-text file and create chunks."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return chunk_text(
        text,
        source=path.name,
        page=None,
    )


def process_markdown(path: str | Path) -> list[dict[str, Any]]:
    """Read a Markdown file and create chunks."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return chunk_text(
        text,
        source=path.name,
        page=None,
    )


def process_document(path: str | Path) -> list[dict[str, Any]]:
    """Process a supported document based on its file extension."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {supported}"
        )

    if extension == ".pdf":
        return process_pdf(path)

    if extension == ".docx":
        return process_docx(path)

    if extension == ".txt":
        return process_text(path)

    if extension == ".md":
        return process_markdown(path)

    # This should never be reached because of the extension check.
    raise ValueError(f"Unsupported file type: {extension}")


# Backwards-compatible alias used by existing code.
def chunk_pdf(path: str | Path) -> list[dict[str, Any]]:
    """Backward-compatible alias for PDF processing."""
    return process_pdf(path)


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "SUPPORTED_EXTENSIONS",
    "chunk_text",
    "chunk_pdf",
    "process_pdf",
    "process_docx",
    "process_text",
    "process_markdown",
    "process_document",
]
