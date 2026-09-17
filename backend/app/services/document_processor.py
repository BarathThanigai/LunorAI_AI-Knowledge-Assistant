"""PDF extraction and boundary-aware chunking for LunorAI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

    # Remove spaces immediately before/after line breaks.
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def _split_into_units(text: str) -> list[str]:
    """Split text into paragraphs, then sentences when necessary."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    units: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= DEFAULT_CHUNK_SIZE:
            units.append(paragraph)
            continue

        # Split long paragraphs into sentence-like units.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph,
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                units.append(sentence)

    return units


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Create meaningful overlapping chunks from extracted text."""

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    units = _split_into_units(text)

    chunks: list[str] = []
    current_units: list[str] = []
    current_length = 0

    for unit in units:
        unit_length = len(unit)

        # Handle a single unit larger than the target size.
        if unit_length > chunk_size:
            if current_units:
                chunks.append("\n\n".join(current_units))
                current_units = []
                current_length = 0

            start = 0

            while start < len(unit):
                end = min(
                    start + chunk_size,
                    len(unit),
                )

                piece = unit[start:end].strip()

                if piece:
                    chunks.append(piece)

                if end >= len(unit):
                    break

                start = end - chunk_overlap

            continue

        # Add unit to current chunk if it fits.
        separator_length = 2 if current_units else 0

        if (
            current_units
            and current_length + separator_length + unit_length
            > chunk_size
        ):
            chunks.append(
                "\n\n".join(current_units)
            )

            # Build overlap from complete previous units rather than
            # cutting arbitrary characters from the middle of words.
            overlap_units: list[str] = []
            overlap_length = 0

            for previous in reversed(current_units):
                extra = len(previous) + (
                    2 if overlap_units else 0
                )

                if overlap_length + extra > chunk_overlap:
                    break

                overlap_units.insert(
                    0,
                    previous,
                )

                overlap_length += extra

            current_units = overlap_units
            current_length = sum(
                len(item)
                for item in current_units
            ) + max(
                0,
                (len(current_units) - 1) * 2,
            )

        current_units.append(unit)

        current_length = sum(
            len(item)
            for item in current_units
        ) + max(
            0,
            (len(current_units) - 1) * 2,
        )

    if current_units:
        chunks.append(
            "\n\n".join(current_units)
        )

    return chunks


def process_pdf(
    pdf_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Extract text from a PDF and return metadata-rich chunks."""

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

    chunks: list[dict[str, Any]] = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            raw_text = page.get_text("text")

            cleaned_text = _clean_text(
                raw_text
            )

            if not cleaned_text:
                continue

            page_chunks = _chunk_text(
                cleaned_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk_number, chunk_text in enumerate(
                page_chunks,
                start=1,
            ):
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": path.name,
                        "page": page_number,
                        "chunk_id": (
                            f"{path.stem}_"
                            f"{page_number}_"
                            f"{chunk_number}"
                        ),
                    }
                )

    return chunks


def process_document(
    pdf_path: str | Path,
) -> list[dict[str, Any]]:
    """Alias for process_pdf."""
    return process_pdf(pdf_path)


def chunk_pdf(
    pdf_path: str | Path,
) -> list[dict[str, Any]]:
    """Alias for process_pdf."""
    return process_pdf(pdf_path)


__all__ = [
    "process_pdf",
    "process_document",
    "chunk_pdf",
]