from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.services.document_processor import _chunk_text, extract_pdf_chunks


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "This is the first page of the sample PDF. " * 20)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "This is the second page with more content for chunk overlap tests. " * 18)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extract_pdf_chunks_returns_metadata_and_valid_chunks(sample_pdf_path: Path) -> None:
    chunks = extract_pdf_chunks(sample_pdf_path)

    assert isinstance(chunks, list)
    assert chunks
    assert all(set(chunk.keys()) == {"text", "source", "page", "chunk_id"} for chunk in chunks)
    assert all(chunk["text"].strip() for chunk in chunks)
    assert all(chunk["source"] == sample_pdf_path.name for chunk in chunks)
    assert all(isinstance(chunk["page"], int) and chunk["page"] >= 1 for chunk in chunks)
    assert all(chunk["chunk_id"] for chunk in chunks)
    assert any(chunk["page"] == 2 for chunk in chunks)


def test_extract_pdf_chunks_handles_missing_or_invalid_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        extract_pdf_chunks(missing_path)

    invalid_pdf = tmp_path / "not_a_pdf.txt"
    invalid_pdf.write_text("this is not a pdf", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_pdf_chunks(invalid_pdf)


def test_extract_pdf_chunks_skips_empty_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty_page.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    chunks = extract_pdf_chunks(pdf_path)
    assert chunks == []


def test_chunk_text_short_text() -> None:
    text = "Short text only."
    assert _chunk_text(text) == [text]


def test_chunk_text_multiple_paragraphs() -> None:
    text = (
        "First paragraph introduces the topic and gives context.\n\n"
        "Second paragraph adds more detail for retrieval quality.\n\n"
        "Third paragraph closes the example with a final thought."
    )

    chunks = _chunk_text(text, chunk_size=90, chunk_overlap=20)

    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    assert not any(chunk == "" for chunk in chunks)
    assert any("First paragraph" in chunk for chunk in chunks)


def test_chunk_text_respects_sentence_boundaries() -> None:
    text = "First sentence is intentionally short. Second sentence is longer and should stay intact. Third sentence ends the paragraph."

    chunks = _chunk_text(text, chunk_size=60, chunk_overlap=12)

    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    assert all("." in chunk or len(chunk) < 60 for chunk in chunks)


def test_chunk_text_longer_than_chunk_size() -> None:
    text = ("alpha beta gamma delta " * 40).strip()

    chunks = _chunk_text(text, chunk_size=80, chunk_overlap=15)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_chunk_text_handles_no_convenient_sentence_boundary() -> None:
    text = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau " * 25).strip()

    chunks = _chunk_text(text, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert not any(chunk == "" for chunk in chunks)


def test_chunk_text_overlap_behavior() -> None:
    text = ("word " * 200).strip()

    chunks = _chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert any(chunks[0][-20:] in chunks[1] for _ in [0])
