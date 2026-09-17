from __future__ import annotations

from pathlib import Path

from app.services.embeddings import embed_query
from app.services.vector_store import VectorStore


def test_empty_store(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vector_store")
    assert store.count == 0
    assert store.get_all_metadata() == []
    assert store.search(embed_query("hello world"), top_k=5) == []


def test_add_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vector_store")
    chunks = [
        {"text": "AI helps humans analyze large datasets.", "source": "doc1.pdf", "page": 1, "chunk_id": "doc1_1"},
        {"text": "Machine learning models can classify text accurately.", "source": "doc1.pdf", "page": 2, "chunk_id": "doc1_2"},
    ]

    store.add_chunks(chunks)

    assert store.count == 2
    assert len(store.get_all_metadata()) == 2
    assert store.get_all_metadata()[0]["source"] == "doc1.pdf"


def test_persistence_and_reload(tmp_path: Path) -> None:
    store_dir = tmp_path / "vector_store"
    store = VectorStore(store_dir)
    chunks = [{"text": "Persistence keeps data safe across restarts.", "source": "persist.pdf", "page": 1, "chunk_id": "persist_1"}]

    store.add_chunks(chunks)
    reloaded = VectorStore(store_dir)

    assert reloaded.count == 1
    assert reloaded.get_all_metadata()[0]["text"] == chunks[0]["text"]


def test_similarity_search(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vector_store")
    chunks = [
        {"text": "Cats are domestic animals that often live indoors.", "source": "cat.pdf", "page": 1, "chunk_id": "cat_1"},
        {"text": "Dogs are loyal companions and frequently work with humans.", "source": "dog.pdf", "page": 1, "chunk_id": "dog_1"},
    ]
    store.add_chunks(chunks)

    query_embedding = embed_query("What animals are good companions for people?")
    results = store.search(query_embedding, top_k=2)

    assert results
    assert all("text" in item and "score" in item for item in results)
    assert results[0]["source"] == "dog.pdf"


def test_duplicate_document_handling(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vector_store")
    first = [{"text": "Original content for duplicate test.", "source": "dup.pdf", "page": 1, "chunk_id": "dup_1"}]
    second = [{"text": "Replacement content for duplicate test.", "source": "dup.pdf", "page": 3, "chunk_id": "dup_2"}]

    store.add_chunks(first)
    store.add_chunks(second)

    assert store.count == 1
    assert store.get_all_metadata()[0]["chunk_id"] == "dup_2"
    assert store.get_all_metadata()[0]["text"] == "Replacement content for duplicate test."


def test_delete_document(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vector_store")
    chunks = [
        {"text": "Alpha text belongs to source one.", "source": "one.pdf", "page": 1, "chunk_id": "one_1"},
        {"text": "Beta text belongs to source two.", "source": "two.pdf", "page": 1, "chunk_id": "two_1"},
    ]
    store.add_chunks(chunks)

    store.delete_document("one.pdf")

    assert store.count == 1
    assert store.get_all_metadata()[0]["source"] == "two.pdf"


def test_searching_empty_store(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vector_store")
    assert store.search(embed_query("anything"), top_k=3) == []
