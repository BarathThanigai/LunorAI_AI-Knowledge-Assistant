"""Persistent FAISS vector store for LunorAI chunk metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.services.embeddings import (
    embed_texts,
    get_embedding_dimension,
)


class VectorStore:
    """Store normalized embeddings alongside chunk metadata on disk.

    If a document source already exists, its previous chunks are replaced.
    The FAISS index is rebuilt whenever documents are added or deleted so that
    vector positions always remain aligned with metadata positions.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        if storage_dir is None:
            root_dir = Path(__file__).resolve().parents[2]
            storage_dir = root_dir / "data" / "vector_store"

        self.storage_dir = Path(storage_dir)
        self.index_path = self.storage_dir / "index.faiss"
        self.metadata_path = self.storage_dir / "metadata.json"

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._dimension = get_embedding_dimension()

        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []

        self.load()

    @property
    def count(self) -> int:
        """Return the number of stored chunks."""
        return len(self._metadata)

    def _create_empty_index(self) -> faiss.IndexFlatIP:
        """Create an empty FAISS index with the configured embedding dimension."""
        return faiss.IndexFlatIP(self._dimension)

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from the current metadata."""
        texts = [entry["text"] for entry in self._metadata]

        if not texts:
            self._index = self._create_empty_index()
            return

        vectors = embed_texts(texts)

        if vectors.size == 0:
            self._index = self._create_empty_index()
            return

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                "Embedding dimension does not match the configured model."
            )

        index = self._create_empty_index()

        index.add(
            vectors.astype(
                np.float32,
                copy=False,
            )
        )

        self._index = index

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks and persist them, replacing existing documents by source."""
        if not chunks:
            return

        normalized: list[dict[str, Any]] = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise TypeError("Each chunk must be a dictionary.")

            required = {
                "text",
                "source",
                "page",
                "chunk_id",
            }

            missing = required - set(chunk)

            if missing:
                raise ValueError(
                    f"Chunk is missing required keys: {sorted(missing)}"
                )

            text = str(chunk["text"]).strip()

            if not text:
                continue

            normalized.append(
                {
                    "text": text,
                    "source": str(chunk["source"]),
                    "page": chunk["page"],
                    "chunk_id": str(chunk["chunk_id"]),
                }
            )

        if not normalized:
            return

        sources = {
            chunk["source"]
            for chunk in normalized
        }

        # Remove existing chunks belonging to the same documents.
        remaining = [
            entry
            for entry in self._metadata
            if entry.get("source") not in sources
        ]

        # Keep metadata in exactly the same order that will be represented
        # by the rebuilt FAISS index.
        self._metadata = remaining + normalized

        # Rebuild FAISS so vector positions and metadata positions stay aligned.
        self._rebuild_index()

        self.save()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top relevant chunks with similarity scores."""
        if self._index is None or self._index.ntotal == 0:
            return []

        if top_k <= 0:
            return []

        vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        if vector.shape[1] != self._index.d:
            raise ValueError(
                "Query embedding dimension does not match the FAISS index."
            )

        k = min(
            top_k,
            self._index.ntotal,
        )

        distances, indices = self._index.search(
            vector,
            k,
        )

        results: list[dict[str, Any]] = []

        for distance, index in zip(
            distances[0],
            indices[0],
        ):
            if index < 0 or index >= len(self._metadata):
                continue

            metadata = dict(
                self._metadata[index]
            )

            results.append(
                {
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", ""),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "score": float(distance),
                }
            )

        return results

    def get_all_metadata(self) -> list[dict[str, Any]]:
        """Return all stored chunk metadata."""
        return [
            dict(entry)
            for entry in self._metadata
        ]

    def delete_document(self, source: str) -> None:
        """Remove all chunks belonging to a source and rebuild the index."""
        if not source:
            return

        remaining = [
            entry
            for entry in self._metadata
            if entry.get("source") != source
        ]

        # Nothing to delete.
        if len(remaining) == len(self._metadata):
            return

        self._metadata = remaining

        self._rebuild_index()

        self.save()

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self._index is None:
            self._index = self._create_empty_index()

        temp_index = self.index_path.with_suffix(
            ".faiss.tmp"
        )

        temp_metadata = self.metadata_path.with_suffix(
            ".json.tmp"
        )

        faiss.write_index(
            self._index,
            str(temp_index),
        )

        temp_metadata.write_text(
            json.dumps(
                self._metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temp_index.replace(
            self.index_path
        )

        temp_metadata.replace(
            self.metadata_path
        )

    def load(self) -> None:
        """Load the persisted FAISS index and metadata if available."""
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # No existing knowledge base.
        if (
            not self.index_path.exists()
            or not self.metadata_path.exists()
        ):
            self._metadata = []
            self._index = self._create_empty_index()
            return

        try:
            index = faiss.read_index(
                str(self.index_path)
            )

            with self.metadata_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                metadata = json.load(handle)

        except (
            OSError,
            ValueError,
            RuntimeError,
            json.JSONDecodeError,
        ):
            self._metadata = []
            self._index = self._create_empty_index()
            return

        if not isinstance(metadata, list):
            self._metadata = []
            self._index = self._create_empty_index()
            return

        if index is None:
            self._metadata = []
            self._index = self._create_empty_index()
            return

        # The persisted index must use the same embedding dimension.
        if index.d != self._dimension:
            self._metadata = []
            self._index = self._create_empty_index()
            return

        # FAISS vector positions must exactly match metadata positions.
        if index.ntotal != len(metadata):
            self._metadata = []
            self._index = self._create_empty_index()
            return

        self._metadata = metadata
        self._index = index


__all__ = ["VectorStore"]