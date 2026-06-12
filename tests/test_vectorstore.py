"""Tests for ChromaDB vector store."""

from pathlib import Path
import pytest
from langchain_core.documents import Document

from rag.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
    VectorStoreError,
    _sanitize_metadata,
)


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "vector_db",
        collection_name="test_collection",
    )
    vector_store.create_collection(reset=True)
    return vector_store


def test_sanitize_metadata_converts_unsupported_types() -> None:
    metadata = _sanitize_metadata({"page": 1, "source": "book.pdf", "extra": {"nested": True}})

    assert metadata == {"page": 1, "source": "book.pdf", "extra": "{'nested': True}"}


def test_create_collection(store: ChromaVectorStore) -> None:
    assert store.collection.name == "test_collection"
    assert store.collection.metadata == {"hnsw:space": "cosine"}


def test_add_documents_indexes_chunks(store: ChromaVectorStore) -> None:
    chunks = [
        Document(page_content="attention mechanism", metadata={"page": 0, "source": "paper.pdf"}),
        Document(page_content="transformer architecture", metadata={"page": 1, "source": "paper.pdf"}),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    indexed = store.add_documents(chunks, embeddings)

    assert indexed == 2
    assert store.collection.count() == 2


def test_add_documents_count_mismatch_raises(store: ChromaVectorStore) -> None:
    chunks = [Document(page_content="only one", metadata={})]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(VectorStoreError, match="count mismatch"):
        store.add_documents(chunks, embeddings)


def test_retrieve_returns_similar_documents(store: ChromaVectorStore) -> None:
    chunks = [
        Document(page_content="attention mechanism", metadata={"page": 0}),
        Document(page_content="transformer architecture", metadata={"page": 1}),
        Document(page_content="unrelated cooking recipe", metadata={"page": 2}),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ]
    store.add_documents(chunks, embeddings)

    results = store.retrieve([1.0, 0.0, 0.0], k=2)

    assert len(results) == 2
    assert results[0].page_content == "attention mechanism"
    assert "distance" in results[0].metadata


def test_persist_creates_directory(store: ChromaVectorStore) -> None:
    store.persist()
    assert store.persist_directory.exists()


def test_load_collection_after_restart(tmp_path: Path) -> None:
    persist_dir = tmp_path / "vector_db"

    first_store = ChromaVectorStore(persist_directory=persist_dir, collection_name="persisted")
    first_store.create_collection(reset=True)
    first_store.add_documents(
        [Document(page_content="saved chunk", metadata={"page": 0})],
        [[1.0, 0.0, 0.0]],
    )
    first_store.persist()

    second_store = ChromaVectorStore(persist_directory=persist_dir, collection_name="persisted")
    second_store.load_collection()

    assert second_store.collection.count() == 1
    results = second_store.retrieve([1.0, 0.0, 0.0], k=1)
    assert results[0].page_content == "saved chunk"


def test_retrieve_without_collection_raises(tmp_path: Path) -> None:
    store = ChromaVectorStore(persist_directory=tmp_path / "vector_db")

    with pytest.raises(VectorStoreError, match="Collection not initialized"):
        store.retrieve([1.0, 0.0, 0.0])


def test_default_collection_name() -> None:
    assert DEFAULT_COLLECTION_NAME == "book_research"
