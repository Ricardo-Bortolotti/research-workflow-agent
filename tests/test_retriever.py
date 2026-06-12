"""Tests for RAG document retrieval."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag.retriever import DEFAULT_TOP_K, DocumentRetriever, RetrieverError


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.retrieve.return_value = [
        Document(page_content="attention mechanism", metadata={"page": 0, "distance": 0.1}),
        Document(page_content="transformer block", metadata={"page": 1, "distance": 0.2}),
    ]
    return store


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def retriever(mock_vector_store: MagicMock, mock_embedder: MagicMock) -> DocumentRetriever:
    return DocumentRetriever(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        top_k=3,
    )


def test_default_top_k() -> None:
    assert DEFAULT_TOP_K == 4


def test_invoke_embeds_query_and_retrieves_chunks(
    retriever: DocumentRetriever,
    mock_vector_store: MagicMock,
    mock_embedder: MagicMock,
) -> None:
    results = retriever.invoke("What is attention?")

    mock_embedder.embed_query.assert_called_once_with("What is attention?")
    mock_vector_store.retrieve.assert_called_once_with([0.1, 0.2, 0.3], k=3)
    assert len(results) == 2
    assert results[0].page_content == "attention mechanism"


def test_invoke_allows_top_k_override(
    retriever: DocumentRetriever,
    mock_vector_store: MagicMock,
) -> None:
    retriever.invoke("Explain transformers", top_k=5)

    mock_vector_store.retrieve.assert_called_once_with([0.1, 0.2, 0.3], k=5)


def test_retrieve_is_alias_for_invoke(retriever: DocumentRetriever) -> None:
    assert retriever.retrieve("test query") == retriever.invoke("test query")


def test_empty_query_raises(retriever: DocumentRetriever) -> None:
    with pytest.raises(RetrieverError, match="Query text cannot be empty"):
        retriever.invoke("   ")


def test_invalid_top_k_on_init(mock_vector_store: MagicMock) -> None:
    with pytest.raises(RetrieverError, match="top_k must be positive"):
        DocumentRetriever(vector_store=mock_vector_store, top_k=0)


def test_invalid_top_k_on_invoke(retriever: DocumentRetriever) -> None:
    with pytest.raises(RetrieverError, match="top_k must be positive"):
        retriever.invoke("valid query", top_k=-1)
