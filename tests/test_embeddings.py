"""Tests for chunk embedding generation."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.embeddings import (
    BGE_QUERY_PREFIX,
    BGE_SMALL_DIMENSION,
    DEFAULT_MODEL_NAME,
    ChunkEmbedder,
    EmbeddingError,
)


@pytest.fixture
def mock_inference_client() -> MagicMock:
    client = MagicMock()
    client.feature_extraction.side_effect = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    return client


@pytest.fixture
def embedder(mock_inference_client: MagicMock) -> ChunkEmbedder:
    with patch("rag.embeddings.InferenceClient", return_value=mock_inference_client):
        yield ChunkEmbedder(api_token="hf_test_token")


def test_default_model_name() -> None:
    assert DEFAULT_MODEL_NAME == "BAAI/bge-small-en-v1.5"


def test_embed_texts_returns_vectors(
    embedder: ChunkEmbedder,
    mock_inference_client: MagicMock,
) -> None:
    result = embedder.embed_texts(["first chunk", "second chunk"], normalize=False)

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert mock_inference_client.feature_extraction.call_count == 2


def test_embed_texts_empty_input(embedder: ChunkEmbedder) -> None:
    assert embedder.embed_texts([]) == []


def test_embed_chunks_uses_page_content(
    embedder: ChunkEmbedder,
    mock_inference_client: MagicMock,
) -> None:
    chunks = [
        Document(page_content="chunk one", metadata={"page": 0}),
        Document(page_content="chunk two", metadata={"page": 1}),
    ]

    result = embedder.embed_chunks(chunks, normalize=False)

    calls = mock_inference_client.feature_extraction.call_args_list
    assert calls[0].kwargs["model"] == DEFAULT_MODEL_NAME
    assert calls[0].args[0] == "chunk one"
    assert calls[1].args[0] == "chunk two"
    assert len(result) == 2


def test_embed_query_adds_bge_prefix(
    mock_inference_client: MagicMock,
) -> None:
    mock_inference_client.feature_extraction.return_value = [0.9, 0.8, 0.7]
    mock_inference_client.feature_extraction.side_effect = None
    with patch("rag.embeddings.InferenceClient", return_value=mock_inference_client):
        embedder = ChunkEmbedder(api_token="hf_test_token")
        result = embedder.embed_query("what is attention?", normalize=False)

    mock_inference_client.feature_extraction.assert_called_once_with(
        f"{BGE_QUERY_PREFIX}what is attention?",
        model=DEFAULT_MODEL_NAME,
    )
    assert result == [0.9, 0.8, 0.7]


def test_embed_query_empty_raises(embedder: ChunkEmbedder) -> None:
    with pytest.raises(EmbeddingError, match="Query text cannot be empty"):
        embedder.embed_query("   ")


def test_embedding_dimension(embedder: ChunkEmbedder) -> None:
    assert embedder.embedding_dimension == BGE_SMALL_DIMENSION


def test_api_failure_raises() -> None:
    mock_client = MagicMock()
    mock_client.feature_extraction.side_effect = RuntimeError("api down")
    with patch("rag.embeddings.InferenceClient", return_value=mock_client):
        embedder = ChunkEmbedder(api_token="hf_test_token")
        with pytest.raises(EmbeddingError, match="Failed to generate embeddings"):
            embedder.embed_texts(["test"], normalize=False)
