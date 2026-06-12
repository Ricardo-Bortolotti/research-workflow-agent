"""Tests for chunk embedding generation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from rag.embeddings import (
    BGE_QUERY_PREFIX,
    DEFAULT_MODEL_NAME,
    ChunkEmbedder,
    EmbeddingError,
)


@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    model = MagicMock()
    model.get_embedding_dimension.return_value = 384
    model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return model


@pytest.fixture
def embedder(mock_sentence_transformer: MagicMock) -> ChunkEmbedder:
    with patch("rag.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        yield ChunkEmbedder()


def test_default_model_name() -> None:
    assert DEFAULT_MODEL_NAME == "BAAI/bge-small-en-v1.5"


def test_embed_texts_returns_vectors(embedder: ChunkEmbedder) -> None:
    result = embedder.embed_texts(["first chunk", "second chunk"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embed_texts_empty_input(embedder: ChunkEmbedder) -> None:
    assert embedder.embed_texts([]) == []


def test_embed_chunks_uses_page_content(
    embedder: ChunkEmbedder,
    mock_sentence_transformer: MagicMock,
) -> None:
    chunks = [
        Document(page_content="chunk one", metadata={"page": 0}),
        Document(page_content="chunk two", metadata={"page": 1}),
    ]

    result = embedder.embed_chunks(chunks)

    mock_sentence_transformer.encode.assert_called_once_with(
        ["chunk one", "chunk two"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    assert len(result) == 2


def test_embed_query_adds_bge_prefix(
    embedder: ChunkEmbedder,
    mock_sentence_transformer: MagicMock,
) -> None:
    mock_sentence_transformer.encode.return_value = np.array([[0.9, 0.8, 0.7]])

    result = embedder.embed_query("what is attention?")

    mock_sentence_transformer.encode.assert_called_once_with(
        [f"{BGE_QUERY_PREFIX}what is attention?"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    assert result == [0.9, 0.8, 0.7]


def test_embed_query_empty_raises(embedder: ChunkEmbedder) -> None:
    with pytest.raises(EmbeddingError, match="Query text cannot be empty"):
        embedder.embed_query("   ")


def test_embedding_dimension(embedder: ChunkEmbedder) -> None:
    assert embedder.embedding_dimension == 384


def test_model_load_failure_raises() -> None:
    with patch(
        "rag.embeddings.SentenceTransformer",
        side_effect=RuntimeError("download failed"),
    ):
        embedder = ChunkEmbedder()
        with pytest.raises(EmbeddingError, match="Failed to load model"):
            embedder.embed_texts(["test"])
