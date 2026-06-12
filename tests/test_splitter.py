"""Tests for document text splitting."""

import pytest
from langchain_core.documents import Document

from rag.splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SplitConfigError,
    create_text_splitter,
    split_documents,
)


def test_split_documents_produces_chunks() -> None:
    text = " ".join(["word"] * 500)
    documents = [Document(page_content=text, metadata={"source": "test.pdf", "page": 0})]

    chunks = split_documents(documents, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_split_documents_preserves_metadata() -> None:
    documents = [
        Document(
            page_content="A" * 300,
            metadata={"source": "test.pdf", "page": 2},
        )
    ]

    chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)

    assert all(chunk.metadata["source"] == "test.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)


def test_split_documents_empty_input() -> None:
    assert split_documents([]) == []


def test_split_documents_uses_defaults() -> None:
    splitter = create_text_splitter()
    assert splitter._chunk_size == DEFAULT_CHUNK_SIZE
    assert splitter._chunk_overlap == DEFAULT_CHUNK_OVERLAP


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-100, 50),
        (100, -10),
        (100, 100),
        (100, 150),
    ],
)
def test_invalid_split_config_raises(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(SplitConfigError):
        create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
